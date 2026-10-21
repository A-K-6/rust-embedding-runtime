// Community text-only EmbeddingGemma 2 encoder.
// Modified: extracted as a reusable library with a bounded HTTP service.
// Architecture adapted from Hugging Face Transformers (Apache-2.0).
// This crate forbids unsafe code; dependencies require a separate safety audit.
#![forbid(unsafe_code)]
use anyhow::{bail, Context, Result};
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{Module, VarBuilder};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, Read, Seek, SeekFrom, Write},
    path::Path,
    time::Instant,
};
use tokenizers::Tokenizer;
const MAX_TOKENS: usize = 1024;
const MAX_INPUT_BYTES: usize = 65536;
const MAX_HEADER_BYTES: u64 = 4 * 1024 * 1024;
const VOCABULARY: usize = 262144;

#[derive(Deserialize)]
struct Weight {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

// Read only text tensors. Avoid mapping/loading the multimodal weight file.
fn load_weights(path: &Path) -> Result<HashMap<String, Tensor>> {
    let mut f = File::open(path)?;
    let length = f.metadata()?.len();
    let mut n = [0u8; 8];
    f.read_exact(&mut n)?;
    let hlen = u64::from_le_bytes(n);
    if hlen > MAX_HEADER_BYTES || hlen.checked_add(8).is_none_or(|v| v > length) {
        bail!("invalid safetensors header size");
    }
    let mut header = vec![0u8; hlen as usize];
    f.read_exact(&mut header)?;
    let entries: HashMap<String, serde_json::Value> = serde_json::from_slice(&header)?;
    let base = hlen + 8;
    let mut tensors = HashMap::new();
    let mut loaded_bytes = 0usize;
    for (name, value) in entries {
        if !name.starts_with("language_model.") {
            continue;
        }
        let w: Weight = serde_json::from_value(value)?;
        let dtype = match w.dtype.as_str() {
            "BF16" => DType::BF16,
            "F32" => DType::F32,
            _ => bail!("unsupported weight dtype"),
        };
        if w.shape.len() > 4 || w.shape.contains(&0) {
            bail!("invalid weight shape");
        }
        let elements = w
            .shape
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .context("shape overflow")?;
        let expected = elements
            .checked_mul(dtype.size_in_bytes())
            .context("weight byte overflow")?;
        let resident = elements
            .checked_mul(if name == "language_model.embed_tokens.weight" {
                dtype.size_in_bytes()
            } else {
                4
            })
            .context("resident byte overflow")?;
        loaded_bytes = loaded_bytes
            .checked_add(resident)
            .context("total weight overflow")?;
        if loaded_bytes > 1200 * 1024 * 1024 {
            bail!("text weight budget exceeded");
        }
        let [start, end] = w.data_offsets;
        if end < start
            || end - start != expected as u64
            || base.checked_add(end).is_none_or(|v| v > length)
            || expected > 600 * 1024 * 1024
        {
            bail!("invalid weight range");
        }
        f.seek(SeekFrom::Start(base + start))?;
        let mut bytes = vec![0u8; expected];
        f.read_exact(&mut bytes)?;
        let tensor = Tensor::from_raw_buffer(&bytes, dtype, &w.shape, &Device::Cpu)?;
        // Keep the large vocabulary table in its original BF16 representation.
        // Convert only gathered token rows to F32 during a request.
        let tensor = if name == "language_model.embed_tokens.weight" {
            tensor
        } else {
            tensor.to_dtype(DType::F32)?
        };
        tensors.insert(name, tensor);
    }
    Ok(tensors)
}

#[derive(Clone)]
struct Norm {
    scale: Option<Tensor>,
    eps: f64,
}
impl Norm {
    fn new(vb: VarBuilder, size: usize, scale: bool) -> candle_core::Result<Self> {
        Ok(Self {
            scale: if scale {
                Some(vb.get(size, "weight")?)
            } else {
                None
            },
            eps: 1e-6,
        })
    }
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let denom = (x.sqr()?.mean_keepdim(D::Minus1)? + self.eps)?.sqrt()?;
        let y = x.broadcast_div(&denom)?;
        match &self.scale {
            Some(w) => y.broadcast_mul(w),
            None => Ok(y),
        }
    }
}

// Optional weight quantization. The mode is explicit, never read from the
// process environment, so a single process cannot change representation
// mid-request and the CLI can reject unknown values before loading weights.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum QuantMode {
    #[default]
    F32,
    #[allow(non_camel_case_types)]
    Q8_0,
    #[allow(non_camel_case_types)]
    Q4_K,
}

impl QuantMode {
    // Exact tokens, case-insensitive and untrimmed: no silent default mode.
    pub fn parse(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "f32" => Ok(Self::F32),
            "q8_0" => Ok(Self::Q8_0),
            "q4_k" => Ok(Self::Q4_K),
            _ => bail!("unsupported quant mode {value:?}; expected f32, q8_0 or q4_k"),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Q8_0 => "q8_0",
            Self::Q4_K => "q4_k",
        }
    }

    fn ggml_dtype(self) -> Option<GgmlDType> {
        match self {
            Self::F32 => None,
            Self::Q8_0 => Some(GgmlDType::Q8_0),
            Self::Q4_K => Some(GgmlDType::Q4K),
        }
    }
}

// A bias-free linear layer whose weight is either resident float32 or a packed
// Candle QTensor. Quantized modes run the real quantized matmul kernel, they do
// not dequantize the weight into a float tensor.
enum ProjectionWeight {
    Float(Tensor),
    Packed(std::sync::Arc<QTensor>, QMatMul),
}

struct QuantProjection {
    // Kept for inspection; the packed storage itself is the authority.
    #[cfg_attr(not(test), allow(dead_code))]
    mode: QuantMode,
    weight: ProjectionWeight,
}

impl QuantProjection {
    fn new(weight: &Tensor, mode: QuantMode) -> candle_core::Result<Self> {
        let weight = match mode.ggml_dtype() {
            None => ProjectionWeight::Float(weight.to_dtype(DType::F32)?),
            Some(dtype) => {
                let packed = std::sync::Arc::new(QTensor::quantize(weight, dtype)?);
                let matmul = QMatMul::QTensor(packed.clone());
                ProjectionWeight::Packed(packed, matmul)
            }
        };
        Ok(Self { mode, weight })
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn mode(&self) -> QuantMode {
        self.mode
    }

    fn is_quantized(&self) -> bool {
        matches!(&self.weight, ProjectionWeight::Packed(..))
    }

    // Actual resident weight bytes, not the conceptual float size.
    fn storage_bytes(&self) -> usize {
        match &self.weight {
            ProjectionWeight::Float(w) => w.elem_count() * w.dtype().size_in_bytes(),
            ProjectionWeight::Packed(packed, _) => packed.storage_size_in_bytes(),
        }
    }
}

impl Module for QuantProjection {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match &self.weight {
            ProjectionWeight::Float(w) => candle_nn::Linear::new(w.clone(), None).forward(x),
            // The quantized CPU kernel only accepts contiguous float32 input.
            ProjectionWeight::Packed(_, matmul) => matmul.forward(&x.contiguous()?),
        }
    }
}

struct WeightLoader<'a> {
    weights: &'a mut HashMap<String, Tensor>,
    mode: QuantMode,
    total_bytes: usize,
    packed_bytes: usize,
}

impl WeightLoader<'_> {
    fn projection(&mut self, name: &str, output: usize, input: usize) -> Result<QuantProjection> {
        self.build(name, output, input, self.mode)
    }

    // Kept in float32 regardless of the selected mode.
    fn float_projection(
        &mut self,
        name: &str,
        output: usize,
        input: usize,
    ) -> Result<QuantProjection> {
        self.build(name, output, input, QuantMode::F32)
    }

    fn build(
        &mut self,
        name: &str,
        output: usize,
        input: usize,
        mode: QuantMode,
    ) -> Result<QuantProjection> {
        let weight = self
            .weights
            .remove(name)
            .with_context(|| format!("missing weight {name}"))?;
        if weight.dims() != [output, input] {
            bail!("unsupported weight shape for {name}");
        }
        let projection = QuantProjection::new(&weight, mode)
            .with_context(|| format!("cannot build linear layer {name}"))?;
        self.total_bytes += projection.storage_bytes();
        if projection.is_quantized() {
            self.packed_bytes += projection.storage_bytes();
        }
        Ok(projection)
    }
}

const LAYERS: usize = 24;
const HIDDEN: usize = 512;

fn is_full_layer(i: usize) -> bool {
    i % 6 == 5
}
fn layer_head_dim(i: usize) -> usize {
    if is_full_layer(i) {
        512
    } else {
        256
    }
}
fn layer_kv_heads(i: usize) -> usize {
    if is_full_layer(i) {
        1
    } else {
        2
    }
}

fn layer_projection(
    loader: &mut WeightLoader,
    layer: usize,
    name: &str,
    output: usize,
    input: usize,
) -> Result<QuantProjection> {
    loader.projection(
        &format!("language_model.layers.{layer}.{name}.weight"),
        output,
        input,
    )
}

// Layer linears are built before the remaining float32 parameters are handed to
// a VarBuilder, so quantized modes never hold a second float copy of a linear
// weight during loading.
struct LayerLinears {
    q: QuantProjection,
    k: QuantProjection,
    v: QuantProjection,
    o: QuantProjection,
    gate: QuantProjection,
    up: QuantProjection,
    down: QuantProjection,
    ple_gate: QuantProjection,
    ple_proj: QuantProjection,
}

impl LayerLinears {
    fn load(loader: &mut WeightLoader, i: usize) -> Result<Self> {
        let hd = layer_head_dim(i);
        let kv = layer_kv_heads(i);
        Ok(Self {
            q: layer_projection(loader, i, "self_attn.q_proj", 4 * hd, HIDDEN)?,
            k: layer_projection(loader, i, "self_attn.k_proj", kv * hd, HIDDEN)?,
            v: layer_projection(loader, i, "self_attn.v_proj", kv * hd, HIDDEN)?,
            o: layer_projection(loader, i, "self_attn.o_proj", HIDDEN, 4 * hd)?,
            gate: layer_projection(loader, i, "mlp.gate_proj", 2048, HIDDEN)?,
            up: layer_projection(loader, i, "mlp.up_proj", 2048, HIDDEN)?,
            down: layer_projection(loader, i, "mlp.down_proj", HIDDEN, 2048)?,
            ple_gate: layer_projection(
                loader,
                i,
                "ple_block.per_layer_input_gate",
                HIDDEN,
                HIDDEN,
            )?,
            ple_proj: layer_projection(
                loader,
                i,
                "ple_block.per_layer_projection",
                HIDDEN,
                HIDDEN,
            )?,
        })
    }
}

struct Layer {
    q: QuantProjection,
    k: QuantProjection,
    v: QuantProjection,
    o: QuantProjection,
    qn: Norm,
    kn: Norm,
    vn: Norm,
    input: Norm,
    post_attn: Norm,
    pre_ff: Norm,
    post_ff: Norm,
    gate: QuantProjection,
    up: QuantProjection,
    down: QuantProjection,
    ple_gate: QuantProjection,
    ple_proj: QuantProjection,
    ple_norm: Norm,
    scalar: Tensor,
    head_dim: usize,
    kv_heads: usize,
    full: bool,
}
impl Layer {
    fn new(vb: VarBuilder, linears: LayerLinears, i: usize) -> Result<Self> {
        let full = is_full_layer(i);
        let hd = layer_head_dim(i);
        let kv = layer_kv_heads(i);
        let a = vb.pp("self_attn");
        let p = vb.pp("ple_block");
        Ok(Self {
            q: linears.q,
            k: linears.k,
            v: linears.v,
            o: linears.o,
            qn: Norm::new(a.pp("q_norm"), hd, true)?,
            kn: Norm::new(a.pp("k_norm"), hd, true)?,
            vn: Norm::new(a.pp("v_norm"), hd, false)?,
            input: Norm::new(vb.pp("input_layernorm"), HIDDEN, true)?,
            post_attn: Norm::new(vb.pp("post_attention_layernorm"), HIDDEN, true)?,
            pre_ff: Norm::new(vb.pp("pre_feedforward_layernorm"), HIDDEN, true)?,
            post_ff: Norm::new(vb.pp("post_feedforward_layernorm"), HIDDEN, true)?,
            gate: linears.gate,
            up: linears.up,
            down: linears.down,
            ple_gate: linears.ple_gate,
            ple_proj: linears.ple_proj,
            ple_norm: Norm::new(p.pp("post_per_layer_input_norm"), HIDDEN, true)?,
            scalar: vb.get(1, "layer_scalar")?,
            head_dim: hd,
            kv_heads: kv,
            full,
        })
    }
    fn forward(
        &self,
        x: &Tensor,
        ple: &Tensor,
        rope: &(Tensor, Tensor),
        mask: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let n = x.dim(1)?;
        let h = self.input.forward(x)?;
        let q = self.qn.forward(
            &self
                .q
                .forward(&h)?
                .reshape((1, n, 4, self.head_dim))?
                .transpose(1, 2)?,
        )?;
        let k = self.kn.forward(
            &self
                .k
                .forward(&h)?
                .reshape((1, n, self.kv_heads, self.head_dim))?
                .transpose(1, 2)?,
        )?;
        let v = self.vn.forward(
            &self
                .v
                .forward(&h)?
                .reshape((1, n, self.kv_heads, self.head_dim))?
                .transpose(1, 2)?,
        )?;
        let rotate = |t: &Tensor| -> candle_core::Result<Tensor> {
            let left = t.narrow(3, 0, self.head_dim / 2)?;
            let right = t.narrow(3, self.head_dim / 2, self.head_dim / 2)?.neg()?;
            let rotated = Tensor::cat(&[&right, &left], 3)?;
            t.broadcast_mul(&rope.0)?
                .add(&rotated.broadcast_mul(&rope.1)?)
        };
        let q = rotate(&q)?.contiguous()?;
        let k = rotate(&k)?;
        let repeat = |t: &Tensor| -> candle_core::Result<Tensor> {
            t.unsqueeze(2)?
                .broadcast_as((1, self.kv_heads, 4 / self.kv_heads, n, self.head_dim))?
                .contiguous()?
                .reshape((1, 4, n, self.head_dim))
        };
        let k = repeat(&k)?;
        let v = repeat(&v)?;
        // Reference attention scaling is 1.0, not 1/sqrt(head_dim).
        let scores = q
            .matmul(&k.transpose(2, 3)?.contiguous()?)?
            .broadcast_add(mask)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let attention =
            probs
                .matmul(&v)?
                .transpose(1, 2)?
                .contiguous()?
                .reshape((1, n, 4 * self.head_dim))?;
        let h = x.add(&self.post_attn.forward(&self.o.forward(&attention)?)?)?;
        let f = self.pre_ff.forward(&h)?;
        let f = self.gate.forward(&f)?.gelu()?.mul(&self.up.forward(&f)?)?;
        let h = h.add(&self.post_ff.forward(&self.down.forward(&f)?)?)?;
        let p = self.ple_gate.forward(&h)?.gelu()?.mul(ple)?;
        h.add(&self.ple_norm.forward(&self.ple_proj.forward(&p)?)?)?
            .broadcast_mul(&self.scalar)
    }
}

struct Encoder {
    embed: Tensor,
    ple: QuantProjection,
    ple_norm: Norm,
    layers: Vec<Layer>,
    norm: Norm,
    projection: QuantProjection,
    total_weight_bytes: usize,
    packed_weight_bytes: usize,
}
impl Encoder {
    fn load(dir: &Path, mode: QuantMode) -> Result<Self> {
        let config: serde_json::Value =
            serde_json::from_reader(File::open(dir.join("config.json"))?)?;
        let c = &config["text_config"];
        if c["hidden_size"] != 512
            || c["num_hidden_layers"] != 24
            || c["embedding_dim"] != 768
            || c["sliding_window"] != 512
        {
            bail!("unsupported model architecture");
        }
        let mut weights = load_weights(&dir.join("model.safetensors"))?;
        let embed = weights
            .remove("language_model.embed_tokens.weight")
            .context("missing vocabulary table")?;
        if embed.dims() != [VOCABULARY, 512] {
            bail!("unsupported vocabulary shape");
        }
        // Only linear weights are quantized. The token embedding table stays
        // BF16 and every norm/scalar stays float32.
        let mut loader = WeightLoader {
            weights: &mut weights,
            mode,
            total_bytes: 0,
            packed_bytes: 0,
        };
        let ple = loader.projection(
            "language_model.ple.per_layer_model_projection.weight",
            LAYERS * HIDDEN,
            HIDDEN,
        )?;
        // The final embedding projection always stays float32.
        let projection =
            loader.float_projection("language_model.embedding_projection.weight", 768, HIDDEN)?;
        let linears = (0..LAYERS)
            .map(|i| LayerLinears::load(&mut loader, i))
            .collect::<Result<Vec<_>>>()?;
        let total_weight_bytes = loader.total_bytes;
        let packed_weight_bytes = loader.packed_bytes;
        let vb = VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu).pp("language_model");
        let mut layers = Vec::with_capacity(LAYERS);
        for (i, linears) in linears.into_iter().enumerate() {
            layers.push(Layer::new(vb.pp(format!("layers.{i}")), linears, i)?);
        }
        Ok(Self {
            embed,
            ple,
            ple_norm: Norm::new(vb.pp("ple.per_layer_projection_norm"), HIDDEN, true)?,
            layers,
            norm: Norm::new(vb.pp("norm"), HIDDEN, true)?,
            projection,
            total_weight_bytes,
            packed_weight_bytes,
        })
    }
    fn weight_bytes(&self) -> usize {
        self.total_weight_bytes
    }
    fn packed_weight_bytes(&self) -> usize {
        self.packed_weight_bytes
    }
    fn encode(&self, ids: &[u32], dims: usize) -> Result<Vec<f32>> {
        validate_input(ids, dims)?;
        let n = ids.len();
        let t = Tensor::new(ids, &Device::Cpu)?;
        let mut h = (self
            .embed
            .index_select(&t, 0)?
            .to_dtype(DType::F32)?
            .unsqueeze(0)?
            * (512f64).sqrt())?;
        let ple = self
            .ple_norm
            .forward(&(self.ple.forward(&h)? * (512f64).powf(-0.5))?.reshape((1, n, 24, 512))?)?;
        let rope = |hd: usize, theta: f32| -> candle_core::Result<(Tensor, Tensor)> {
            let mut cos = Vec::with_capacity(n * hd);
            let mut sin = Vec::with_capacity(n * hd);
            for pos in 0..n {
                for d in 0..hd {
                    let angle = pos as f32 * theta.powf(-(((d % (hd / 2)) * 2) as f32) / hd as f32);
                    cos.push(angle.cos());
                    sin.push(angle.sin());
                }
            }
            Ok((
                Tensor::from_vec(cos, (1, 1, n, hd), &Device::Cpu)?,
                Tensor::from_vec(sin, (1, 1, n, hd), &Device::Cpu)?,
            ))
        };
        let sliding_rope = rope(256, 10000.)?;
        let full_rope = rope(512, 1000000.)?;
        let full_mask = Tensor::zeros((1, 1, n, n), DType::F32, &Device::Cpu)?;
        let mask: Vec<f32> = (0..n)
            .flat_map(|i| {
                (0..n).map(move |j| {
                    if i.abs_diff(j) <= 512 {
                        0.
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect();
        let sliding_mask = Tensor::from_vec(mask, (1, 1, n, n), &Device::Cpu)?;
        for (i, l) in self.layers.iter().enumerate() {
            h = l.forward(
                &h,
                &ple.narrow(2, i, 1)?.squeeze(2)?,
                if l.full { &full_rope } else { &sliding_rope },
                if l.full { &full_mask } else { &sliding_mask },
            )?;
        }
        let pooled = self.norm.forward(&h)?.mean(1)?;
        let projected = self
            .projection
            .forward(&pooled)?
            .squeeze(0)?
            .narrow(0, 0, dims)?
            .to_vec1::<f32>()?;
        let norm = projected.iter().map(|v| v * v).sum::<f32>().sqrt();
        if !norm.is_finite() || norm < 1e-12 {
            bail!("invalid embedding norm");
        }
        Ok(projected.into_iter().map(|v| v / norm).collect())
    }
}

fn validate_input(ids: &[u32], dims: usize) -> Result<()> {
    if ids.is_empty() || ids.len() > MAX_TOKENS {
        bail!("token count outside 1..={MAX_TOKENS}");
    }
    if ![128, 256, 512, 768].contains(&dims) {
        bail!("unsupported embedding dimension");
    }
    // Reject before passing indices into dependency numerical kernels.
    if ids.iter().any(|&id| id as usize >= VOCABULARY) {
        bail!("token id outside vocabulary");
    }
    Ok(())
}

// Bound buffering, drain an oversized frame, and preserve the next request.
fn read_frame(input: &mut impl BufRead) -> std::io::Result<Option<Result<Vec<u8>>>> {
    let mut bytes = Vec::new();
    if (&mut *input)
        .take((MAX_INPUT_BYTES + 1) as u64)
        .read_until(b'\n', &mut bytes)?
        == 0
    {
        return Ok(None);
    }
    if bytes.len() > MAX_INPUT_BYTES {
        if bytes.last() != Some(&b'\n') {
            loop {
                let mut discard = Vec::new();
                let n = (&mut *input).take(4096).read_until(b'\n', &mut discard)?;
                if n == 0 || discard.last() == Some(&b'\n') {
                    break;
                }
            }
        }
        return Ok(Some(Err(anyhow::anyhow!(
            "request exceeds {MAX_INPUT_BYTES} bytes"
        ))));
    }
    Ok(Some(Ok(bytes)))
}

#[derive(Deserialize)]
struct Request {
    text: Option<String>,
    ids: Option<Vec<u32>>,
    #[serde(default = "default_dims")]
    dims: usize,
}
fn default_dims() -> usize {
    768
}
#[derive(Serialize)]
struct Response {
    embedding: Vec<f32>,
    tokens: usize,
    elapsed_ms: f64,
}
const USAGE: &str = "usage: embedding-jsonl [--quant f32|q8_0|q4_k] MODEL_DIR";

// Reject every invalid or extra argument before any model weight is touched.
fn parse_args(args: &[String]) -> Result<(QuantMode, &Path)> {
    let mut mode = QuantMode::default();
    let mut quant_seen = false;
    let mut dir = None;
    let mut i = 1;
    while let Some(arg) = args.get(i) {
        i += 1;
        if arg == "--quant" {
            let value = args.get(i).with_context(|| USAGE.to_string())?;
            i += 1;
            if quant_seen {
                bail!("duplicate --quant\n{USAGE}");
            }
            mode = QuantMode::parse(value)?;
            quant_seen = true;
        } else if let Some(value) = arg.strip_prefix("--quant=") {
            if quant_seen {
                bail!("duplicate --quant\n{USAGE}");
            }
            mode = QuantMode::parse(value)?;
            quant_seen = true;
        } else if arg.starts_with('-') {
            bail!("unknown argument {arg:?}\n{USAGE}");
        } else if dir.replace(Path::new(arg)).is_some() {
            bail!("unexpected extra argument {arg:?}\n{USAGE}");
        }
    }
    let dir = dir.with_context(|| USAGE.to_string())?;
    Ok((mode, dir))
}

pub fn run_jsonl() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (mode, dir) = parse_args(&args)?;
    let start = Instant::now();
    verify_model(dir)?;
    let model = Encoder::load(dir, mode)?;
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    eprintln!(
        "{}",
        serde_json::json!({
            "ready": true,
            "threads": rayon::current_num_threads(),
            "cpu_backend": if candle_core::utils::has_accelerate() { "accelerate" } else { "gemm" },
            "load_ms": start.elapsed().as_secs_f64() * 1000.,
            "max_tokens": MAX_TOKENS,
            "dtype": "f32",
            "device": "cpu",
            "weight_quantization": mode.name(),
            "linear_weight_bytes": model.weight_bytes(),
            "packed_weight_bytes": model.packed_weight_bytes(),
        })
    );
    let stdin = std::io::stdin();
    let mut input = stdin.lock();
    let mut output = std::io::BufWriter::new(std::io::stdout());
    loop {
        let Some(frame) = read_frame(&mut input)? else {
            break;
        };
        let result = (|| -> Result<Response> {
            let bytes = frame?;
            let r: Request = serde_json::from_slice(&bytes)?;
            let ids = match (r.text, r.ids) {
                (Some(text), None) => {
                    let e = tokenizer
                        .encode(text, true)
                        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
                    e.get_ids().to_vec()
                }
                (None, Some(ids)) => ids,
                _ => bail!("provide exactly one of text or ids"),
            };
            let start = Instant::now();
            let embedding = model.encode(&ids, r.dims)?;
            Ok(Response {
                embedding,
                tokens: ids.len(),
                elapsed_ms: start.elapsed().as_secs_f64() * 1000.,
            })
        })();
        match result {
            Ok(r) => serde_json::to_writer(&mut output, &r)?,
            Err(e) => {
                serde_json::to_writer(&mut output, &serde_json::json!({"error":e.to_string()}))?
            }
        };
        output.write_all(b"\n")?;
        output.flush()?;
    }
    Ok(())
}

#[cfg(test)]
mod quantization_tests;

#[cfg(test)]
mod safety_tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn token_and_dimension_boundaries() {
        assert!(validate_input(&[0, (VOCABULARY - 1) as u32], 512).is_ok());
        assert!(validate_input(&vec![0; MAX_TOKENS], 768).is_ok());
        for ids in [
            vec![],
            vec![0; MAX_TOKENS + 1],
            vec![VOCABULARY as u32],
            vec![u32::MAX],
        ] {
            assert!(validate_input(&ids, 512).is_err());
        }
        for dims in [0, 127, 513, usize::MAX] {
            assert!(validate_input(&[0], dims).is_err());
        }
    }

    #[test]
    fn oversized_frames_recover_with_bounded_buffering() {
        let mut data = vec![b'x'; MAX_INPUT_BYTES * 20];
        data.extend_from_slice(b"\n{}\n");
        let mut input = Cursor::new(data);
        assert!(read_frame(&mut input).unwrap().unwrap().is_err());
        assert_eq!(read_frame(&mut input).unwrap().unwrap().unwrap(), b"{}\n");
        assert!(read_frame(&mut input).unwrap().is_none());
        let mut input = Cursor::new(vec![b'x'; MAX_INPUT_BYTES]);
        assert_eq!(
            read_frame(&mut input).unwrap().unwrap().unwrap().len(),
            MAX_INPUT_BYTES
        );
        let mut input = Cursor::new(vec![b'x'; MAX_INPUT_BYTES + 1]);
        assert!(read_frame(&mut input).unwrap().unwrap().is_err());
    }

    #[test]
    fn malformed_weights_fail_before_tensor_allocation() {
        let cases = [
            serde_json::json!({"dtype":"F32", "shape":[usize::MAX,2], "data_offsets":[0,0]}),
            serde_json::json!({"dtype":"F32", "shape":[0,512], "data_offsets":[0,0]}),
            serde_json::json!({"dtype":"F32", "shape":[1,1,1,1,1], "data_offsets":[0,4]}),
            serde_json::json!({"dtype":"F32", "shape":[1], "data_offsets":[4,0]}),
            serde_json::json!({"dtype":"F32", "shape":[1], "data_offsets":[u64::MAX-4,u64::MAX]}),
            serde_json::json!({"dtype":"F32", "shape":[1], "data_offsets":[0,2]}),
            serde_json::json!({"dtype":"I64", "shape":[1], "data_offsets":[0,8]}),
            serde_json::json!({"dtype":"F32", "shape":[400_000_000], "data_offsets":[0,1_600_000_000u64]}),
        ];
        for (i, weight) in cases.into_iter().enumerate() {
            let header =
                serde_json::to_vec(&serde_json::json!({"language_model.test":weight})).unwrap();
            assert!(try_weights(&format!("bad-{i}"), &header, Some(header.len() as u64)).is_err());
        }
        assert!(try_weights("too-large", b"{}", Some(MAX_HEADER_BYTES + 1)).is_err());
        assert!(try_weights("truncated", b"{}", Some(20)).is_err());
        assert!(try_weights("bad-json", b"bad-json", Some(8)).is_err());
        let header = serde_json::to_vec(&serde_json::json!({"language_model.test":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})).unwrap();
        let loaded = try_weights("valid", &header, Some(header.len() as u64)).unwrap();
        assert_eq!(
            loaded["language_model.test"].to_vec1::<f32>().unwrap(),
            vec![0.]
        );
    }

    fn try_weights(
        name: &str,
        header: &[u8],
        declared: Option<u64>,
    ) -> Result<HashMap<String, Tensor>> {
        let path =
            std::env::temp_dir().join(format!("embedding-weights-{}-{name}", std::process::id()));
        let mut bytes = declared.unwrap().to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[0; 8]);
        std::fs::write(&path, bytes)?;
        let result = load_weights(&path);
        std::fs::remove_file(path)?;
        result
    }
}

pub mod server;

/// The exact upstream checkpoint supported by this implementation.
pub const MODEL_REVISION: &str = "914f7f89142e33e77833254d9c9b90c3cef7303b";
pub const MODEL_ID: &str = "google/embeddinggemma-2";

/// Text or model-native token IDs. Token IDs must use this model's tokenizer.
#[derive(Clone, Debug)]
pub enum Input {
    Text(String),
    Tokens(Vec<u32>),
}

#[derive(Debug)]
pub(crate) struct InvalidInput(pub &'static str);
impl std::fmt::Display for InvalidInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for InvalidInput {}

/// Resident CPU encoder. Callers control serialization; the HTTP server uses one worker.
pub struct Runtime {
    encoder: Encoder,
    tokenizer: Tokenizer,
}

impl Runtime {
    /// Load the pinned text checkpoint after checking every bundled asset hash.
    pub fn load(dir: &Path, mode: QuantMode) -> Result<Self> {
        verify_model(dir)?;
        Ok(Self {
            encoder: Encoder::load(dir, mode)?,
            tokenizer: Tokenizer::from_file(dir.join("tokenizer.json"))
                .map_err(|_| anyhow::anyhow!("cannot load tokenizer"))?,
        })
    }

    /// Encode raw text without adding a task prefix, returning normalized vectors.
    /// The entire batch is validated before inference. Batch execution is sequential.
    pub fn embed(&mut self, inputs: &[Input], dims: usize) -> Result<(Vec<Vec<f32>>, usize)> {
        if inputs.is_empty() || inputs.len() > 16 {
            return Err(InvalidInput("Batch size must be between 1 and 16").into());
        }
        let ids: Vec<Vec<u32>> = inputs
            .iter()
            .map(|input| -> Result<_> {
                let ids = match input {
                    Input::Text(text) => {
                        if text.trim().is_empty() || text.len() > MAX_INPUT_BYTES {
                            return Err(InvalidInput("Text must be nonempty and at most 65536 bytes").into());
                        }
                        self.tokenizer
                            .encode(text.as_str(), true)
                            .map_err(|_| InvalidInput("Tokenization failed"))?
                            .get_ids()
                            .to_vec()
                    }
                    Input::Tokens(ids) => ids.clone(),
                };
                validate_input(&ids, dims).map_err(|_| InvalidInput("Each input must contain 1–1024 valid model tokens and use a supported dimension"))?;
                Ok(ids)
            })
            .collect::<Result<_>>()?;
        let tokens = ids.iter().map(Vec::len).sum();
        if tokens > 4096 {
            return Err(InvalidInput("Batch exceeds 4096 total tokens").into());
        }
        let vectors = ids
            .iter()
            .map(|ids| self.encoder.encode(ids, dims))
            .collect::<Result<_>>()?;
        Ok((vectors, tokens))
    }
}

/// Verify known hashes, independent of a caller-supplied manifest.
pub fn verify_model(dir: &Path) -> Result<()> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    for (name, expected) in [
        (
            "model.safetensors",
            "1535bba554267ed2eb95d24c9eab0b7d19a9855d347adcbd44b354e74c7e13d9",
        ),
        (
            "tokenizer.json",
            "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4",
        ),
        (
            "config.json",
            "b8f1e9931b57fbc054acdb445c41765d55b0074c58d145fa82839941ad1b5bb3",
        ),
    ] {
        let mut file =
            File::open(dir.join(name)).with_context(|| format!("missing model asset: {name}"))?;
        let mut hash = Sha256::new();
        let mut buf = [0u8; 65536];
        loop {
            let n = file.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hash.update(&buf[..n]);
        }
        if format!("{:x}", hash.finalize()) != expected {
            bail!("model checksum mismatch: {name}");
        }
    }
    Ok(())
}
