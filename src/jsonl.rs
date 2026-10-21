fn main() -> anyhow::Result<()> {
    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "2".into());
    std::env::set_var("RAYON_NUM_THREADS", &threads);
    std::env::set_var("VECLIB_MAXIMUM_THREADS", &threads);
    rust_embedding_runtime::run_jsonl()
}
