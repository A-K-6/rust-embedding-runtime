# aocl-kernels

Small checked Rust ownership layer for persistent **F32** linear weights using
AMD AOCL-DLP. Unsafe FFI is confined to this crate; the calling runtime can keep
`#![forbid(unsafe_code)]`.

```rust,no_run
use aocl_kernels::{available, PackedLinear};

# fn example() -> Result<(), Box<dyn std::error::Error>> {
if available() {
    // Original contiguous model weight layout: [output, input].
    let weights = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
    let linear = PackedLinear::new(&weights, 2, 3, 2)?;
    drop(weights); // Packing copies weights once; no source allocation retained.
    let output = linear.forward(&[1.0, 2.0, 3.0], 1)?;
    assert_eq!(output, vec![6.0, -12.0]);
}
# Ok(())
# }
```

`PackedLinear::new(weights, output, input, threads)` validates exact lengths,
dimensions 1–32768, checked element/byte counts, and threads 1–64.
`forward(input, rows)` accepts contiguous `[rows,input]`, rows 1–1024, and returns
owned `[rows,output]`. `input_size()`, `output_size()` and `storage_bytes()` expose
checked metadata. Errors implement `std::error::Error`.

One handle owns its aligned packed allocation and releases it with `Drop`.
A mutex serializes calls on the same handle; independent handles may run
separately. The opaque owned handle is `Send`; `Sync` comes from its mutex.
The C shim repeats pointer, size, dimension, and CPU feature checks and reports
upstream errors. Rust input/output borrows and separate output allocation prevent
aliasing. No raw pointers or packed storage escape the public API.

Linux x86_64 requires AVX2+FMA, with upstream runtime dispatch to suitable kernels.
`available()` returns false on unsupported CPUs and other operating systems or
architectures, allowing the runtime to choose its existing portable backend.
On non-Linux/x86_64 targets the crate builds a stub and needs no AOCL installation.

## Pinned external build

The repository's checked build helper is the simplest setup on Linux x86_64:

```sh
python3 scripts/build_aocl.py --output "$PWD/.aocl" --jobs 2
AOCL_ROOT="$PWD/.aocl" \
  cargo test --workspace --locked --features aocl
```

It needs Python 3.10+, GCC 13+, CMake 3.26+, Ninja and GNU OpenMP development
files. It checks the pinned archive hash, builds only the static library, and
saves compiler/source/artifact hashes in `.aocl/build-manifest.json`. The output
directory must be new. For compiler archives that support PIE, the optional
`--static-runtime` copies static GNU runtime archives and `AOCL_STATIC_RUNTIME=1`
links them into the Rust executable. Other builds use shared compiler runtimes.
The Docker build uses GCC 14.3 on Debian 12 for its runtime ABI and bundles
matching shared GNU libraries, preserving PIE even when `libgomp.a` is not PIC.

There is no download or network operation in `build.rs`. On Linux x86_64 set
`AOCL_ROOT` to an installed AOCL-DLP prefix with this contract:

```text
$AOCL_ROOT/include/aocl_dlp.h
$AOCL_ROOT/include/aocl_dlp_config.h
$AOCL_ROOT/include/classic/*.h
$AOCL_ROOT/lib/libaocl-dlp.a       # lib64/ is also accepted
```

Required upstream source:

- Repository: <https://github.com/amd/aocl-dlp>
- Release `5.3.2`, commit `fe36da707e6f25e8b3e2b742dfa5957d4484fca5`
- Archive: <https://codeload.github.com/amd/aocl-dlp/tar.gz/fe36da707e6f25e8b3e2b742dfa5957d4484fca5>
- Archive SHA256: `694d4464e790159a57c069c123132f2a7e8145c3d1182f449b838aa11ba4032b`

With CMake 3.26+ (preparation used 4.1.2), Ninja or Make, a C11/C++17 compiler,
and GCC OpenMP/libgomp:

```sh
cmake -S "$AOCL_SOURCE" -B "$AOCL_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$AOCL_ROOT" \
  -DDLP_THREADING_MODEL=openmp \
  -DBUILD_TESTING=OFF -DBUILD_BENCHMARKS=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_DOXYGEN=OFF -DBUILD_SPHINX=OFF
cmake --build "$AOCL_BUILD" --parallel 2
cmake --install "$AOCL_BUILD"
AOCL_ROOT="$AOCL_ROOT" cargo test --manifest-path crates/aocl-kernels/Cargo.toml
```

Leave `CFLAGS`/`CXXFLAGS` free of `-march=native` or other host-only instruction
flags. Upstream builds generic control code and ISA-specific kernel objects, then
dispatches at runtime. This crate uses pinned `cc = 1.9.0`, compiles its C shim
without host-specific flags, checks the header version/OpenMP configuration, and
links the static AOCL archive with `+whole-archive` to preserve JIT registration
constructors. Runtime dependencies are libgomp, libstdc++, libm, libgcc and libc;
atomic/pthread are linked where provided by the Linux toolchain. Rebuild AOCL
inside the container's builder for its ABI rather than copying host-built files.

## Runtime invariants and limits

- Packing uses row-major source `W[N,K]`, transpose `T`, matrix `B`; inference
  computes row-major `A[M,K] @ W.T`. The shim queries a byte count and allocates
  at 64-byte alignment before packing.
- Packed format depends on CPU/ISA/kernel context and K,N, and supports changing
  M without repacking. It is never persisted. Keep process ISA preferences fixed
  for the handle's lifetime; recreate weights on startup or hardware changes.
- Each call sets **thread-local** AOCL ways to unset and its thread count to the
  requested value. It checks the active count, overrides global/env defaults,
  and does not mutate process environment or global AOCL thread settings.
  These settings remain on the calling thread until the next AOCL configuration.
- Avoid outer parallel GEMM scheduling that multiplies AOCL's own OpenMP workers.
- Releasing original F32 linear weights after packing avoids retaining both
  representations. `storage_bytes()` reports the padded packed allocation only.
- Numeric tests cover odd/unaligned dimensions, M1,7,16,24,128,512,1024, source
  storage release, invalid dimensions/lengths/threads, and concurrent calls on
  shared and independent handles. These bounded checks do not establish broad
  concurrency, cross-CPU performance, full-model equivalence, or a speedup.

## Licenses

This wrapper is Apache-2.0 under the repository license. Upstream AMD AOCL-DLP
and its vendored Xbyak runtime JIT assembler use BSD-3-Clause. Preserve
[AOCL-DLP-LICENSE.txt](licenses/AOCL-DLP-LICENSE.txt) and
[AOCL-DLP-NOTICES.txt](licenses/AOCL-DLP-NOTICES.txt) in binary distributions.
Upstream also lists optional documentation dependencies under BSD-2-Clause and
BSD-3-Clause; documentation components are disabled in the build above.
Static GNU runtime builds also carry the GCC Runtime Library Exception and
GPLv3 notices under `licenses/`; see the repository's `NOTICE` for source links.
