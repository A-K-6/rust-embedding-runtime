#![forbid(unsafe_code)]
use anyhow::{bail, Context, Result};
use rust_embedding_runtime::{
    server::{router, Service},
    QuantMode, Runtime,
};
use std::{net::SocketAddr, path::PathBuf};

fn main() -> Result<()> {
    std::env::set_var("TOKENIZERS_PARALLELISM", "false");
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?
        .block_on(run())
}
async fn run() -> Result<()> {
    if std::env::args().skip(1).any(|a| a == "--help" || a == "-h") {
        println!("embedding-runtime\nConfiguration: MODEL_DIR=./model LISTEN_ADDR=0.0.0.0:8080 THREADS=2 QUANTIZATION=f32 API_KEY=<optional>");
        return Ok(());
    }
    if std::env::args().len() > 1 {
        bail!("unknown argument; use --help");
    }
    let threads: usize = std::env::var("THREADS")
        .unwrap_or_else(|_| "2".into())
        .parse()
        .context("invalid THREADS")?;
    if !(1..=64).contains(&threads) {
        bail!("THREADS must be in 1..=64");
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    // Tokenization stays serial; parallel kernels use the configured Rayon pool.
    let address: SocketAddr = std::env::var("LISTEN_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:8080".into())
        .parse()?;
    let model = PathBuf::from(std::env::var("MODEL_DIR").unwrap_or_else(|_| "model".into()));
    let mode = QuantMode::parse(&std::env::var("QUANTIZATION").unwrap_or_else(|_| "f32".into()))?;
    let key = std::env::var("API_KEY").ok();
    if key.as_ref().is_some_and(|s| s.trim().is_empty()) {
        bail!("API_KEY must be nonempty when set");
    }
    let runtime = Runtime::load(&model, mode)?;
    let service = Service::new(runtime, key);
    let listener = tokio::net::TcpListener::bind(address).await?;
    eprintln!(
        "embedding-runtime ready on {address}; quantization={}; threads={threads}",
        mode.name()
    );
    axum::serve(listener, router(service))
        .with_graceful_shutdown(async {
            #[cfg(unix)]
            {
                let mut terminate =
                    tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                        .expect("signal handler");
                tokio::select! { _ = tokio::signal::ctrl_c() => {}, _ = terminate.recv() => {} }
            }
            #[cfg(not(unix))]
            {
                let _ = tokio::signal::ctrl_c().await;
            }
        })
        .await?;
    Ok(())
}
