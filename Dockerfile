# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm AS model
WORKDIR /build
COPY scripts/download_model.py ./download_model.py
RUN python download_model.py --output /model && chmod -R a+rX /model

FROM rust:1.91-bookworm AS build
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release --locked --bin embedding-runtime

FROM gcr.io/distroless/cc-debian12:nonroot
ARG SOURCE_REVISION=unknown
LABEL org.opencontainers.image.source="https://github.com/A-K-6/rust-embedding-runtime" \
      org.opencontainers.image.title="Rust Embedding Runtime" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.revision=$SOURCE_REVISION
COPY --from=build /build/target/release/embedding-runtime /usr/local/bin/embedding-runtime
COPY --from=model /model /opt/model
COPY LICENSE NOTICE THIRD_PARTY_LICENSES.txt /usr/share/licenses/rust-embedding-runtime/
ENV MODEL_DIR=/opt/model LISTEN_ADDR=0.0.0.0:8080 THREADS=2 QUANTIZATION=f32
EXPOSE 8080
USER 65532:65532
ENTRYPOINT ["/usr/local/bin/embedding-runtime"]
