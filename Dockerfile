# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm AS model
WORKDIR /build
COPY scripts/download_model.py ./download_model.py
RUN python download_model.py --output /model && chmod -R a+rX /model

FROM gcc:14.3-bookworm@sha256:5e927c284bf55a7dc796262e311a0703344f62f41f5621eb56843111b1d37e15 AS aocl
ARG TARGETARCH
# Python is a build tool only; both stages use the Debian 12 ABI.
COPY --from=model /usr/local/ /usr/local/
COPY scripts/build_aocl.py /build_aocl.py
RUN if [ "$TARGETARCH" = "amd64" ]; then \
      apt-get update && apt-get install -y --no-install-recommends ninja-build \
      && rm -rf /var/lib/apt/lists/* && ldconfig \
      && python3 -m pip install --no-cache-dir cmake==4.1.2 \
      && python3 /build_aocl.py --output /opt/aocl --jobs 2 --static-runtime; \
    else mkdir -p /opt/aocl/lib /opt/aocl/include /opt/aocl/share/licenses; fi

FROM rust:1.98-bookworm AS build
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY .cargo ./.cargo
COPY src ./src
COPY crates ./crates
COPY --from=aocl /opt/aocl /opt/aocl
ENV AOCL_ROOT=/opt/aocl AOCL_STATIC_RUNTIME=1
RUN RAYON_NUM_THREADS=2 cargo test --workspace --locked --features aocl \
    && cargo build --release --locked --features aocl --bin embedding-runtime

FROM gcr.io/distroless/cc-debian12:nonroot
ARG SOURCE_REVISION=unknown
LABEL org.opencontainers.image.source="https://github.com/A-K-6/rust-embedding-runtime" \
      org.opencontainers.image.title="Rust Embedding Runtime" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.revision=$SOURCE_REVISION
COPY --from=build /build/target/release/embedding-runtime /usr/local/bin/embedding-runtime
COPY --from=model /model /opt/model
COPY LICENSE NOTICE THIRD_PARTY_LICENSES.txt /usr/share/licenses/rust-embedding-runtime/
COPY crates/aocl-kernels/licenses/ /usr/share/licenses/rust-embedding-runtime/backend/
ENV MODEL_DIR=/opt/model LISTEN_ADDR=0.0.0.0:8080 THREADS=2 QUANTIZATION=f32
EXPOSE 8080
USER 65532:65532
ENTRYPOINT ["/usr/local/bin/embedding-runtime"]
