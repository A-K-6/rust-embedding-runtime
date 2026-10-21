# syntax=docker/dockerfile:1
ARG TARGETARCH
FROM python:3.12-slim-bookworm AS python-tools

FROM python-tools AS model
WORKDIR /build
COPY scripts/download_model.py ./download_model.py
RUN python download_model.py --output /model && chmod -R a+rX /model

FROM gcc:14.3-bookworm@sha256:5e927c284bf55a7dc796262e311a0703344f62f41f5621eb56843111b1d37e15 AS aocl-amd64
# Python is a build tool only; both stages use the Debian 12 ABI.
COPY --from=python-tools /usr/local/ /usr/local/
COPY scripts/build_aocl.py /build_aocl.py
RUN apt-get update && apt-get install -y --no-install-recommends ninja-build \
      && rm -rf /var/lib/apt/lists/* && ldconfig \
      && python3 -m pip install --no-cache-dir cmake==4.1.2 \
      && python3 /build_aocl.py --output /opt/aocl --jobs 2 \
      && mkdir -p /opt/aocl/runtime \
      && for lib in libgomp.so.1 libstdc++.so.6 libatomic.so.1 libgcc_s.so.1; do \
           runtime="$(gcc -print-file-name=$lib)"; \
           case "$runtime" in /*) ;; *) exit 1 ;; esac; \
           test -f "$runtime" && cp -L "$runtime" "/opt/aocl/runtime/$lib" || exit 1; \
         done \
      && ln -s ../runtime/libgomp.so.1 /opt/aocl/lib/libgomp.so \
      && ln -s ../runtime/libstdc++.so.6 /opt/aocl/lib/libstdc++.so \
      && ln -s ../runtime/libatomic.so.1 /opt/aocl/lib/libatomic.so \
      && cp -L "$(gcc -print-libgcc-file-name)" /opt/aocl/lib/libgcc.a \
      && printf 'GROUP ( /opt/aocl/runtime/libgcc_s.so.1 -lgcc )\n' > /opt/aocl/lib/libgcc_s.so

FROM python-tools AS aocl-arm64
RUN mkdir -p /opt/aocl/lib /opt/aocl/include /opt/aocl/share/licenses /opt/aocl/runtime

FROM aocl-${TARGETARCH} AS aocl

FROM rust:1.98-bookworm AS build
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY .cargo ./.cargo
COPY src ./src
COPY crates ./crates
COPY --from=aocl /opt/aocl /opt/aocl
ENV AOCL_ROOT=/opt/aocl LD_LIBRARY_PATH=/opt/aocl/runtime
RUN RAYON_NUM_THREADS=2 cargo test --workspace --locked --features aocl \
    && cargo build --release --locked --features aocl --bin embedding-runtime

FROM gcr.io/distroless/cc-debian12:nonroot
ARG SOURCE_REVISION=unknown
LABEL org.opencontainers.image.source="https://github.com/A-K-6/rust-embedding-runtime" \
      org.opencontainers.image.title="Rust Embedding Runtime" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.revision=$SOURCE_REVISION
COPY --from=build /build/target/release/embedding-runtime /usr/local/bin/embedding-runtime
COPY --from=aocl /opt/aocl/runtime/ /opt/aocl-runtime/
COPY --from=model /model /opt/model
COPY LICENSE NOTICE THIRD_PARTY_LICENSES.txt /usr/share/licenses/rust-embedding-runtime/
COPY crates/aocl-kernels/licenses/ /usr/share/licenses/rust-embedding-runtime/backend/
ENV MODEL_DIR=/opt/model LISTEN_ADDR=0.0.0.0:8080 THREADS=2 QUANTIZATION=f32 \
    LD_LIBRARY_PATH=/opt/aocl-runtime
EXPOSE 8080
USER 65532:65532
ENTRYPOINT ["/usr/local/bin/embedding-runtime"]
