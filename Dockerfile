FROM rust:1.99.0-bookworm AS build
WORKDIR /build
COPY Cargo.toml Cargo.lock WEIGHTLANG_REVISION ./
COPY .vendor/weightlang .vendor/weightlang
COPY src src
RUN cargo build --locked --release && cargo build --locked --release --manifest-path .vendor/weightlang/Cargo.toml -p wgtime-service --target-dir /build/target
FROM debian:bookworm-slim@sha256:734a72b3d6142a71993b4bc580ca01d8ca687f77a2c422d96a961d3376b179bc
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev python3 ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=build /build/target/release/weight-init-bench /usr/local/bin/weight-init-bench
COPY --from=build /build/target/release/wgtime /usr/local/bin/wgtime
WORKDIR /app
