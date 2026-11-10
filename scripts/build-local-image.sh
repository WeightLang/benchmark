#!/usr/bin/env bash
# Reuse an installed official Rust toolchain inside Debian when registry pulls stall.
set -euo pipefail
cd "$(dirname "$0")/.."
toolchain=$(rustc +1.99.0 --print sysroot)
registry=${CARGO_HOME:-$HOME/.cargo}/registry
docker build -f Dockerfile.runtime -t weight-init-runtime:0.1.0 .
docker run --rm --network none --user "$(id -u):$(id -g)" \
    -v "$PWD:/work" -v "$toolchain:/opt/rust:ro" -v "$registry:/cargo/registry:ro" \
    -w /work -e CARGO_HOME=/cargo -e CARGO_TARGET_DIR=/work/target/bookworm \
    -e RUSTC=/opt/rust/bin/rustc --tmpfs /cargo:mode=1777 \
    weight-init-runtime:0.1.0 /bin/sh -c '
        /opt/rust/bin/cargo build --offline --locked --release &&
        /opt/rust/bin/cargo build --offline --locked --release --manifest-path .vendor/weightlang/Cargo.toml -p wgtime-service'
context=$(mktemp -d "$PWD/target/image.XXXXXX")
trap 'rm -rf -- "$context"' EXIT
cp target/bookworm/release/weight-init-bench target/bookworm/release/wgtime "$context/"
cat > "$context/Dockerfile" <<'DOCKER'
FROM weight-init-runtime:0.1.0
COPY weight-init-bench wgtime /usr/local/bin/
WORKDIR /app
DOCKER
docker build -t weight-init-bench:0.1.0 "$context"
