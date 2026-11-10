#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
revision=$(cat WEIGHTLANG_REVISION)
if [[ ! -f .vendor/weightlang/.benchmark-revision ]]; then
    if [[ -d .vendor/weightlang ]]; then
        echo 'Existing source directory has no revision marker; verify it before reuse.' >&2
        exit 1
    fi
    mkdir -p .vendor/weightlang
    curl --fail --location "https://codeload.github.com/WeightLang/WeightLang/tar.gz/$revision" |
        tar -xz --strip-components=1 -C .vendor/weightlang
    printf '%s\n' "$revision" > .vendor/weightlang/.benchmark-revision
fi
test "$(cat .vendor/weightlang/.benchmark-revision)" = "$revision"
cargo build --locked
cargo build --locked --manifest-path .vendor/weightlang/Cargo.toml -p wgtime-service --target-dir target
target/debug/weight-init-bench manifest . > tasks/weight-init/environment/manifest.json
python3 scripts/sync_task.py
