#!/usr/bin/env bash
set -euo pipefail
mkdir -p /logs/artifacts
weight-init-bench emit /app xavier_uniform > /logs/artifacts/submission.json
