#!/usr/bin/env python3
"""Exercise the pinned SDK and a supplied native daemon with six one-step runs."""

import argparse
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/debug/weight-init-bench")
    parser.add_argument("--daemon", type=Path, default=ROOT / "target/debug/wgtime")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    daemon = args.daemon.resolve(strict=True)
    report = args.report.resolve()
    if report.exists():
        parser.error("report exists; choose a new output path")
    if not report.parent.is_dir():
        parser.error("report parent directory must already exist")

    revision = (ROOT / "WEIGHTLANG_REVISION").read_text().strip()
    manifest = json.loads(subprocess.check_output([str(binary), "manifest", str(ROOT)]))
    if manifest["weightlang_revision"] != revision:
        parser.error("benchmark executable was built from a different SDK pin")
    if (ROOT / ".vendor/weightlang/.benchmark-revision").read_text().strip() != revision:
        parser.error("vendored source does not match the SDK pin")

    config = json.loads((ROOT / "config/benchmark.json").read_text())
    config.update(draws=1, initialization_seeds=[11], steps=1, batch_size=1)
    with tempfile.TemporaryDirectory(prefix="weight-init-sdk-smoke-") as directory:
        scratch = Path(directory)
        (scratch / "models").symlink_to(ROOT / "models", target_is_directory=True)
        (scratch / "data").symlink_to(ROOT / "data", target_is_directory=True)
        (scratch / "config").mkdir()
        (scratch / "config/benchmark.json").write_text(json.dumps(config))
        env = os.environ.copy()
        env["WGTIME_DAEMON"] = str(daemon)
        subprocess.run(
            [str(binary), "baselines", str(scratch), str(report)],
            env=env,
            check=True,
        )

    result = json.loads(report.read_text())
    if result["backend"] != "native-cpu":
        raise RuntimeError("smoke did not use the native CPU backend")
    if result["protocol"]["weightlang_revision"] != revision:
        raise RuntimeError("smoke report does not match the SDK pin")
    rows = result["trials"]
    if len(rows) != 6 or {row["task"] for row in rows} != {"digits", "charlm"}:
        raise RuntimeError("smoke did not complete both tasks and all initializers")
    methods = {"normal", "xavier_uniform", "kaiming_normal"}
    for task in ("digits", "charlm"):
        if {row["method"] for row in rows if row["task"] == task} != methods:
            raise RuntimeError("smoke did not complete every task/initializer pair")
    for row in rows:
        metrics = row["metrics"]
        if len(metrics["training_loss"]) != 1:
            raise RuntimeError("smoke did not complete exactly one training step")
        values = metrics["training_loss"] + [
            metrics[stage][metric]
            for stage in ("initial", "final")
            for metric in ("accuracy", "cross_entropy")
        ]
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("smoke returned nonfinite loss or evaluation metrics")
    print(json.dumps({
        "weightlang_revision": revision,
        "backend": result["backend"],
        "executions": len(rows),
        "families": sorted({row["family"] for row in rows}),
        "tasks": sorted({row["task"] for row in rows}),
        "steps_per_execution": 1,
        "batch_size": 1,
        "report": str(report),
    }, indent=2))


if __name__ == "__main__":
    main()
