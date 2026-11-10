#!/usr/bin/env python3
"""Validate continuous-score Oracle parity; perfect accuracy is not required."""
import json
import sys

report = json.load(open(sys.argv[1]))
assert report["backend"] == "native-cpu"
assert 0 <= report["reward"] <= 1
assert report["paired_accuracy_delta"]["xavier_uniform"] == 0
pairs = {}
for row in report["trials"]:
    if row["method"] in ("candidate", "xavier_uniform"):
        pairs.setdefault(row["trial"], {})[row["method"]] = row["metrics"]
assert len(pairs) == len(report["protocol"]["trials"])
assert all(p["candidate"] == p["xavier_uniform"] for p in pairs.values())
assert {r["task"] for r in report["trials"]} == {"digits", "charlm"}
print(f"{len(pairs)} trials: Oracle matches Xavier exactly; reward={report['reward']:.8f}")
