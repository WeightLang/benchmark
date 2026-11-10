#!/usr/bin/env python3
"""Copy public and trusted inputs into separate Harbor build contexts."""
import shutil
from pathlib import Path
root = Path(__file__).resolve().parents[1]
task = root / "tasks/weight-init"
for context in (task / "environment", task / "tests"):
    for name in ("models", "config"):
        shutil.copytree(root / name, context / name, dirs_exist_ok=True)
    (context / "data").mkdir(exist_ok=True)
    for path in (root / "data").iterdir():
        if context.name == "environment" and ("test" in path.name or path.name in ("digits.csv.gz", "provenance.json", "README.md")):
            continue
        shutil.copy2(path, context / "data" / path.name)
