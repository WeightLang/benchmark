#!/usr/bin/env bash
set -euo pipefail
mkdir -p /logs/verifier
printf '0\n' > /logs/verifier/reward.txt
weight-init-bench evaluate /tests /logs/artifacts/submission.json /logs/verifier/report.json
python3 - <<'PYTHON'
import json
from pathlib import Path
report = json.loads(Path('/logs/verifier/report.json').read_text())
reward = report['reward']
assert isinstance(reward, (int, float)) and 0 <= reward <= 1
Path('/logs/verifier/reward.txt').write_text(str(reward) + '\n')
PYTHON
