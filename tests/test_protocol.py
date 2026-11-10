import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "target/debug/weight-init-bench"


class ProtocolTests(unittest.TestCase):
    def test_data_provenance(self):
        provenance = json.loads((ROOT / "data/provenance.json").read_text())
        self.assertFalse(set(provenance["digits_train_indices"]) & set(provenance["digits_test_indices"]))
        for name, digest in provenance["files_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / "data" / name).read_bytes()).hexdigest(), digest)

    def test_agent_does_not_receive_holdout(self):
        data = ROOT / "tasks/weight-init/environment/data"
        self.assertFalse(any("test" in p.name for p in data.iterdir()))
        self.assertFalse((data / "digits.csv.gz").exists())
        self.assertFalse((data / "provenance.json").exists())

    def test_submission_validation_happens_before_service_start(self):
        for bad in ({"version": 1, "trials": {}}, {"version": 99, "trials": {}}):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "submission.json"
                path.write_text(json.dumps(bad))
                result = subprocess.run([str(BIN), "evaluate", str(ROOT), str(path), str(Path(tmp) / "report.json")],
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("submission version or trial count mismatch", result.stderr)
                self.assertFalse((Path(tmp) / "report.json").exists())

    def test_manifest_is_repeatable(self):
        cmd = [str(BIN), "manifest", str(ROOT)]
        self.assertEqual(subprocess.check_output(cmd), subprocess.check_output(cmd))
        manifest = json.loads(subprocess.check_output(cmd))
        self.assertEqual(len(manifest["trials"]), 24)
        self.assertEqual({t["task"] for t in manifest["trials"]}, {"digits", "charlm"})


if __name__ == "__main__":
    unittest.main()
