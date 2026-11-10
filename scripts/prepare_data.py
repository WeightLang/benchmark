#!/usr/bin/env python3
"""Deterministic, dependency-free data preparation. Never run in the agent image."""
import csv
import gzip
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DIGITS_SHA = "09f66e6debdee2cd2b5ae59e0d6abbb73fc2b0e0185d2e1957e9ebb51e23aa22"
VOCAB = "\n .acehimnorstuw"  # 16 symbols; fixed one-hot order, no learned tokenizer.


def write(name, inputs, labels):
    (DATA / name).write_text(json.dumps({"inputs": inputs, "labels": labels}, separators=(",", ":")) + "\n")


def main():
    config = json.loads((ROOT / "config/benchmark.json").read_text())
    compressed = (DATA / "digits.csv.gz").read_bytes()
    assert hashlib.sha256(compressed).hexdigest() == DIGITS_SHA
    rows = list(csv.reader(gzip.decompress(compressed).decode().splitlines()))
    order = list(range(len(rows)))
    random.Random(config["train_seed"]).shuffle(order)
    counts = (config["train_samples"], config["test_samples"])
    assert sum(counts) <= len(rows)
    provenance = {"digits_source_sha256": DIGITS_SHA, "vocabulary": VOCAB, "context_length": 4}
    offset = 0
    for split, count in zip(("train", "test"), counts):
        indices = order[offset:offset + count]
        offset += count
        provenance[f"digits_{split}_indices"] = indices
        write(f"digits-{split}.json", [[float(v) / 16 for v in rows[i][:64]] for i in indices],
              [int(float(rows[i][64])) for i in indices])
    # Original benchmark-authored documents; split entire sentences before windows.
    documents = {
        "train": ["the cat sat on a mat.", "we can see the sun.", "a man can run.",
                  "the rain is near.", "we eat rice.", "the moon is white.", "a cat can eat."],
        "test": ["the man sat near the cat.", "we can eat rice at noon.",
                 "the sun is near the moon.", "a cat sat in the rain.", "we can run on a mat."],
    }
    assert len(VOCAB) == 16
    for split, count in zip(("train", "test"), counts):
        (DATA / f"charlm-{split}.txt").write_text("\n".join(documents[split]) + "\n")
        examples = []
        for doc in documents[split]:
            text = "\n" * 4 + doc + "\n"
            assert set(text) <= set(VOCAB)
            for i in range(4, len(text)):
                x = [0.0] * 64
                for j, ch in enumerate(text[i-4:i]):
                    x[j * 16 + VOCAB.index(ch)] = 1.0
                examples.append((x, VOCAB.index(text[i])))
        assert len(examples) >= count, "add original documents before increasing the sample budget"
        random.Random(config[f"{split}_seed"]).shuffle(examples)
        examples = examples[:count]
        write(f"charlm-{split}.json", [e[0] for e in examples], [e[1] for e in examples])
    provenance["files_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(DATA.glob("*-*.json"))}
    (DATA / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
