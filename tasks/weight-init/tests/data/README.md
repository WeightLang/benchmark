# Fixed tasks

`digits.csv.gz`: 1,797-example Optical Recognition of Handwritten Digits dataset,
8×8 pixels, distributed by scikit-learn 1.7.2. Originally E. Alpaydin and C. Kaynak,
UCI (1998). This is **not MNIST**.
Source: https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.7.2/sklearn/datasets/data/digits.csv.gz
Attribution: https://doi.org/10.24432/C50P49 (CC BY 4.0).
We select disjoint train/test rows, normalize by 16, record row IDs and hashes in
`provenance.json`. This is not the original UCI writer-disjoint protocol.

Language documents are original benchmark-authored text under the project's MIT
license. Split full sentences before making windows. Vocabulary order:
`\n .acehimnorstuw`, context length four. Predict the next character from that
context. The Transformer sees only past characters: attention cannot observe the
target. It is a tiny fixed-context encoder predictor, not a GPT-scale benchmark.
All families receive the same 64-value input.

`scripts/prepare_data.py` generates the checked-in JSON and hashes with Python's
standard library. Holdouts are excluded from the agent container. The repository
is public; these are reproducible holdouts, not permanently secret tests.
