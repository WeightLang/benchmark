# Initialize the weights

Create `/logs/artifacts/submission.json` containing initial weights for every trial
in `/app/manifest.json`. Fixed tasks: 8×8 handwritten-digit recognition and
next-character prediction with a four-character context and 16-character vocabulary.
The manifest gives sampled families, seeds and exact parameter names/shapes.
WGT templates are in `/app/models`; replace the three-class template head with
10 outputs for digits or 16 for charlm. Public training data is in `/app/data`.
Language inputs concatenate four one-hot vectors; digit pixels are divided by 16.

Format: `{"version":1,"trials":{"TRIAL_ID":{"PARAMETER_NAME":[0.1,...]}}}`.
Supply all trials/parameters exactly, as flat row-major finite F32 arrays matching
the declared shapes. Maximum JSON size: 8 MiB. Submit only data.

You may implement task-aware initialization using public data. The verifier
constructs fresh instances from your weights and runs fixed Adam training.
You cannot change architecture, training schedule or evaluation data. Start with:

```sh
weight-init-bench emit /app xavier_uniform > /logs/artifacts/submission.json
```

Score: mean held-out accuracy after fixed training, equally weighting both tasks
and sampled trials. Reports also record zero-step accuracy, cross-entropy,
language perplexity and paired differences against normal, Xavier-uniform and
Kaiming-normal initialization. Preserve initializer source in
`/logs/artifacts/initializer.py` (or another source extension) for review.
Only numeric weights are consumed by the verifier; source provenance is not
automatically audited in this initial version.
