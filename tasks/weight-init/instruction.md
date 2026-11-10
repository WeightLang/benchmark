# Design an architecture-aware weight initialization algorithm

You are the AI agent being evaluated. Your task is to write a reproducible weight
initialization algorithm for the sampled neural-network architectures. The goal
is to test whether your understanding of parameter roles and computation across
general-purpose architectures leads to better initial weights than standard
random initialization. The downstream digit classifier and character language
model are evaluation instruments, not the AI agent being scored.

Read the WGT architecture definitions. Account for tensor layout, fan-in/fan-out,
activations, residual paths, attention projections and normalization where relevant.
Implement rules that operate on the architecture and parameter schema, rather
than hardcoding weight arrays or looking up answers by trial ID. Architecture-
specific rules are allowed; explain their purpose and use them consistently across
seeds. Randomness is allowed when controlled by the supplied initialization seed.

Create `/logs/artifacts/initializer.py` (or another source extension) with your
algorithm and `/logs/artifacts/initialization.md` describing its rationale and
the exact command used to reproduce the weights. Run the algorithm to create
`/logs/artifacts/submission.json` containing initial weights for every trial
in `/app/manifest.json`. Fixed tasks: 8×8 handwritten-digit recognition and
next-character prediction with a four-character context and 16-character vocabulary.
The manifest gives sampled families, seeds and exact parameter names/shapes.
WGT templates are in `/app/models`; replace the three-class template head with
10 outputs for digits or 16 for charlm. Public training data is in `/app/data`.
Language inputs concatenate four one-hot vectors; digit pixels are divided by 16.

Format: `{"version":1,"trials":{"TRIAL_ID":{"PARAMETER_NAME":[0.1,...]}}}`.
Supply all trials/parameters exactly, as flat row-major finite F32 arrays matching
the declared shapes. Maximum JSON size: 8 MiB. The JSON must contain only data.

You may use public training-input statistics to design task-aware initialization,
but do not fit the task labels, pretrain the downstream model, or import pretrained
weights. All downstream optimization belongs to the fixed verifier phase. The verifier
constructs fresh instances from your weights and runs fixed Adam training.
You cannot change architecture, training schedule or evaluation data. Start with:

```sh
weight-init-bench emit /app xavier_uniform > /logs/artifacts/submission.json
```

Score: mean held-out accuracy after fixed training, equally weighting both tasks
and sampled trials. Compare architecture-by-architecture improvements against the
random controls to assess whether your algorithm generalizes across the sampled
families; an aggregate score alone is not proof of architectural understanding.
Reports also record zero-step accuracy, cross-entropy,
language perplexity and paired differences against normal, Xavier-uniform and
Kaiming-normal initialization.

Current validation boundary: only numeric weights are consumed by the automated
verifier. Algorithm source and rationale are submitted for review; this version
does not automatically audit their provenance or replay the algorithm on unseen
architectures. These limitations do not waive the task's algorithm requirements.
