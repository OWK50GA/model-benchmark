# Phase 3 implementation report

Date: 2026-10-15. Scope: garment embeddings, retrieval and physical identity
evaluation. Phase 4 has not been started.

## Delivered

| Requirement | Implementation |
| --- | --- |
| FashionCLIP 2.0 | Lazy native CLIP image processor/model, projected image features, normalized output |
| DINOv3 | Lazy native ViT processor/backbone, final CLS features, explicit access/configuration errors |
| Embeddings | Single-image/region generation, finite nonzero validation, L2 normalization, embedding-space fingerprints |
| Caching | Content-addressed JSON, checksum/schema validation, atomic publication, no overwrite, success-only storage |
| Nearest neighbors | Offline saved-run inspection, cosine ranking, deterministic ties, self/duplicate exclusion, optional cross-view filter |
| Identity benchmarks | Recall@1/5, cross-view retrieval, pair confusion counts, same-pair recall, negative/hard-negative false-positive rates |
| CLI and reports | Identity task on benchmark/compare; working neighbors command; additional audit artifacts and Markdown sections |

The adapters remain independent of evaluation logic. Model runtimes are imported
only when inference needs them. NumPy and scikit-learn are installed for scoring;
PyTorch/Transformers and model weights were not installed/downloaded in this phase.

## Important implementation decisions

The embedding runtime uses Transformers >=4.56,<5, which conflicts with Florence's
existing 4.49.x custom-code lane. The documented solution is separate virtual
environments, retaining the original Florence behavior. Both embedding models can
share one environment. Saved result reporting does not require model runtimes.

Checkpoint IDs must be pinned to full commit hashes. Mutable local checkpoint
directories are rejected. Both encoders use float32, native processors, evaluation
mode and inference mode; no remote code. DINOv3 is deliberately limited to ViT
checkpoints with the documented CLS extraction policy. CPU/CUDA selection reuses
the existing device checks; actual hardware execution remains untested.

Embedding spaces record the checkpoint, revision, pooling, processor settings,
Transformers version, dimensionality, dtype and normalization. Retrieval rejects
mixed spaces/dimensions. Cache keys additionally cover image bytes, region, seed,
runtime/host signature and implementation hash. Cache hits preserve the origin
metadata and report lookup latency rather than reusing the original inference
duration as a new timing measurement.

Identity eligibility uses physical IDs, review status, explicit splits and garment
crops. Cross-split identities and ambiguous unboxed targets are excluded. Unknown
IDs never become negatives. Self-images and exact decoded image/crop duplicates
cannot supply retrieval positives or pair scores. Near-duplicates still require
manual curation.

Retrieval denominators are determined before inference. Failed eligible queries
are misses, and failed gallery embeddings cannot create hits. Pair false-positive
rates use only scored negative pairs, with failed pairs reported separately;
failures are never counted as true negatives. The cosine threshold is explicitly
supplied, not chosen against the test set. A common threshold across models is not
automatically a fair calibration, and the documentation explains this limitation.

## Artifacts

Existing manifest, normalized/raw JSONL, metrics and report files are retained.
Identity runs add:

- `observations.json`: annotation/image/identity IDs, viewpoints, pixel hashes,
  embedding-space identifiers and vectors or missing-embedding status.
- `retrieval.jsonl`: query eligibility, gallery sizes, failures and top-five matches.
- `pairs.jsonl`: truth labels, hard-negative flags, cosine values and threshold outcomes.

`neighbors` reads these saved observations, displays vector dimensions/norm/space,
and optionally the vector. It does not contact a model. Reports reject different
dataset/image/protocol comparison keys.

## Verification

Executed on Linux with Python 3.12.3:

```text
.venv/bin/python -m pytest -q
115 passed in 3.99s

.venv/bin/python -m pip check
No broken requirements found.

git diff --check
Passed.
```

The installed `vision-lab neighbors --help` entry point also passed. Core numeric
versions used were NumPy 2.5.4 and scikit-learn 1.9.1.

Tests cover projected CLIP versus CLS extraction, lazy loading, pinned loader
options, crops, normalization, zero-vector rejection, revision-sensitive spaces,
gated-token errors, unsupported architectures, cache reuse/invalidation/corruption,
retained origin timing, retrieval ties, Recall@5 cutoffs, large-vector numerical
stability, duplicate exclusion, failed-query denominators, pair coverage, hard
negatives, split leakage, mixed spaces, offline neighbors, and full CLI identity
comparison with saved setup failures. Existing Phase 1/2 tests also pass.

All new model execution tests use test doubles. Numeric scoring uses real
NumPy/scikit-learn operations. No fabricated fixture score is presented as model
performance, and there are no real garment accuracy results yet.

## Remaining live checks and practical limits

- Test pinned FashionCLIP/DINOv3 checkpoint loading and processor behavior with
  actual weights. DINOv3 needs account approval/license acceptance where required.
- Measure CPU/GPU memory and latency on the target machine; Windows/MPS are untested.
- Collect reviewed physical identities with distinct views and difficult negatives;
  choose matching thresholds on a separate validation split.
- Use `--no-cache` for speed comparisons; cached hits may reuse a prior auto-selected
  device's vector while retaining its original device metadata.
- Current evaluation retains observations and pair audits in memory and performs
  quadratic pair comparisons. Large-scale streaming/indexing is not implemented.

Read [Phase 3 setup and evaluation policy](phase-3.md) for commands and configuration.
Primary model/API sources are linked there and in [compatibility notes](model-compatibility.md).
No Git commit was created.
