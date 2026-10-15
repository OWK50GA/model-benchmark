# Phase 1 implementation report

Date: 2026-10-14. Scope: foundation only. Phase 2 has not started.

## Delivered against the six Phase 1 tasks

| Task | Delivered |
| --- | --- |
| Initialize Python project | Python 3.11+ package, `src/` layout, build metadata, editable install, requirements, ignore rules, environment-variable example |
| Dataset and annotation schemas | Versioned Pydantic schemas, JSON/JSONL I/O, provenance, garment identity, attributes, boxes, viewpoint/context, optional local image validation |
| Normalized model interfaces | Three capability-specific methods, explicit unsupported/unimplemented errors, lazy adapter factory, normalized prediction and failure contracts |
| Configuration and CLI | Validated JSON config, config-relative paths, stable SHA-256 fingerprint, credential-variable references, dataset/config/model/schema commands |
| Example annotations and tests | Three fictional images, four garment observations, repeated identity and hard negative; 58 passing offline tests |
| Dataset preparation documentation | Annotation workflow, coordinate conventions, unknown-value semantics, review provenance, JSONL format, evaluation leakage considerations |

## Design decisions

The installable `src/fashionista_vision_lab` package keeps imports explicit and
separates model contracts from dataset utilities. Runtime dependencies are limited
to Pydantic and Pillow. PyTorch, Transformers, hosted SDKs, NumPy, and scikit-learn
will be added with the adapters or metrics that actually require them.

Dataset IDs and annotation IDs are validated for uniqueness at the appropriate
level. Physical garment identity and visual appearance groups are distinct fields.
Unknown category/identity/attribute values remain null. Empty attribute lists
represent reviewed absence, and `garments_complete` distinguishes partial labels
from exhaustive image annotations. Boxes use EXIF-oriented pixel coordinates and
must have positive area within image dimensions.

Synthetic annotations are explicitly labeled and cannot claim verified status.
Real verified garment records require a reviewer identifier. These safeguards
validate provenance declarations; they cannot prove annotation correctness.
Dataset inspection warns about unverified examples, incomplete garment coverage,
and physical identities appearing in multiple explicit dataset splits.

Model descriptors distinguish planned capabilities from implementations. All four
descriptors remain unimplemented. The registry can load a future factory lazily,
and unsupported tasks fail explicitly. Normalized outputs retain an optional raw
JSON response alongside typed predictions or structured errors. Metadata supports
model identity/revision, latency, device, token usage, and explained cost estimates.

Configuration hashing is deterministic across mapping order and sensitive to
configuration changes. Credential values are read only through an explicit helper
and are excluded from serialized configuration. The CLI avoids printing rejected
configuration values in validation errors. Config fingerprints are not full
experiment fingerprints: model/data hashes and environment manifests remain future
runner responsibilities.

## Working CLI surface

- `dataset validate PATH`: validate JSON/JSONL and summarize annotations.
- `dataset inspect PATH`: currently the same validation and summary payload.
- `--check-images --image-root ROOT`: optionally verify actual image files,
  orientation-aware dimensions, and containment within the image root.
- `config show PATH`: print validated defaults, resolved paths, and config hash.
- `models list`: report the four planned adapters and their capability boundaries.
- `schema dataset|result|config`: emit JSON Schema for tooling and annotation work.

The installed `vision-lab` entry point, `python run.py`, and
`python -m fashionista_vision_lab` are available. `run` has an initial argument
shape but returns a later-phase error. `benchmark`, `compare`, `report`, and
`neighbors` reserve command names and return explicit nonzero errors. No fake
inference output or benchmark report is emitted by these commands.

## Verification evidence

Executed in the project virtual environment on Linux using Python 3.12:

```text
.venv/bin/python -m pytest -q
58 passed in 0.66s

.venv/bin/python -m pip check
No broken requirements found.
```

The tests cover duplicate IDs, unsafe paths, invalid/out-of-bounds boxes,
synthetic/verified provenance, unknown versus absent attributes, JSON and JSONL
round trips, overwrite protection, malformed JSONL, missing/corrupt images,
dimension mismatches, EXIF orientation, symlink containment, split leakage,
output/capability agreement, embedding validation, cost metadata, lazy factories,
configuration fingerprints, missing credentials, secret-value omission, schema
export, and deferred-command failures.

A subprocess test checks that model listing imports none of `torch`,
`transformers`, `google.genai`, `numpy`, or `sklearn`. The test-only adapter and
prediction fixtures exercise contracts; they are not benchmark evidence.

Manual installed-CLI smoke checks also passed:

```text
vision-lab config show config/default.json
python run.py dataset validate datasets/annotations/example.json
python -m fashionista_vision_lab --version
```

The example validation reports 3 images, 4 garment annotations, 3 unique physical
identities, 3 boxes, and 1 identity present across multiple images. Every annotation
is marked example. These are dataset-structure counts, not model results.

Dependency versions in the tested environment: Pydantic 2.14.1, Pillow 12.3.0,
pytest 9.1.1. The project uses bounded dependency ranges; it does not claim a fully
locked experiment environment. Windows and other Python versions were documented
but not executed in this environment.

## Compatibility findings and outstanding uncertainties

Primary-source research, identifiers, links, and access/loading requirements are
recorded in [model-compatibility.md](model-compatibility.md). In particular,
Gemini needs a user-selected available model ID, Florence captioning must not be
silently treated as structured attributes, FashionCLIP similarity is not physical
identity, and DINOv3 needs gated access/license review and a defined pooling policy.

No model weights were downloaded, no hosted inference was requested, and no
model scores were invented. Actual SDK/version compatibility, CPU feasibility,
GPU selection/fallback, model access failures, and inference normalization still
need integration tests when their adapters are implemented. Phase 1 device values
are configuration declarations only.

Real datasets must still be collected and manually reviewed. Benchmark eligibility,
metric denominators, query/gallery construction, threshold tuning, embedding
caching, raw-output persistence, and comparative report generation are deferred
to their specified phases. Phase 4 requires measured individual benchmark results.

The workspace exposed an unusable `.git` directory (`git status` reported that
this was not a Git repository), so no Git diff/status or commit was available.
The implementation and tests were verified directly from the files on disk.

Phase 1 is ready for review. Work stops here before model adapter implementation.
