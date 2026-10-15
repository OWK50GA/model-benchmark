# Phase 2 implementation report

Date: 2026-10-15. Scope: garment understanding and evaluation software.

## Outcome

Implemented Gemini and Florence adapters, detection and attribute benchmarks,
single-image inference, persisted raw/normalized outputs, and comparison reports.
The software can be developed and tested without the real evaluation dataset.
Real model access/loading and quality still need live smoke tests and reviewed
data. Phase 3 embeddings and Phase 4 combinations have not been implemented.

## Delivered functionality

| Area | Implementation |
| --- | --- |
| Gemini | Lazy official SDK client; configurable model ID; detection and visible attributes; validated JSON; pixel coordinate conversion; usage/model-version recording; optional configured cost estimates |
| Florence | Original Microsoft checkpoint loader with pinned custom code; CPU/CUDA selection; native object detection with localization and detailed captions; lazy weights; no forced structured attributes |
| Detection metrics | Category-aware one-to-one matching; precision/recall; optional IoU threshold; matched IoU; missed annotations and unmatched prediction indices |
| Attribute metrics | Per-field exact accuracy; category accuracy; unknown-field masking; missing/incorrect counts; extra values; unscored claims on unknown annotations |
| Runner | Eligibility filtering; ground-truth crops; dataset/image fingerprints; changed-image detection; saved errors; failure-inclusive denominators |
| Artifacts | Unique run directories; manifest, normalized JSONL, raw JSONL, metrics JSON, Markdown report; no overwrites |
| Comparison | Same-task multi-model execution and offline reporting; rejects unsupported tasks and different dataset/image/protocol keys |
| CLI | Working `run`, `benchmark`, `compare`, `report`; retained dataset/config/model/schema tools |

## Evaluation decisions

Detection evaluates complete images with known categories. Box mode requires all
ground-truth boxes; category mode explicitly omits localization metrics. Matching
maximizes the number of valid one-to-one matches and cannot reward duplicate
predictions. Mean IoU describes matched boxes and is not COCO AP.

Attribute extraction uses annotated garment crops, separating description errors
from detection errors. Without a box, only an unambiguous single garment in a
completely annotated image is eligible. Known list fields are treated as exhaustive
sets; unknown fields do not contribute to accuracy. Extra values disagree with
the annotation but are not automatically declared hallucinations.

Failures stay in eligible recall/attribute denominators and are reported by code.
Undefined ratios are null/N/A. Reports show exclusion counts and link to exact
exclusion reasons in manifests. Synthetic/unverified data needs an explicit flag
and produces clearly marked smoke-test reports. No automatic winner is selected.

Florence's generic native detector retains all object labels. Non-garments count
as false positives; there is no hidden post-processing filter. Category aliases
are explicit, saved, and applied to both predictions and annotations.

## Reproducibility and operational behavior

Manifests record raw/normalized dataset hashes, image hashes, effective config and
its fingerprint, code hash, package versions, host details, preprocessing, and
scoring protocol. Per-sample metadata records model identity, device, wall latency
and available API usage. CUDA GPU names are recorded when the runtime initializes.
Gemini raw records include the prompt version, prompt, requested schema and response.

Seeds are used, but do not guarantee deterministic provider/GPU behavior. Wall
latency includes the first lazy model load. Image decoding is excluded; image
failures have zero adapter latency. Cost is unknown unless explicit rates and a
pricing basis are configured; usage/cost coverage is reported alongside totals.

Results stream into JSONL one sample at a time. An interruption can leave an
incomplete run without metrics; resume/caching is not implemented. Repeated runs
create separate directories. API exception messages are sanitized to avoid
persisting credentials from arbitrary exception text.

## Verification evidence

Executed on Linux / Python 3.12.3:

```text
.venv/bin/python -m pytest -q
88 passed

.venv/bin/python -m pip check
No broken requirements found.

git diff --check
Passed.
```

The tests include:

- Detection overlap geometry, duplicate predictions, wrong categories, missing
  boxes, category-only matching, and a matching case where naive greedy scoring
  would undercount valid matches.
- Attribute unknown/missing/incorrect distinctions, list order, unsupported extra
  values, and failures remaining in accuracy denominators.
- Gemini coordinate normalization, malformed response preservation, token/cost
  accounting, cropping, missing configuration/keys, and sanitized provider errors.
- **Actual Google Gen AI SDK 1.75.0 request serialization through an HTTP mock**,
  exercising image encoding and response parsing without a network/API call.
- Florence loading/generation contracts through test doubles: pinned revisions,
  cache-only loading, float32 CPU behavior, integer token preservation, generation
  settings, captions/detections and malformed post-processing.
- Dataset exclusions, changed-image detection, raw/normalized joins, report
  comparability, model cleanup, single-image failure artifacts and complete CLI
  benchmark/compare/report flows without network access.
- Existing Phase 1 validation and lazy-import regressions. Model listing still
  imports no heavy runtimes or hosted SDKs.

Installed-entry-point checks passed for `vision-lab models list` and
`vision-lab compare --help`. Git is available in this session; no commit was made.

## Live-validation limits

No user API credentials or available Gemini model ID were supplied, so there was
no paid inference. No large weights or Florence runtime were downloaded. The
Florence Transformers 4.49.x custom-code loading lane is implemented and tested
with doubles, not validated against an actual checkpoint on this machine.
It still requires a real loading smoke test, a reviewed full revision, and the
appropriate CPU/GPU environment. Windows and actual GPU execution remain untested.

These limits are not hidden by successful fixture tests. The tests establish
software behavior, not garment-recognition accuracy or universal runtime support.
No real dataset scores have been produced.

## How to use it next

Follow [Phase 2 setup and scoring](phase-2.md) to configure one adapter and run a
single image. When the labeled dataset is ready, validate it, run the desired
benchmark, then compare supported models on the same inputs. There is no need to
rewrite the runner when adding more images in the supported annotation format.

Useful sources checked during implementation:

- [Official Google Gen AI SDK](https://googleapis.github.io/python-genai/): hosted
  request API and JSON-schema output support.
- [Microsoft Florence-2-large model card](https://huggingface.co/microsoft/Florence-2-large):
  checkpoint, custom-code loading route, native tasks and license metadata.
- [Native Transformers Florence documentation](https://huggingface.co/docs/transformers/model_doc/florence2):
  newer converted-checkpoint route; deliberately not substituted for the original
  Microsoft checkpoint without testing.
