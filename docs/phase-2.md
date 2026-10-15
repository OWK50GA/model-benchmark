# Phase 2: model runs, benchmarks, and reports

The software supports real inference, but the implementation tests use controlled
fixtures. No weights were downloaded and no paid inference was run during this
phase. A real labeled dataset is not required to test the software; it is required
to measure model quality. See the implementation report for exact evidence.

## Gemini setup

Install `python -m pip install -e '.[gemini]'`. The adapter uses the official
`google-genai` SDK (`generate_content`), tested locally with version 1.75.0 and a
mock HTTP transport. Set `models.gemini.model_id` in your config to an image-capable
model available to your account, and export `GEMINI_API_KEY`. No model name is
assumed. Do not put keys in JSON or commit them.

The SDK request uses JSON-schema output. Responses are validated again locally.
Gemini's normalized `[ymin,xmin,ymax,xmax]` coordinates are converted into the lab's
pixel `xyxy` convention. Bad coordinates or malformed JSON become saved failures,
with the raw response retained. Unknown attributes remain null. Attribute prompts
ask only for visible evidence and do not receive ground-truth labels.

```bash
vision-lab run --model gemini --task detection --image datasets/images/outfit.jpg
vision-lab run --model gemini --task attributes --image datasets/images/outfit.jpg --region 20 30 300 500
```

`--region` is a pixel box in the EXIF-oriented image. Crops round minima down and
maxima up. It is supported for attribute and caption tasks; detection is always
whole-image inference. On multi-garment photographs, supply a region to avoid
ambiguous attribute descriptions.

## Florence setup

This adapter targets the original **Microsoft `microsoft/Florence-2-large`**
checkpoint and custom-code loader, not the separately converted community
checkpoints used in recent native Transformers examples. These loading routes
must not be silently interchanged.

1. Install the PyTorch build appropriate to CPU/CUDA on your machine.
2. Install `python -m pip install -e '.[florence]'`. The original-loader dependency
   lane is Transformers 4.49.x, PyTorch >=2.6,<3, timm 1.x and einops 0.8.x.
   This lane has not been tested with actual checkpoint weights here; revise and
   freeze exact versions after a real loading smoke test on the target machine.
3. On the Microsoft checkpoint's **Files and versions** page, choose a full
   40-character commit hash. Review the custom model/processor code at that
   revision. Set `models.florence.revision` to the hash and
   `models.florence.trust_remote_code` to `true`.
4. Set `allow_downloads=true` for the initial download if the checkpoint is not
   already cached. The default is `false`; no model weights download implicitly
   during tests or model listing. If authentication is required, set
   `credential_env` to an exported token variable name.

```bash
vision-lab run --model florence --task detection --image datasets/images/outfit.jpg
vision-lab run --model florence --task caption --image datasets/images/shirt.jpg
```

The native tasks are `<OD>` and `<MORE_DETAILED_CAPTION>`. Object detection returns
localized boxes. All native object labels are retained, including non-garments:
these count as false positives in garment detection. There is no hidden label
filter or caption-to-attribute conversion. Different vocabularies can be aligned
with an explicit evaluation alias file, described below.

Weights load lazily once per adapter/run and are released afterward. CPU uses
float32 and CUDA float16. Floating image tensors change dtype; integer token IDs
remain integers. Auto device selection falls back to CPU; an explicit unavailable
device fails with an actionable message. The first recorded inference latency
includes model loading. `timeout_seconds` currently applies to Gemini requests,
not local Florence generation; `max_output_tokens` bounds generation length.

## Benchmarks

```bash
vision-lab benchmark --model gemini --task detection \
  --dataset datasets/annotations/my-dataset.json --image-root datasets/images

vision-lab benchmark --model gemini --task attributes \
  --dataset datasets/annotations/my-dataset.json

vision-lab compare --models gemini florence --task detection \
  --dataset datasets/annotations/my-dataset.json
```

Common options: `--config PATH`, `--output-root DIR`, `--dataset PATH`,
`--image-root DIR`. If omitted, dataset/images/results paths come from the config
and resolve relative to it. Explicit command-line paths resolve from the working
directory. The `compare` command requires every model to support the selected
task. Florence cannot participate in a structured attribute comparison.

By default, synthetic or unverified annotations are excluded. Use
`--allow-unverified` only for testing the workflow; the report states this choice.
It does not bypass geometry, image checks, or other eligibility rules. No eligible
samples is an error before any model loads. An image marked `garments_complete`
with an empty garment list is treated as an annotator-declared negative image.
This completeness declaration must be reviewed too.

### Detection policy

Default `--detection-mode boxes --iou-threshold 0.5`:

- Require complete garment annotations, known categories, and all ground-truth
  boxes for an image. Exclude the whole image if any requirement is missing.
- Match predictions to annotations one-to-one by category and IoU threshold.
  Matching maximizes the number of valid matches, using deterministic IoU-first
  traversal. It does not maximize total IoU among equally sized matchings.
- Unmatched predictions are false positives; unmatched annotations are misses.
  Duplicate predictions cannot match the same annotation twice.
- Report micro precision/recall, matched-box mean IoU, counts and per-image lists
  of missed annotation IDs and incorrect prediction indices. This is not COCO AP.

`--detection-mode categories` matches category counts without boxes and reports
IoU as N/A. Category-only results must not be presented as localization accuracy.
Comparisons across modes/thresholds are rejected.

Labels are compared after case folding and whitespace normalization. Plurals,
synonyms and taxonomy differences are not guessed. Use `--category-aliases FILE`
with a JSON mapping such as `{"t-shirt":"shirt","tee":"shirt"}` when justified
by your annotation policy. Both sides use the mapping, its content is saved in
the protocol, and alias chains are rejected. Do not tune aliases against test
outcomes just to improve scores.

### Attribute policy

Each eligible garment is tested independently using its ground-truth crop. A
garment without a box is eligible only if it is the sole garment in a completely
annotated image. This isolates attribute extraction; it is not an end-to-end
detection-plus-description score.

Each field has an explicit eligible denominator. Unknown ground-truth fields are
excluded. Scalar values use normalized exact equality; list values use exact set
equality (order does not matter). No semantic judge is used. A predicted null—or
empty list against a nonempty target—is missing. Other mismatches are incorrect.
Extra list values are also counted as unsupported by the annotation. Predictions
on unknown ground-truth fields are tracked as **unscored**, not false claims.
The score assumes known list-valued annotations are exhaustive for that field.

Reports include per-field accuracy, missing/incorrect counts, unsupported values,
unscored predictions, and category accuracy. Unsupported value counts are
annotation disagreement, not proof that the model hallucinated.

### Failures and comparability

Setup, inference, malformed-response and image failures remain recorded attempts.
Their eligible targets remain in recall/attribute accuracy denominators. Failed
detection returns no matches, so its targets count as misses. Precision may remain
high despite failures; always inspect failures and recall together. An empty
denominator yields null/N/A, never fabricated perfect accuracy.

Comparison runs share the same dataset snapshot, image hashes, task and scoring
protocol. Image bytes are rechecked before inference; files changed after the
snapshot fail rather than silently producing mismatched evidence. Reports refuse
to combine runs with different comparison keys. They do not automatically declare
a best model from a small or failure-heavy dataset.

## Saved outputs

Each run creates a new directory beneath the output root:

- `manifest.json`: config/fingerprint, scoring protocol, dataset and image hashes,
  exclusions, source-code hash, runtime versions, timestamp and provenance status.
- `normalized.jsonl`: sample/image IDs, normalized prediction or error, inference
  metadata, and per-sample scoring details.
- `raw.jsonl`: raw response linked by sample ID, including prompts/schema for
  Gemini and generated text/post-processing for Florence.
- `metrics.json`: aggregate metrics, failures, latency, usage and cost coverage.
- `report.md`: human-readable summary. Comparisons add `comparison.md` to the
  first run directory.

JSONL rows are written as each sample finishes; interrupted runs can retain
partial output without final metrics. There is no resume/cache implementation yet.
Repeated runs never overwrite existing directories. Re-render completed results:

```bash
vision-lab report results/RUN_A results/RUN_B --output results/comparison-review.md
```

The output path must not exist. Reporting is offline and does not load models.
Keep raw responses private just like the source photographs. Error messages do
not copy arbitrary provider exception text, which can contain credentials.

## Usage, cost and reproducibility

Gemini records reported input, output and thinking token counts and resolved model
version when supplied by the API. Cost stays null unless all four model config
fields are supplied: `input_cost_per_million`, `output_cost_per_million`,
`cost_currency`, `cost_basis`. These are user-supplied rates, not fetched prices.
The estimate treats thinking tokens as output tokens. Configure rates accordingly;
tiered pricing, cache discounts, tool fees and other billing rules are not modeled.
Unknown costs are not presented as zero. Reported token totals include only known
usage, with coverage counts alongside them.

Latency measures adapter wall time including lazy loading and parsing, excluding
image decode. Image/pre-adapter failures have zero adapter latency. Run manifests
describe host hardware; local inference records include the chosen device and
CUDA GPU name when applicable. Hosted hardware is unknown.

Seeds, prompts, config, code and inputs are recorded, but external model revisions
and hardware can still affect reproducibility. Freeze the tested environment and
provider/checkpoint revision when collecting decision-quality results.
