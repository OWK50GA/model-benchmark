# Phase 3: garment identity

FashionCLIP and DINOv3 turn garment images into vectors (embeddings). Cosine
similarity ranks these vectors. Similarity is not proof of physical identity;
the benchmark checks it against the physical garment IDs in your annotations.

## Setup

Use a separate environment from Florence. The original Florence adapter requires
Transformers 4.49.x; embedding extras require >=4.56,<5. FashionCLIP and DINOv3 can
share an environment, and Gemini can also be installed there. Do not install all
extras together. Saved reports can be read from either environment.

```bash
python3 -m venv .venv-identity
source .venv-identity/bin/activate
# Install the PyTorch build for your CPU/GPU first if needed.
python -m pip install -e '.[dev,fashionclip,dinov3]'
```

On Windows, create the environment with `py -3.12 -m venv .venv-identity` and use
`.\.venv-identity\Scripts\python.exe` for pip and `run.py`. Windows and actual
GPU execution were not tested here.

In `config/default.json`, keep the appropriate Hub ID, set `revision` to a full
40-character checkpoint commit hash, and set `allow_downloads=true` only when
you intend to download weights. Downloads default to false. Moving branch names
and mutable local checkpoint directories are not supported. Both adapters disable
remote code, run float32 in inference/evaluation mode, and use CPU or CUDA (`auto`
chooses CUDA when available). MPS is selectable but untested. CPU performance has
not been measured.

FashionCLIP 2.0 uses `patrickjohncyh/fashion-clip`; its card lists MIT. DINOv3 uses
`facebook/dinov3-vits16-pretrain-lvd1689m`, has gated access and its own license.
Review the access conditions, obtain approval, and export `HF_TOKEN`. To use
already-cached authorized weights without token lookup, set `credential_env=null`.
Sources: [FashionCLIP card](https://huggingface.co/patrickjohncyh/fashion-clip),
[DINOv3 card](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m).

## Generate embeddings

```bash
vision-lab run --model fashionclip --task embedding --image datasets/images/shirt.jpg
vision-lab run --model dinov3 --task embedding --image datasets/images/outfit.jpg --region 20 30 300 500
```

Regions use EXIF-oriented pixel coordinates and outward rounding. Single-image
runs save vectors and metadata but no identity accuracy scores; they are uncached.

FashionCLIP uses `CLIPImageProcessor` and projected `CLIPModel.get_image_features`.
DINOv3 uses `AutoImageProcessor` and `AutoModel`, selecting the final CLS token
(`last_hidden_state[:,0,:]`), excluding register/patch tokens. Only ViT checkpoints
are supported; ConvNeXt is rejected. Outputs are checked for finite nonzero norm
and L2-normalized. The embedding-space ID includes checkpoint, revision, pooling,
processor settings, Transformers version, dimensions, dtype and normalization.
Retrieval rejects vectors from different spaces or dimensions.

## Prepare annotations

Each garment needs a known physical `identity_id`. Reuse it only for the same
physical item. Add a box for multi-garment images, `viewpoint` for cross-view tests,
and optionally `appearance_group` for similar-looking distinct garments. Reviewed
annotations are required unless `--allow-unverified` marks the run as a smoke test.

All photos of a physical garment should share a split. Identities spanning explicit
train/validation/test splits are excluded. The evaluated split defaults to `test`;
use `--split validation` for calibration or `--split unspecified` for unsplit data.
Unknown IDs are excluded, never treated as negative pairs. Without a box, only a
single garment in a completely annotated image is eligible. Singleton identities
can supply gallery negatives but cannot be positive retrieval queries.

## Run identity benchmarks

```bash
# 0.8 demonstrates syntax; it is NOT a recommended/calibrated threshold.
vision-lab benchmark --model fashionclip --task identity \
  --dataset datasets/annotations/my-dataset.json --match-threshold 0.8

vision-lab compare --models fashionclip dinov3 --task identity \
  --dataset datasets/annotations/my-dataset.json --match-threshold 0.8
```

Choose a threshold on separate validation data and freeze it before test runs.
There is no automatic test-set tuning or universal default. The same numerical
cutoff need not be equally calibrated across models. `compare` uses one shared
cutoff; independently calibrated thresholds require separate runs and cannot be
merged into a same-protocol report. Retrieval ranks are independent of the cutoff.

### Retrieval

Each observation queries the other eligible observations within the chosen split.
Its own image is excluded, even when multiple annotations refer to that image.
Identical decoded EXIF-oriented RGB images and identical target crops are also
excluded. Edited/near-duplicate captures are not detected; curate them manually.

A query is eligible only if a distinct positive gallery image exists before
inference. Failed queries count as misses, and failed gallery items cannot produce
hits. Queries without positives are counted separately. Recall@1/5 measures whether
a positive is among the first one/five available candidates, with stable sample-ID
tie-breaking. Small galleries can make Recall@5 trivial: inspect gallery sizes.
Cross-view scoring additionally requires known, different query/gallery viewpoints.

### Pair matching

Every unordered distinct-image/nonduplicate pair is evaluated once. Equal physical
IDs define positives; different known IDs define negatives. Cosine >= threshold
means a predicted match. Different IDs sharing `appearance_group` are hard negatives.

Reports include TP/FP/TN/FN, mean same/different cosine, failure-inclusive same-pair
recall, and false-positive rates for scored negatives and hard negatives. Failed
pairs are unscored, never true negatives. Negative rates use scored denominators
with failure coverage alongside; empty denominators are N/A. No automatic winner
or blended quality score is generated.

## Cache

Identity benchmarks cache successful embeddings in `OUTPUT_ROOT/.embedding-cache`.
Use `--cache-dir DIR` or `--no-cache`. Keys include image bytes, region, model config,
seed, runtime/host signature, preprocessing version and source-code hash. Labels
do not affect vectors; scores are always recomputed. The conservative keys may
invalidate entries after unrelated source/config changes.

Records are validated JSON with checksums, atomically published without replacing
existing entries. Failed inference is never cached. Corrupt entries fail the sample;
remove the affected entry or use `--no-cache`. No pickle is loaded. The cache
filesystem must support hard links. Keep it private like the source dataset.

A hit skips weight loading and measures lookup latency, retaining the original
inference metadata in the raw record. Auto-device hits may reuse a prior CPU/GPU
result under the same runtime signature; its recorded device remains the original
one. Use `--no-cache` for performance measurement. Files changed after the run's
input snapshot fail rather than silently reusing stale evidence.

## Inspect saved results

Identity runs add `observations.json` (IDs, pixel hashes and vectors),
`retrieval.jsonl` (query eligibility, gallery sizes/failures and top-five matches),
and `pairs.jsonl` (truth, cosine, threshold outcome and hard-negative flag) to the
existing manifest, normalized/raw JSONL, metrics and Markdown artifacts.

```bash
vision-lab neighbors results/RUN_ID --query GARMENT_ANNOTATION_ID --k 5
vision-lab neighbors results/RUN_ID --query GARMENT_ANNOTATION_ID --cross-view --show-vector
vision-lab report results/RUN_A results/RUN_B --output results/identity-comparison.md
```

Inspection is offline and does not load a model. Nearest-neighbor output includes
dimensions, embedding-space ID, norm, and optionally the vector. Similarities are
not probabilities. Existing reports require matching input/protocol fingerprints.

The current implementation holds observations/pair audits in memory and evaluates
pairs quadratically. It targets initial small collections. Large-scale indexing,
streaming sampling, threshold fitting, and combined detection/identity pipelines
are outside this phase. Phase 4 still requires real individual benchmark results.
