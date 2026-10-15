# Fashionista Vision Lab

A standalone laboratory for testing garment detection, visible attributes, and
physical garment identity. **Phases 1–3 are implemented:** Gemini detection/attributes,
Florence detection/captioning, detection and attribute benchmarks, saved outputs,
and comparison reports, plus FashionCLIP/DINOv3 embeddings, caching, nearest
neighbors and physical garment identity evaluation. Phase 4 remains deferred.

This is an evaluation lab, not the Fashionista production backend. No model
performance is claimed without real labeled data.

## Setup

Python 3.11+ is required; development tests run on Python 3.12.3 / Linux.
Dataset tools, scoring, and offline tests require no model weights or API access.

Ubuntu / WSL Ubuntu, from the repository root:

```bash
sudo apt-get install python3-venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

Windows PowerShell (activation is optional):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe run.py models list
```

The installed `vision-lab` command, `python run.py`, and
`python -m fashionista_vision_lab` expose the same CLI. Commands below assume the
virtual environment is active. Windows instructions have not been executed here.

## Dataset and configuration tools

```bash
vision-lab models list
vision-lab config show config/default.json
vision-lab dataset validate datasets/annotations/example.json
vision-lab dataset inspect datasets/annotations/example.json
vision-lab schema dataset
vision-lab schema result
vision-lab schema config
```

The committed example contains invented annotations and references nonexistent
photos. It demonstrates the format; it is not benchmark ground truth. Validation
and inspection currently return the same JSON summary. Add `--check-images` to
verify actual files, EXIF-oriented dimensions, and image-root containment:

```bash
vision-lab dataset validate datasets/annotations/my-dataset.json \
  --check-images --image-root datasets/images
```

Read [dataset preparation](docs/datasets.md) for labeling guidance. Keep physical
identity separate from visual resemblance. Unknown attributes must remain null.

## Run models and benchmarks

Install only the optional runtime needed:

```bash
python -m pip install -e '.[gemini]'
# Set models.gemini.model_id in config/default.json, then export GEMINI_API_KEY.
vision-lab run --model gemini --task attributes --image datasets/images/shirt.jpg
vision-lab benchmark --model gemini --task attributes --dataset datasets/annotations/my-dataset.json
vision-lab compare --models gemini florence --task detection --dataset datasets/annotations/my-dataset.json
vision-lab report results/RUN_ID --output results/my-report.md
```

Florence requires its own optional dependencies and a pinned checkpoint revision;
see [Phase 2 setup and scoring](docs/phase-2.md) before running the comparison.
Florence supports native detection/localization and captions, not structured
attribute extraction. Comparisons reject unsupported model/task combinations.

Each run saves `manifest.json`, `normalized.jsonl`, `raw.jsonl`, `metrics.json`,
and `report.md` in a unique directory. Comparisons also save `comparison.md` in
one of those directories. Single-image runs produce predictions, not accuracy
scores. Benchmark failures stay in eligible score denominators. Reports identify
missing usage/cost coverage and excluded data.

Benchmarks require reviewed annotations by default. `--allow-unverified` is for
smoke tests only and marks the report. The example dataset cannot run inference
without corresponding image files. No combined Phase 4 pipeline is implemented.

## Garment identity

FashionCLIP and DINOv3 use a newer Transformers dependency lane than Florence.
Set them up in a separate environment using [the Phase 3 guide](docs/phase-3.md).
The core installation includes NumPy/scikit-learn for offline similarity scoring;
model runtimes remain optional and lazy.

```bash
vision-lab run --model fashionclip --task embedding --image datasets/images/shirt.jpg
# Example threshold only; choose your cutoff on separate validation data.
vision-lab compare --models fashionclip dinov3 --task identity \
  --dataset datasets/annotations/my-dataset.json --match-threshold 0.8
vision-lab neighbors results/RUN_ID --query GARMENT_ANNOTATION_ID --k 5
```

Identity benchmarks use reviewed physical IDs and garment crops, default to the
test split, exclude self/duplicate images, and report Recall@1/5, cross-view
retrieval and same/different matching. Successful embeddings are cached; use
`--no-cache` for timing comparisons. Cosine similarity does not prove identity.

Exit codes: `0` success, `1` invalid input or any recorded run/image failures,
`2` CLI usage errors or deferred functionality. Run failures still save artifacts.
An interrupted run can have partial JSONL files and no final metrics/report.

## Configuration and credentials

Configuration paths are relative to the configuration file. Explicit CLI paths
are relative to the current working directory. `config show` validates and prints
defaults, resolved paths, and a SHA-256 configuration fingerprint; it never reads
credentials or checks remote model access.

Gemini's model ID deliberately defaults to null. Set it to a model available to
your account. Florence requires a reviewed full checkpoint commit hash; see the
Phase 2 guide. Local downloads default to disabled.

`.env.example` documents variable names; `.env` is **not automatically loaded**.
Export credentials in your shell or use your own secret manager:

```bash
export GEMINI_API_KEY='your-key'
export HF_TOKEN='your-token'
```

```powershell
$env:GEMINI_API_KEY = 'your-key'
$env:HF_TOKEN = 'your-token'
```

Credential values do not belong in config JSON. Private images, custom annotation
files, `.env` files, and results are ignored by default. Review files before
explicitly adding them to Git.

The configured seed is passed to Gemini and used when initializing Florence's
PyTorch runtime. This does not guarantee deterministic API/GPU behavior. Run
manifests record input hashes, code hashes, runtime versions and preprocessing;
normalized records include device/model metadata. Dependency ranges are not a
fully locked experiment environment.

## CPU / GPU

Gemini runs on provider hardware; local device selection does not apply.
Florence `auto` selects CUDA if available, otherwise CPU. Explicit unavailable
GPU requests return a saved error. CPU uses float32 and CUDA uses float16. MPS
is selectable but untested. CPU latency/memory and actual GPU execution have not
been measured; Florence-2-large generation may be slow on CPU.

Install PyTorch for your hardware using its
[official instructions](https://pytorch.org/get-started/locally/). Under WSL,
verify NVIDIA driver/GPU visibility inside WSL before installing a CUDA build.
No weights are needed for the offline test suite.

## Layout

```text
config/default.json                  Configuration template
 datasets/annotations/example.json   Synthetic annotation example
 datasets/images/                    Local/private images
src/fashionista_vision_lab/
  schemas/                           Dataset and normalized output contracts
  models/                            Independent lazy adapters and registry
  benchmarks/                        Pure detection and attribute scoring
  config.py                          Configuration, paths, credentials, hash
  datasets.py                        JSON/JSONL I/O and image checks
  runner.py                          Eligibility, orchestration and input hashes
  artifacts.py                       Run manifests and raw/normalized files
  reporting.py                       Saved-run Markdown reports
  cli.py                             CLI
results/                             Local run outputs
 tests/                              Offline tests and SDK transport fixtures
 docs/                               Guides and implementation reports
```

The `src/` layout avoids generic import names such as `models`. Model code does
not depend on benchmark scoring. Runtime extras are imported only when used.

See [architecture](docs/architecture.md), [model compatibility](docs/model-compatibility.md),
and [Phase 2 usage](docs/phase-2.md), [Phase 3 usage](docs/phase-3.md). The [Phase 1 report](docs/phase-1-report.md)
is a historical record, not the current feature list.
