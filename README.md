# Fashionista Vision Lab

A standalone Python laboratory for evaluating garment detection, visible
attributes, and physical garment identity. **Phase 1 foundations only.** There
are no implemented model adapters, benchmark metrics, or model performance
claims yet. Phase 2 requires review of this foundation.

## Setup

Python 3.11+ is required. Python 3.12 is a practical starting point for later ML
dependencies; support for newer Python releases must be checked when adapters
are added. Phase 1 needs only Pydantic and Pillow at runtime, plus pytest for tests.

Ubuntu / WSL Ubuntu, from the repository root:

```bash
sudo apt-get install python3-venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest
```

Windows PowerShell, with Python 3.12 installed:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe run.py models list
```

Activation on Windows is optional; explicit interpreter paths avoid PowerShell
activation-policy issues. The commands below assume the virtual environment is
active. `python run.py ...`, `python -m fashionista_vision_lab ...`, and
`vision-lab ...` provide the same CLI after installation.

## Working commands

```bash
vision-lab models list
vision-lab config show config/default.json
vision-lab dataset validate datasets/annotations/example.json
vision-lab dataset inspect datasets/annotations/example.json
vision-lab schema dataset
vision-lab schema result
vision-lab schema config
```

`dataset validate` and `dataset inspect` currently return the same JSON validation
and summary payload. A successful check confirms structural validity, not the
truth of annotations. The example references nonexistent photographs on purpose.
To check your real image files as well:

```bash
vision-lab dataset validate datasets/annotations/my-dataset.json \
  --check-images --image-root datasets/images
```

Exit codes: `0` success; `1` invalid input or failed image checks; `2` CLI usage
errors or functionality deferred to later phases. Errors go to stderr, except
image-check findings, which are included in the dataset JSON payload.

`run --model gemini --image photo.jpg --task attributes` has a reserved CLI shape
and exits with an explicit Phase 2 message. `benchmark`, `compare`, `report`, and
`neighbors` are reserved commands that also exit nonzero. They do not execute
models or produce reports. Their detailed options will be designed in the phase
that implements them.

## Repository layout

```text
config/default.json                  Reproducible configuration template
datasets/annotations/example.json    Clearly synthetic format examples
datasets/images/                     Local/private image root
src/fashionista_vision_lab/
  schemas/                           Annotation and normalized output contracts
  models/base.py                     Capability-specific adapter methods
  models/registry.py                 Model descriptors and lazy factory loading
  config.py                          Configuration, paths, credential lookup, hash
  datasets.py                        JSON/JSONL I/O, image checks, inspection
  cli.py                             CLI entry point
docs/                                Dataset guide, architecture, compatibility
results/                             Reserved local result directory
tests/                               Offline foundation tests
run.py                               Repository CLI convenience entry point
pyproject.toml                       Package metadata and dependency declarations
requirements.txt                     Editable installation including tests
```

The `src/` layout keeps imports tied to the installed package and avoids generic
top-level package names such as `models`. Benchmark and evaluation modules will
be added inside the package when their implementation starts.

## Configuration and credentials

Paths in `config/default.json` resolve relative to that file, independent of the
working directory. Explicit CLI dataset/image-root arguments resolve from the
working directory. `config show` validates configuration, prints effective
defaults and resolved paths, and provides a SHA-256 fingerprint of canonical
configuration JSON. It does not validate remote model access or read credentials.

The seed is recorded for future runners; Phase 1 does not initialize random ML
runtimes. The hash is a configuration identity, not a full experiment identity:
future results also need dataset/image hashes, model revisions, preprocessing,
prompts, runtime versions, and hardware. Dependencies have compatible bounds,
not an experiment lockfile; capture exact versions when running actual benchmarks.

Gemini's model ID deliberately remains `null`: supply an API model available to
your account before its adapter is implemented. Local checkpoint IDs are
documented candidates, and revisions remain unpinned until integration testing.
The `models` mapping permits new adapter names without changing configuration
schemas; execution must resolve a registered adapter.

`.env.example` documents variable names; `.env` is **not automatically loaded**.
Export credentials from your shell or use your own secret manager:

```bash
export GEMINI_API_KEY='your-key'
export HF_TOKEN='your-token'
```

```powershell
$env:GEMINI_API_KEY = 'your-key'
$env:HF_TOKEN = 'your-token'
```

No credentials are needed for Phase 1. Keep actual values out of configuration.
Private images, custom annotation files, `.env` files, and results are ignored by
default; only the synthetic annotation example is included. Review explicitly
added files before committing.

## CPU and GPU expectations

All current functionality runs on CPU with no PyTorch, CUDA, model weights, or
API calls. `device` accepts `auto`, `cpu`, `cuda`, or `mps`, but is configuration
only until model adapters implement device selection and availability checks.
No claim of tested GPU support is made.

For later phases, install a PyTorch build appropriate to the machine and driver
using the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/).
For NVIDIA acceleration under WSL, verify the Windows driver and GPU visibility
inside WSL before installing CUDA-enabled packages. Florence-2-large CPU
generation may be slow; actual feasibility and memory usage must be measured.
Smaller DINOv3 variants are initial candidates, not a measured recommendation.
Do not download weights to run the foundation tests.

Read [dataset preparation](docs/datasets.md), [architecture](docs/architecture.md),
and [model compatibility notes](docs/model-compatibility.md) before extending the lab.
