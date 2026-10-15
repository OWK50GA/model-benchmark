# Architecture and extension contract

Pydantic schemas are independent of model loading. Dataset I/O uses only Python,
Pydantic, and Pillow. No inference SDK or ML runtime is required to inspect data.
Benchmark/evaluation modules depend on these contracts; adapters must
not import benchmark code or compute benchmark scores.

## Model contracts

`ModelAdapter` defines:

```python
detect_garments(image: PIL.Image.Image) -> InferenceResult
extract_attributes(image: PIL.Image.Image, region: BoundingBox | None = None) -> InferenceResult
generate_embedding(image: PIL.Image.Image, region: BoundingBox | None = None) -> InferenceResult
caption(image: PIL.Image.Image, region: BoundingBox | None = None) -> InferenceResult
```

Images are already EXIF-oriented. Regions refer to full-image coordinates;
adapters must validate/crop them and translate any local detections back to the
full-image frame. Unsupported methods raise `UnsupportedCapabilityError`.
Advertised but unimplemented methods raise `AdapterNotImplementedError`.

A future adapter subclasses `ModelAdapter`, declares a `frozenset` of supported
capabilities, and overrides only the methods it implements. Its constructor
stores configuration; SDK/model loading should happen on first inference. Register
a `module:ClassName` factory in `models/registry.py`. `create_adapter` imports
only that factory. The registry is a small explicit mapping, not a plugin framework.

Gemini and Florence have implemented factories. FashionCLIP and DINOv3 remain
unimplemented Phase 3 descriptors. `planned_capabilities` lists each adapter's
task contract, while `implemented` distinguishes available code from plans.
Implementation does not imply live model-validation evidence. Florence has a
separate caption capability; captions are not promoted to structured attributes.

## Normalized output contracts

`InferenceResult` contains one successful output or one structured error, plus
metadata and an optional JSON-compatible `raw_response`. The capability must
match the normalized output type. Its output types are:

- `DetectionOutput`: categories, optional boxes, optional confidence in [0, 1].
- `AttributeOutput`: category, attributes, and explanations keyed by field path
  for uncertain values. Unknown attributes remain null.
- `EmbeddingOutput`: a finite nonzero vector, an embedding-space identifier, and
  a checked unit-normalization claim when `normalized=true`.
- `CaptionOutput`: native caption text, without claiming structured attributes.

Embedding-space IDs must eventually identify checkpoint/revision, preprocessing,
pooling, dimensionality, and normalization. Vectors from different spaces must
not be compared. The foundation validates individual vectors, not retrieval pairs.

Metadata supports model name/ID/revision, device, latency, optional token counts,
and estimated cost with currency and pricing basis. It does not invent missing
usage values. A null revision or cost means unavailable/unrecorded, not zero.
The run artifacts also record host hardware, prompts, dataset hashes, timestamps,
environment versions, and per-image associations. GPU name is recorded when local
CUDA inference initializes; provider hardware is unknown.

Serialization is available through Pydantic's `model_dump_json()` and
`model_validate_json()`. Phase 2 persists raw and normalized inference in separate
JSONL files linked by sample ID. Raw responses exclude authorization headers.

## Reproducibility and scope

Configuration is versioned JSON. Relative paths use the config file location;
the stable config fingerprint includes default values and excludes resolved
environment credentials. Paths remain part of that fingerprint. The recorded
seed does not by itself guarantee deterministic inference.

Schema version changes are explicit; migrations are not implemented yet. All
reads validate the contract. Detection and attribute benchmarks apply explicit
eligibility rules for optional boxes, unknown attributes, partial annotations and
unverified data. Metric denominators and excluded samples are reported. Identity
eligibility and scoring remain Phase 3.

Phase 2 provides Gemini/Florence integration, detection and attribute metrics,
run artifacts and comparative reporting; see [its guide](phase-2.md). Phase 3 will add embedding adapters,
caching, retrieval and identity evaluation. Phase 4 requires results from those
individual evaluations. No Phase 4 pipeline is implemented here.
