# Phase 1 architecture and extension contract

Pydantic schemas are independent of model loading. Dataset I/O uses only Python,
Pydantic, and Pillow. No inference SDK or ML runtime is required to inspect data.
Future benchmark/evaluation modules will depend on these contracts; adapters must
not import benchmark code or compute benchmark scores.

## Model contracts

`ModelAdapter` defines:

```python
detect_garments(image: PIL.Image.Image) -> InferenceResult
extract_attributes(image: PIL.Image.Image, region: BoundingBox | None = None) -> InferenceResult
generate_embedding(image: PIL.Image.Image, region: BoundingBox | None = None) -> InferenceResult
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

All four current descriptors have `factory_path=null` and `implemented=false`.
Their `planned_capabilities` are planning information, not live support claims.
No placeholder adapter pretends to return meaningful predictions. Florence
captioning may later have a separate output contract; it is not silently promoted
to structured garment attributes.

## Normalized output contracts

`InferenceResult` contains one successful output or one structured error, plus
metadata and an optional JSON-compatible `raw_response`. The capability must
match the normalized output type. Its three output types are:

- `DetectionOutput`: categories, optional boxes, optional confidence in [0, 1].
- `AttributeOutput`: category, attributes, and explanations keyed by field path
  for uncertain values. Unknown attributes remain null.
- `EmbeddingOutput`: a finite nonzero vector, an embedding-space identifier, and
  a checked unit-normalization claim when `normalized=true`.

Embedding-space IDs must eventually identify checkpoint/revision, preprocessing,
pooling, dimensionality, and normalization. Vectors from different spaces must
not be compared. The foundation validates individual vectors, not retrieval pairs.

Metadata supports model name/ID/revision, device, latency, optional token counts,
and estimated cost with currency and pricing basis. It does not invent missing
usage values. A null revision or cost means unavailable/unrecorded, not zero.
Hardware inventory, prompts, dataset hashes, timestamps, environment versions,
and per-image run associations belong in the future experiment/run manifest.

Serialization is available through Pydantic's `model_dump_json()` and
`model_validate_json()`. Actual persistence of raw and normalized inference runs
is Phase 2. Raw responses must exclude credentials/request authorization headers.

## Reproducibility and scope

Configuration is versioned JSON. Relative paths use the config file location;
the stable config fingerprint includes default values and excludes resolved
environment credentials. Paths remain part of that fingerprint. The recorded
seed does not by itself guarantee deterministic inference.

Schema version changes are explicit; migrations are not implemented yet. All
reads validate the contract. Benchmark-specific eligibility rules must be added
with their benchmark: optional boxes, unknown attributes, partial annotations,
unverified data, and uncertain identities cannot be silently scored as failures
or successes. Metric denominators and excluded examples must be reported.

Phase 2 will add Gemini/Florence integration, detection and attribute metrics,
run artifacts and comparative reporting. Phase 3 will add embedding adapters,
caching, retrieval and identity evaluation. Phase 4 requires results from those
individual evaluations. No Phase 4 pipeline is implemented here.
