"""Ground-truth garment crops -> cached embeddings -> auditable identity metrics."""

import json
from pathlib import Path

from PIL import Image

from .artifacts import RunWriter, digest, environment, json_digest, write_json
from .benchmarks.identity import IdentityProtocol, Observation, score_identity
from .embedding_cache import EmbeddingCache
from .models.registry import create_adapter, get_descriptor
from .models.runtime import crop_image
from .runner import execution_summary, failed_result, image_bytes, load_image
from .schemas.outputs import EmbeddingOutput, InferenceResult


def select_samples(dataset, protocol):
    samples, excluded = [], []
    # Prevent a physical identity leaking into the selected evaluation split.
    splits = {}
    for image in dataset.images:
        for garment in image.garments:
            if garment.identity_id and image.split != "unspecified":
                splits.setdefault(garment.identity_id, set()).add(image.split)
    for image in dataset.images:
        for garment in image.garments:
            reason = None
            if image.split != protocol.split:
                reason = "different_split"
            elif not protocol.allow_unverified and (dataset.synthetic or garment.verification != "verified"):
                reason = "unverified_or_synthetic"
            elif not garment.identity_id:
                reason = "unknown_physical_identity"
            elif len(splits.get(garment.identity_id, set())) > 1:
                reason = "identity_spans_splits"
            elif garment.bbox is None and (len(image.garments) != 1 or not image.garments_complete):
                reason = "ambiguous_target_without_box"
            if reason:
                excluded.append({"sample_id": garment.annotation_id, "reason": reason})
            else:
                samples.append((image, garment))
    return samples, excluded


def pixel_digest(image):
    return digest(str(image.size).encode() + b"\x00RGB\x00" + image.tobytes())


def read_observations(run: Path):
    values = json.loads((run / "observations.json").read_text(encoding="utf-8"))
    return [Observation.model_validate(value) for value in values]


def identity_benchmark(dataset, dataset_bytes, image_root, model_names, config, output_root,
                       protocol: IdentityProtocol, cache_root=None, adapter_factory=create_adapter):
    if not model_names or len(set(model_names)) != len(model_names):
        raise ValueError("provide distinct model names")
    for name in model_names:
        if "embedding" not in get_descriptor(name).planned_capabilities:
            raise ValueError(f"{name} does not support identity embeddings")
        if name not in config.models:
            raise ValueError(f"missing model configuration: {name}")
    samples, excluded = select_samples(dataset, protocol)
    if not samples:
        raise ValueError("No eligible identity samples: check split, physical IDs, review status and garment boxes.")
    # Snapshot full-image and target-pixel fingerprints before any model starts.
    snapshots, templates = {}, []
    for record, garment in samples:
        pixels = crop_pixels = None
        try:
            data = image_bytes(image_root, record)
            file_hash = digest(data)
            with load_image(data) as image:
                if image.size != (record.width, record.height):
                    raise ValueError("image dimensions differ from annotations")
                pixels = pixel_digest(image)
                with crop_image(image, garment.bbox) as crop:
                    crop_pixels = pixel_digest(crop)
        except (OSError, ValueError, RuntimeError, Image.DecompressionBombError):
            file_hash = None
        snapshots[record.image_id] = file_hash
        templates.append(Observation(sample_id=garment.annotation_id, image_id=record.image_id,
            identity_id=garment.identity_id, appearance_group=garment.appearance_group,
            viewpoint=garment.viewpoint, file_path=record.file_path, image_digest=pixels, crop_digest=crop_pixels))
    comparison = {"task": "identity", "dataset_sha256": digest(dataset_bytes),
                  "dataset_normalized_sha256": json_digest(dataset.model_dump(mode="json")),
                  "image_sha256": snapshots, "protocol": protocol.model_dump(mode="json")}
    status = "SYNTHETIC / SMOKE TEST ONLY" if dataset.synthetic else (
        "UNVERIFIED DATA ALLOWED" if protocol.allow_unverified else "REVIEWED ANNOTATIONS (as declared)")
    paths = []
    runtime = environment()
    for name in model_names:
        adapter = adapter_factory(name, config.models[name])
        adapter.seed = config.seed
        signature = {"model": name, "config": config.models[name].model_dump(mode="json"),
                     "seed": config.seed, "environment": runtime,
                     "preprocessing": "exif-rgb-crop-outward-v1"}
        cache = EmbeddingCache(cache_root, signature) if cache_root is not None else None
        writer = RunWriter(output_root, {**comparison, "model": name, "comparison_key": json_digest(comparison),
            "dataset_id": dataset.dataset_id, "data_status": status, "excluded": excluded,
            "config": config.model_dump(mode="json"), "config_fingerprint": config.fingerprint(),
            "cache_signature": signature, "cache_enabled": cache is not None,
            "preprocessing": "ground-truth crop; EXIF RGB; floor minima/ceil maxima; pinned model processor",
            "gallery_policy": "leave-one-image-out within split; exclude identical decoded images or crops",
            "threshold_policy": "user supplied; not fitted by this benchmark"})
        observations = [o.model_copy(deep=True) for o in templates]
        results = []
        try:
            for (record, garment), observation in zip(samples, observations):
                cache_key = None
                try:
                    data = image_bytes(image_root, record)
                    if snapshots[record.image_id] is None or digest(data) != snapshots[record.image_id]:
                        raise ValueError("image missing or changed after snapshot")
                    with load_image(data) as image:
                        cache_key = cache.key(data, garment.bbox) if cache else None
                        result = cache.get(cache_key) if cache else None
                        if result is None:
                            result = adapter.generate_embedding(image, garment.bbox)
                            result = InferenceResult.model_validate_json(result.model_dump_json())
                            if result.capability != "embedding" or (result.output is not None and not isinstance(result.output, EmbeddingOutput)):
                                raise ValueError("adapter returned non-embedding output")
                            if cache:
                                cache.put(cache_key, result)
                    observation.embedding = result.output
                except Exception as exc:
                    result = failed_result(name, config.models[name], "embedding", "embedding_pipeline_error",
                        f"Embedding/image/cache processing failed ({type(exc).__name__}); check files and cache or use --no-cache.")
                writer.append(garment.annotation_id, record.image_id, result, {"cache_key": cache_key})
                results.append(result)
        finally:
            adapter.close()
        write_json(writer.path / "observations.json", [o.model_dump(mode="json") for o in observations])
        metrics, queries, pairs = score_identity(observations, protocol.threshold)
        for filename, rows in (("retrieval.jsonl", queries), ("pairs.jsonl", pairs)):
            with (writer.path / filename).open("x", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
        writer.finish({**metrics, **execution_summary(results),
                       "cache_hits": sum(r.metadata.cache_hit for r in results),
                       "successful_embeddings": sum(o.embedding is not None for o in observations)})
        paths.append(writer.path)
    return paths
