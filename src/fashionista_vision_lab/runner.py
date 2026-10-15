"""Experiment orchestration: eligibility, immutable input fingerprints, and artifacts."""

from collections import Counter
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps

from .artifacts import RunWriter, digest, json_digest
from .benchmarks.attributes import aggregate_attributes, score_attributes
from .benchmarks.detection import aggregate_detection, label, score_detection
from .models.registry import create_adapter, get_descriptor
from .schemas.common import Schema, Identifier
from .schemas.outputs import InferenceError, InferenceMetadata, InferenceResult
from .schemas.dataset import Dataset
from pydantic import Field, field_validator
from typing import Literal


class BenchmarkProtocol(Schema):
    version: Literal["phase2-v1"] = "phase2-v1"
    detection_mode: Literal["boxes", "categories"] = "boxes"
    iou_threshold: float = Field(default=0.5, gt=0, le=1)
    allow_unverified: bool = False
    category_aliases: dict[Identifier, Identifier] = Field(default_factory=dict)

    @field_validator("category_aliases")
    @classmethod
    def normalized_aliases(cls, value):
        normalized = {label(k): label(v) for k, v in value.items()}
        if len(normalized) != len(value):
            raise ValueError("duplicate category aliases after normalization")
        if any(v in normalized and normalized[v] != v for v in normalized.values()):
            raise ValueError("alias chains are not supported; map directly to canonical labels")
        return normalized


def eligible_samples(dataset: Dataset, task: str, protocol: BenchmarkProtocol):
    samples, excluded = [], []
    for image in dataset.images:
        if task == "detection":
            reason = None
            if not image.garments_complete:
                reason = "incomplete_garment_labels"
            elif not protocol.allow_unverified and (dataset.synthetic or any(g.verification != "verified" for g in image.garments)):
                reason = "unverified_or_synthetic"
            elif any(g.category is None for g in image.garments):
                reason = "unknown_category"
            elif protocol.detection_mode == "boxes" and any(g.bbox is None for g in image.garments):
                reason = "missing_ground_truth_boxes"
            if reason:
                excluded.append({"sample_id": image.image_id, "reason": reason})
            else:
                samples.append((image, None))
        elif task == "attributes":
            for garment in image.garments:
                reason = None
                if not protocol.allow_unverified and (dataset.synthetic or garment.verification != "verified"):
                    reason = "unverified_or_synthetic"
                elif garment.bbox is None and (len(image.garments) != 1 or not image.garments_complete):
                    reason = "ambiguous_target_without_box"
                elif garment.category is None and all(v is None for v in garment.attributes.model_dump().values()):
                    reason = "no_known_attributes"
                if reason:
                    excluded.append({"sample_id": garment.annotation_id, "reason": reason})
                else:
                    samples.append((image, garment))
        else:
            raise ValueError("Phase 2 benchmarks support detection or attributes only")
    return samples, excluded


def image_bytes(root: Path, record):
    path = (root.resolve() / record.file_path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("image path escapes root")
    return path.read_bytes()


def load_image(data: bytes):
    with Image.open(BytesIO(data)) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def failed_result(model, model_config, task, code, message):
    return InferenceResult(capability=task, metadata=InferenceMetadata(
        model_name=model, model_id=model_config.model_id or "unconfigured",
        model_revision=model_config.revision, device="not_loaded", latency_seconds=0),
        error=InferenceError(code=code, message=message))


def execution_summary(results):
    priced = [r.metadata for r in results if r.metadata.estimated_cost is not None]
    currencies = {m.cost_currency for m in priced}
    return {"attempted": len(results), "failed": sum(r.error is not None for r in results),
            "failure_codes": dict(Counter(r.error.code for r in results if r.error)),
            "mean_latency_seconds": sum(r.metadata.latency_seconds for r in results) / len(results) if results else None,
            "input_tokens": sum(r.metadata.input_tokens or 0 for r in results),
            "output_tokens": sum(r.metadata.output_tokens or 0 for r in results),
            "thinking_tokens": sum(r.metadata.thinking_tokens or 0 for r in results),
            "usage_records": sum(r.metadata.input_tokens is not None and r.metadata.output_tokens is not None for r in results),
            "estimated_cost": sum(m.estimated_cost for m in priced) if priced and len(currencies) == 1 else None,
            "cost_currency": next(iter(currencies)) if len(currencies) == 1 else None,
            "priced_records": len(priced)}


def benchmark(dataset, dataset_bytes, image_root, model_names, task, config, output_root,
              protocol=None, adapter_factory=create_adapter):
    protocol = protocol or BenchmarkProtocol()
    if not model_names or len(model_names) != len(set(model_names)):
        raise ValueError("provide distinct model names")
    for name in model_names:
        if task not in get_descriptor(name).planned_capabilities:
            raise ValueError(f"{name} does not support {task}; choose a compatible comparison")
        if name not in config.models:
            raise ValueError(f"missing model configuration: {name}")
    samples, excluded = eligible_samples(dataset, task, protocol)
    if not samples:
        reasons = dict(Counter(item["reason"] for item in excluded))
        raise ValueError(f"No eligible samples: {reasons}. Prepare reviewed annotations; --allow-unverified is for smoke tests only.")
    fingerprints = {}
    for record, _ in samples:
        if record.image_id not in fingerprints:
            try:
                fingerprints[record.image_id] = digest(image_bytes(image_root, record))
            except (OSError, ValueError, RuntimeError):
                fingerprints[record.image_id] = None
    comparison = {"task": task, "dataset_sha256": digest(dataset_bytes),
                  "dataset_normalized_sha256": json_digest(dataset.model_dump(mode="json")),
                  "image_sha256": fingerprints, "protocol": protocol.model_dump(mode="json")}
    data_status = "SYNTHETIC / SMOKE TEST ONLY" if dataset.synthetic else (
        "UNVERIFIED DATA ALLOWED" if protocol.allow_unverified else "REVIEWED ANNOTATIONS (as declared)")
    paths = []
    for name in model_names:
        model_config = config.models[name]
        adapter = adapter_factory(name, model_config)
        adapter.seed = config.seed
        writer = RunWriter(output_root, {**comparison, "comparison_key": json_digest(comparison),
            "model": name, "dataset_id": dataset.dataset_id, "data_status": data_status,
            "config": config.model_dump(mode="json"), "config_fingerprint": config.fingerprint(),
            "excluded": excluded, "preprocessing": "EXIF transpose then RGB; crop bounds floor minima/ceil maxima",
            "latency_policy": "wall time including first lazy model load; image decode excluded"})
        results, scores = [], []
        try:
            for record, garment in samples:
                try:
                    data = image_bytes(image_root, record)
                    if fingerprints[record.image_id] is None or digest(data) != fingerprints[record.image_id]:
                        raise ValueError("image changed after manifest snapshot")
                    image = load_image(data)
                    if image.size != (record.width, record.height):
                        raise ValueError("image dimensions do not match annotations")
                except (OSError, ValueError, RuntimeError, Image.DecompressionBombError):
                    result = failed_result(name, model_config, task, "image_error",
                        "Image missing, unreadable, changed, outside root, or dimensions differ from annotations.")
                else:
                    try:
                        result = (adapter.detect_garments(image) if task == "detection"
                                  else adapter.extract_attributes(image, garment.bbox))
                        result = InferenceResult.model_validate_json(result.model_dump_json())
                        if result.capability != task:
                            raise ValueError("adapter returned wrong task")
                        if task == "detection" and result.output and any(
                            g.bbox and not g.bbox.fits(*image.size) for g in result.output.garments):
                            raise ValueError("predicted box out of bounds")
                    except Exception as exc:
                        result = failed_result(name, model_config, task, "adapter_error",
                                               f"Adapter failed its contract ({type(exc).__name__}).")
                    finally:
                        image.close()
                score = (score_detection(record.garments, result.output.garments if result.output else [],
                         protocol.detection_mode, protocol.iou_threshold, protocol.category_aliases)
                         if task == "detection" else score_attributes(garment, result.output, protocol.category_aliases))
                writer.append(garment.annotation_id if garment else record.image_id, record.image_id, result, score)
                results.append(result)
                scores.append(score)
        finally:
            adapter.close()
        metrics = aggregate_detection(scores) if task == "detection" else aggregate_attributes(scores)
        writer.finish({**metrics, **execution_summary(results)})
        paths.append(writer.path)
    return paths


def run_image(image_path, model, task, config, output_root, region=None, adapter_factory=create_adapter):
    if task not in get_descriptor(model).planned_capabilities:
        raise ValueError(f"{model} does not support {task}")
    if model not in config.models:
        raise ValueError(f"missing model configuration: {model}")
    if task == "detection" and region is not None:
        raise ValueError("--region is for attributes/caption/embedding, not whole-image detection")
    data = image_path.read_bytes()
    image = load_image(data)
    adapter = adapter_factory(model, config.models[model])
    adapter.seed = config.seed
    manifest = {"task": task, "model": model, "data_status": "SINGLE IMAGE — NO ACCURACY SCORE",
                "image_sha256": digest(data), "image_file": image_path.name,
                "region": region.model_dump() if region else None,
                "config": config.model_dump(mode="json"), "config_fingerprint": config.fingerprint(),
                "protocol": {"detection_mode": "inference_only"}}
    manifest["comparison_key"] = json_digest({"image": digest(data), "task": task, "region": manifest["region"]})
    writer = RunWriter(output_root, manifest)
    try:
        if task == "detection":
            result = adapter.detect_garments(image)
        elif task == "attributes":
            result = adapter.extract_attributes(image, region)
        elif task == "caption":
            result = adapter.caption(image, region)
        else:
            result = adapter.generate_embedding(image, region)
        writer.append("single-image", "single-image", result)
    finally:
        image.close()
        adapter.close()
    # No benchmark denominators exist for a single-image run.
    metrics = execution_summary([result])
    if task == "detection":
        metrics.update(aggregate_detection([]))
    elif task == "attributes":
        metrics.update(aggregate_attributes([]))
    writer.finish(metrics)
    return writer.path, result
