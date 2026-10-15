"""Dataset I/O and optional local image checks; no inference or downloads."""

import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps

from .schemas.dataset import Dataset


def load_dataset(path: Path) -> Dataset:
    """JSON contains one Dataset. JSONL starts with metadata, then image records."""
    return parse_dataset(path.read_text(encoding="utf-8"), path.suffix)


def parse_dataset(text: str, suffix: str = ".json") -> Dataset:
    if suffix.lower() != ".jsonl":
        return Dataset.model_validate_json(text)
    records = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at line {line_number}: {exc.msg}") from exc
    if not records or not isinstance(records[0], dict):
        raise ValueError("JSONL requires a dataset metadata object on the first nonempty line")
    if "images" in records[0]:
        raise ValueError("JSONL metadata must not contain images; use following lines")
    return Dataset.model_validate({**records[0], "images": records[1:]})


def save_dataset(dataset: Dataset, path: Path) -> None:
    """Explicit conversion helper. Existing files are never overwritten."""
    data = dataset.model_dump(mode="json")
    if path.suffix.lower() == ".jsonl":
        images = data.pop("images")
        text = "\n".join(json.dumps(record, ensure_ascii=False, allow_nan=False) for record in [data, *images])
    else:
        text = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text + "\n")


def check_images(dataset: Dataset, image_root: Path) -> list[str]:
    """Validate readable files and EXIF-oriented dimensions without mutating images."""
    root = image_root.resolve()
    errors = []
    for record in dataset.images:
        try:
            path = (root / record.file_path).resolve()
            if not path.is_relative_to(root):
                raise ValueError("image resolves outside image root")
            with Image.open(path) as source:
                source.load()
                oriented = ImageOps.exif_transpose(source)
                if oriented.size != (record.width, record.height):
                    raise ValueError(f"expected {record.width}x{record.height}, found {oriented.width}x{oriented.height}")
        except (OSError, ValueError, Image.DecompressionBombError, RuntimeError) as exc:
            errors.append(f"{record.image_id}: {exc}")
    return errors


def summarize_dataset(dataset: Dataset) -> dict:
    garments = [g for record in dataset.images for g in record.garments]
    identity_splits: dict[str, set[str]] = defaultdict(set)
    identity_images: dict[str, set[str]] = defaultdict(set)
    for record in dataset.images:
        for garment in record.garments:
            if garment.identity_id:
                identity_images[garment.identity_id].add(record.image_id)
                if record.split != "unspecified":
                    identity_splits[garment.identity_id].add(record.split)
    warnings = []
    if dataset.synthetic:
        warnings.append("Synthetic examples are not verified ground truth or benchmark evidence.")
    if any(g.verification != "verified" for g in garments):
        warnings.append("Unverified/example annotations must not be treated as verified ground truth.")
    if any(not record.garments_complete for record in dataset.images):
        warnings.append("Some images have incomplete garment annotations; detection recall needs complete labels.")
    leaked = sorted(identity for identity, splits in identity_splits.items() if len(splits) > 1)
    if leaked:
        warnings.append("Physical identities span dataset splits; review leakage: " + ", ".join(leaked))
    return {
        "dataset_id": dataset.dataset_id,
        "synthetic": dataset.synthetic,
        "images": len(dataset.images),
        "garments": len(garments),
        "with_boxes": sum(g.bbox is not None for g in garments),
        "with_identity": sum(g.identity_id is not None for g in garments),
        "unique_identities": len(identity_images),
        "identities_in_multiple_images": sum(len(images) > 1 for images in identity_images.values()),
        "verification": dict(sorted(Counter(g.verification for g in garments).items())),
        "contexts": dict(sorted(Counter(record.context for record in dataset.images).items())),
        "warnings": warnings,
    }
