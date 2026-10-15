import json

import pytest
from PIL import Image
from pydantic import ValidationError

from fashionista_vision_lab.datasets import check_images, load_dataset, save_dataset, summarize_dataset
from fashionista_vision_lab.schemas.dataset import Dataset, GarmentAttributes


def test_example_demonstrates_identity_and_hard_negatives(example_data):
    dataset = Dataset.model_validate(example_data)
    summary = summarize_dataset(dataset)
    assert (summary["images"], summary["garments"], summary["unique_identities"]) == (3, 4, 3)
    assert summary["identities_in_multiple_images"] == 1
    a, b = dataset.images[0].garments[0], dataset.images[2].garments[0]
    assert a.appearance_group == b.appearance_group and a.identity_id != b.identity_id
    assert summary["warnings"]


@pytest.mark.parametrize("field", ["image_id", "file_path"])
def test_duplicate_images_rejected(example_data, field):
    example_data["images"][1][field] = example_data["images"][0][field]
    with pytest.raises(ValidationError, match=f"duplicate {field}"):
        Dataset.model_validate(example_data)


def test_duplicate_annotation_ids_rejected(example_data):
    example_data["images"][1]["garments"][0]["annotation_id"] = "a-front"
    with pytest.raises(ValidationError, match="duplicate annotation_id"):
        Dataset.model_validate(example_data)


@pytest.mark.parametrize("path", ["../outside.jpg", "/tmp/image.jpg", "C:/image.jpg", "C:image.jpg", "a\\b.jpg", "a/../../x.jpg", ".", "a/", "bad\x00.jpg"])
def test_unsafe_paths_rejected(example_data, path):
    example_data["images"][0]["file_path"] = path
    with pytest.raises(ValidationError):
        Dataset.model_validate(example_data)


@pytest.mark.parametrize("update", [{"x_min": -1}, {"x_max": 80}, {"x_max": 641}, {"y_max": float("inf")}])
def test_invalid_boxes_rejected(example_data, update):
    example_data["images"][0]["garments"][0]["bbox"].update(update)
    with pytest.raises(ValidationError):
        Dataset.model_validate(example_data)


def test_synthetic_cannot_claim_verified_ground_truth(example_data):
    example_data["images"][0]["garments"][0].update(verification="verified", verified_by="reviewer")
    with pytest.raises(ValidationError, match="synthetic datasets"):
        Dataset.model_validate(example_data)


def test_real_verified_annotations_need_reviewer(example_data):
    example_data["synthetic"] = False
    for image in example_data["images"]:
        for garment in image["garments"]:
            garment["verification"] = "unverified"
    garment = example_data["images"][0]["garments"][0]
    garment["verification"] = "verified"
    with pytest.raises(ValidationError, match="verified_by"):
        Dataset.model_validate(example_data)
    garment["verified_by"] = "reviewer-1"
    assert Dataset.model_validate(example_data).images[0].garments[0].verification == "verified"


def test_unknown_is_different_from_reviewed_absence():
    assert GarmentAttributes().distinctive_details is None
    assert GarmentAttributes(distinctive_details=[]).distinctive_details == []
    with pytest.raises(ValidationError):
        GarmentAttributes(colors=["blue", " blue "])
    with pytest.raises(ValidationError):
        GarmentAttributes(material_composition="cotton")


@pytest.mark.parametrize("extension", ["json", "jsonl"])
def test_dataset_round_trip_without_overwrite(example_data, tmp_path, extension):
    dataset = Dataset.model_validate(example_data)
    path = tmp_path / f"dataset.{extension}"
    save_dataset(dataset, path)
    assert load_dataset(path) == dataset
    with pytest.raises(FileExistsError):
        save_dataset(dataset, path)


def test_malformed_jsonl_reports_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{}\n{"bad":\n')
    with pytest.raises(ValueError, match="line 2"):
        load_dataset(path)


def test_jsonl_cannot_silently_replace_embedded_images(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"images": []}\n')
    with pytest.raises(ValueError, match="must not contain images"):
        load_dataset(path)


def test_image_validation_reports_missing_dimensions_and_corruption(example_data, tmp_path):
    dataset = Dataset.model_validate(example_data)
    assert len(check_images(dataset, tmp_path)) == 3
    (tmp_path / "example").mkdir()
    for record in dataset.images:
        Image.new("RGB", (record.width, record.height)).save(tmp_path / record.file_path)
    assert check_images(dataset, tmp_path) == []
    Image.new("RGB", (10, 10)).save(tmp_path / dataset.images[0].file_path)
    (tmp_path / dataset.images[1].file_path).write_text("corrupt")
    assert len(check_images(dataset, tmp_path)) == 2


def test_exif_orientation_is_applied(example_data, tmp_path):
    example_data["images"] = example_data["images"][:1]
    record = example_data["images"][0]
    record["file_path"] = "oriented.jpg"
    image = Image.new("RGB", (800, 640))
    exif = image.getexif()
    exif[274] = 6
    image.save(tmp_path / "oriented.jpg", exif=exif)
    assert check_images(Dataset.model_validate(example_data), tmp_path) == []


def test_symlink_escape_is_rejected(example_data, tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (root / "example").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    errors = check_images(Dataset.model_validate(example_data), root)
    assert len(errors) == 3
    assert all("outside image root" in message for message in errors)


def test_cross_split_identity_warning(example_data):
    example_data["images"][0]["split"] = "train"
    warnings = summarize_dataset(Dataset.model_validate(example_data))["warnings"]
    assert any("leakage: shirt-a" in warning for warning in warnings)
