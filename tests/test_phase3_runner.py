import json

import pytest
from PIL import Image

from fashionista_vision_lab.benchmarks.identity import IdentityProtocol, neighbors
from fashionista_vision_lab.config import LabConfig, ModelConfig
from fashionista_vision_lab.identity_runner import identity_benchmark, read_observations, select_samples
from fashionista_vision_lab.models.base import ModelAdapter
from fashionista_vision_lab.reporting import render_report
from fashionista_vision_lab.schemas.dataset import Dataset
from fashionista_vision_lab.schemas.outputs import EmbeddingOutput, InferenceMetadata, InferenceResult


class FixtureEncoder(ModelAdapter):
    calls = 0
    closed = False

    def generate_embedding(self, image, region=None):
        self.calls += 1
        red = image.getpixel((0, 0))[0]
        return InferenceResult(capability="embedding", metadata=InferenceMetadata(
            model_name="fixture", model_id="fixture", device="cpu", latency_seconds=0.1),
            output=EmbeddingOutput(vector=[1, red / 255], embedding_space="fixture"), raw_response={"test_only": True})

    def close(self):
        self.closed = True


@pytest.fixture
def identity_data(tmp_path):
    images = []
    for index, identity in enumerate(("a", "a", "b")):
        filename = f"image{index}.png"
        Image.new("RGB", (10, 10), (index * 50, 0, 0)).save(tmp_path / filename)
        images.append({"image_id": str(index), "file_path": filename, "width": 10, "height": 10,
            "split": "test", "garments_complete": True, "garments": [{"annotation_id": f"g{index}",
                "identity_id": identity, "appearance_group": "shirts", "viewpoint": "front" if index != 1 else "back",
                "verification": "example"}]})
    dataset = Dataset(dataset_id="fixtures-only", description="synthetic fixture", synthetic=True, images=images)
    config = LabConfig(models={name: ModelConfig(model_id="fixture", revision="a" * 40) for name in ("fashionclip", "dinov3")})
    return dataset, config


def test_identity_runs_cache_neighbors_and_comparison(identity_data, tmp_path):
    dataset, config = identity_data
    encoders = []

    def factory(*args):
        adapter = FixtureEncoder(ModelConfig())
        encoders.append(adapter)
        return adapter

    args = (dataset, dataset.model_dump_json().encode(), tmp_path, ["fashionclip", "dinov3"], config,
            tmp_path / "results", IdentityProtocol(threshold=0.8, allow_unverified=True), tmp_path / "cache", factory)
    paths = identity_benchmark(*args)
    assert all(e.calls == 3 and e.closed for e in encoders)
    again = identity_benchmark(*args)
    assert all(e.calls == 0 and e.closed for e in encoders[2:])
    assert json.loads((again[0] / "metrics.json").read_text())["cache_hits"] == 3
    assert "SYNTHETIC" in render_report(paths)
    assert (paths[0] / "pairs.jsonl").exists() and (paths[0] / "retrieval.jsonl").exists()
    observations = read_observations(paths[0])
    assert len(neighbors(observations, "g0")) == 2
    from fashionista_vision_lab.cli import main
    assert main(["neighbors", str(paths[0]), "--query", "g0", "--show-vector"]) == 0


def test_identity_filters_and_cross_split_leakage(identity_data):
    dataset, _ = identity_data
    assert not select_samples(dataset, IdentityProtocol(threshold=0.5))[0]
    data = dataset.model_dump()
    data["images"][1]["split"] = "train"
    data["images"][2]["garments"][0]["identity_id"] = None
    samples, excluded = select_samples(Dataset.model_validate(data), IdentityProtocol(threshold=0.5, allow_unverified=True))
    assert not samples
    assert {e["reason"] for e in excluded} == {"identity_spans_splits", "different_split", "unknown_physical_identity"}


def test_missing_images_and_setup_failures_are_not_perfect_scores(identity_data, tmp_path):
    dataset, config = identity_data
    paths = identity_benchmark(dataset, b"fixture", tmp_path / "missing", ["dinov3"], config, tmp_path / "results",
        IdentityProtocol(threshold=0.8, allow_unverified=True), adapter_factory=lambda *args: FixtureEncoder(ModelConfig()))
    metrics = json.loads((paths[0] / "metrics.json").read_text())
    assert metrics["failed"] == 3 and metrics["retrieval"]["all_views"]["recall_at_1"] == 0
    assert metrics["matching"]["false_positive_rate"] is None


def test_exact_duplicate_pixels_are_excluded_across_different_files(identity_data, tmp_path):
    dataset, config = identity_data
    Image.new("RGB", (10, 10)).save(tmp_path / "image1.png")
    paths = identity_benchmark(dataset, b"fixture", tmp_path, ["fashionclip"], config, tmp_path / "results",
        IdentityProtocol(threshold=0.8, allow_unverified=True), adapter_factory=lambda *args: FixtureEncoder(ModelConfig()))
    metrics = json.loads((paths[0] / "metrics.json").read_text())
    assert metrics["retrieval"]["all_views"]["eligible_queries"] == 0
    assert metrics["matching"]["same_pairs"] == 0


def test_reports_reject_different_identity_thresholds(identity_data, tmp_path):
    dataset, config = identity_data
    paths = []
    for threshold in (0.5, 0.9):
        paths += identity_benchmark(dataset, b"fixture", tmp_path, ["fashionclip"], config, tmp_path / "results",
            IdentityProtocol(threshold=threshold, allow_unverified=True), adapter_factory=lambda *args: FixtureEncoder(ModelConfig()))
    with pytest.raises(ValueError, match="different dataset"):
        render_report(paths)


def test_cli_identity_requires_explicit_threshold_and_saves_setup_failures(identity_data, tmp_path, capsys):
    from fashionista_vision_lab.cli import main
    dataset, config = identity_data
    for value in config.models.values():
        value.revision = None
    data_path, config_path = tmp_path / "data.json", tmp_path / "config.json"
    data_path.write_text(dataset.model_dump_json())
    config_path.write_text(config.model_dump_json())
    args = ["compare", "--task", "identity", "--models", "fashionclip", "dinov3", "--dataset", str(data_path),
        "--config", str(config_path), "--image-root", str(tmp_path), "--output-root", str(tmp_path / "results"), "--allow-unverified"]
    assert main(args) == 1
    assert "--match-threshold" in capsys.readouterr().err
    assert main(args + ["--match-threshold", "0.8"]) == 1
    response = json.loads(capsys.readouterr().out)
    assert response["failures"] == 6
    assert len(response["run_dirs"]) == 2
