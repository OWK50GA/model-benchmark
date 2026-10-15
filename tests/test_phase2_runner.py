import json
from pathlib import Path

import pytest
from PIL import Image

from fashionista_vision_lab.config import LabConfig, ModelConfig
from fashionista_vision_lab.models.base import ModelAdapter
from fashionista_vision_lab.reporting import render_report
from fashionista_vision_lab.runner import BenchmarkProtocol, benchmark, eligible_samples, run_image
from fashionista_vision_lab.schemas.dataset import Dataset
from fashionista_vision_lab.schemas.outputs import (AttributeOutput, Detection, DetectionOutput, InferenceError,
                                                    InferenceMetadata, InferenceResult)


class FixtureAdapter(ModelAdapter):
    """Test-only predictions. No model-performance claims."""
    closed = False
    fail = False

    def metadata(self):
        return InferenceMetadata(model_name="fixture", model_id="fixture-only", device="cpu", latency_seconds=0.1)

    def detect_garments(self, image):
        if self.fail:
            return InferenceResult(capability="detection", metadata=self.metadata(),
                error=InferenceError(code="test_failure", message="test-only"), raw_response={"fixture": True})
        return InferenceResult(capability="detection", metadata=self.metadata(),
            output=DetectionOutput(garments=[Detection(category="shirt")]), raw_response={"fixture": True})

    def extract_attributes(self, image, region=None):
        return InferenceResult(capability="attributes", metadata=self.metadata(), output=AttributeOutput(category="shirt"),
                               raw_response={"fixture": True})

    def close(self):
        self.closed = True


@pytest.fixture
def setup_data(example_data, tmp_path):
    example_data["images"] = example_data["images"][:1]
    dataset = Dataset.model_validate(example_data)
    path = tmp_path / dataset.images[0].file_path
    path.parent.mkdir()
    Image.new("RGB", (640, 800)).save(path)
    config = LabConfig(models={"gemini": ModelConfig(model_id="test"), "florence": ModelConfig(model_id="test")})
    return dataset, dataset.model_dump_json().encode(), config


def test_eligibility_filters_and_ambiguous_targets(example_data):
    dataset = Dataset.model_validate(example_data)
    samples, excluded = eligible_samples(dataset, "detection", BenchmarkProtocol())
    assert not samples and len(excluded) == 3
    samples, excluded = eligible_samples(dataset, "detection", BenchmarkProtocol(allow_unverified=True))
    assert len(samples) == 2 and excluded[0]["reason"] == "missing_ground_truth_boxes"
    samples, excluded = eligible_samples(dataset, "attributes", BenchmarkProtocol(allow_unverified=True))
    assert len(samples) == 3 and excluded[0]["sample_id"] == "trousers-worn"


def test_benchmark_preserves_artifacts_and_compares(setup_data, tmp_path):
    dataset, data, config = setup_data
    adapters = []

    def factory(name, config):
        adapter = FixtureAdapter(config)
        adapters.append(adapter)
        return adapter

    paths = benchmark(dataset, data, tmp_path, ["gemini", "florence"], "detection", config, tmp_path / "results",
                      BenchmarkProtocol(allow_unverified=True, detection_mode="categories"), factory)
    assert all(adapter.closed for adapter in adapters)
    assert len(paths) == 2 and paths[0] != paths[1]
    for path in paths:
        metrics = json.loads((path / "metrics.json").read_text())
        assert metrics["recall"] == 1 and metrics["precision"] == 1
        normalized = json.loads((path / "normalized.jsonl").read_text())
        raw = json.loads((path / "raw.jsonl").read_text())
        assert "raw_response" not in normalized["result"]
        assert raw["sample_id"] == normalized["sample_id"]
        assert raw["response"] == {"fixture": True}
        assert json.loads((path / "manifest.json").read_text())["image_sha256"]["example-front"]
    report = render_report(paths)
    assert "SYNTHETIC / SMOKE TEST ONLY" in report
    assert "gemini" in report and "florence" in report


def test_failed_inference_and_missing_image_stay_in_denominators(setup_data, tmp_path):
    dataset, data, config = setup_data
    adapter = FixtureAdapter(ModelConfig())
    adapter.fail = True
    options = BenchmarkProtocol(allow_unverified=True, detection_mode="categories")
    paths = benchmark(dataset, data, tmp_path, ["gemini"], "detection", config, tmp_path / "results", options, lambda *a: adapter)
    metrics = json.loads((paths[0] / "metrics.json").read_text())
    assert metrics["failed"] == 1 and metrics["fn"] == 1 and metrics["recall"] == 0
    paths = benchmark(dataset, data, tmp_path / "missing", ["gemini"], "detection", config, tmp_path / "results", options, lambda *a: adapter)
    metrics = json.loads((paths[0] / "metrics.json").read_text())
    assert metrics["failure_codes"] == {"image_error": 1} and metrics["fn"] == 1


def test_attributes_benchmark_and_incompatible_model(setup_data, tmp_path):
    dataset, data, config = setup_data
    paths = benchmark(dataset, data, tmp_path, ["gemini"], "attributes", config, tmp_path / "results",
                      BenchmarkProtocol(allow_unverified=True), lambda *a: FixtureAdapter(ModelConfig()))
    metrics = json.loads((paths[0] / "metrics.json").read_text())
    assert metrics["category_accuracy"] == 1
    assert metrics["per_attribute"]["colors"]["missing"] == 1
    with pytest.raises(ValueError, match="does not support"):
        benchmark(dataset, data, tmp_path, ["florence"], "attributes", config, tmp_path / "results")


def test_no_eligible_data_never_loads_model(setup_data, tmp_path):
    dataset, data, config = setup_data
    with pytest.raises(ValueError, match="No eligible"):
        benchmark(dataset, data, tmp_path, ["gemini"], "detection", config, tmp_path / "results",
                  adapter_factory=lambda *a: pytest.fail("must not load"))
    assert not (tmp_path / "results").exists()


def test_reports_reject_different_protocols(setup_data, tmp_path):
    dataset, data, config = setup_data
    paths = []
    for mode in ("boxes", "categories"):
        paths += benchmark(dataset, data, tmp_path, ["gemini"], "detection", config, tmp_path / "results",
            BenchmarkProtocol(allow_unverified=True, detection_mode=mode), lambda *a: FixtureAdapter(ModelConfig()))
    with pytest.raises(ValueError, match="different dataset"):
        render_report(paths)


def test_image_changed_after_fingerprinting_is_failure(setup_data, tmp_path):
    dataset, data, config = setup_data

    def factory(*args):
        Image.new("RGB", (640, 800), "red").save(tmp_path / dataset.images[0].file_path)
        return FixtureAdapter(ModelConfig())

    paths = benchmark(dataset, data, tmp_path, ["gemini"], "detection", config, tmp_path / "results",
        BenchmarkProtocol(allow_unverified=True), factory)
    assert json.loads((paths[0] / "metrics.json").read_text())["failure_codes"] == {"image_error": 1}


def test_single_image_failure_is_saved_without_fake_scores(tmp_path):
    image = tmp_path / "test.jpg"
    Image.new("RGB", (10, 10)).save(image)
    config = LabConfig(models={"gemini": ModelConfig()})
    path, result = run_image(image, "gemini", "attributes", config, tmp_path / "results")
    assert result.error.code == "setup_error"
    metrics = json.loads((path / "metrics.json").read_text())
    assert metrics["category_accuracy"] is None
    assert "SINGLE IMAGE" in (path / "report.md").read_text()


def test_alias_chains_rejected():
    with pytest.raises(ValueError, match="alias chains"):
        BenchmarkProtocol(category_aliases={"tee": "t-shirt", "t-shirt": "shirt"})


def test_cli_benchmark_and_report_without_network(setup_data, tmp_path, capsys, monkeypatch):
    from fashionista_vision_lab.cli import main
    dataset, data, config = setup_data
    config.models["gemini"] = ModelConfig()  # missing ID causes an explicit saved setup failure
    config_path, dataset_path = tmp_path / "config.json", tmp_path / "data.json"
    config_path.write_text(config.model_dump_json())
    dataset_path.write_bytes(data)
    args = ["benchmark", "--model", "gemini", "--task", "detection", "--config", str(config_path),
            "--dataset", str(dataset_path), "--image-root", str(tmp_path), "--output-root", str(tmp_path / "results"),
            "--allow-unverified", "--detection-mode", "categories"]
    assert main(args) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["failures"] == 1
    report_path = tmp_path / "report.md"
    assert main(["report", payload["run_dirs"][0], "--output", str(report_path)]) == 0
    assert report_path.exists()
    assert main(["report", payload["run_dirs"][0], "--output", str(report_path)]) == 1


def test_cli_compare_writes_report_even_when_models_need_setup(setup_data, tmp_path, capsys):
    from fashionista_vision_lab.cli import main
    dataset, data, config = setup_data
    config.models["gemini"] = ModelConfig()
    config_path = tmp_path / "config.json"
    dataset_path = tmp_path / "data.json"
    config_path.write_text(config.model_dump_json())
    dataset_path.write_bytes(data)
    assert main(["compare", "--models", "gemini", "florence", "--task", "detection",
        "--config", str(config_path), "--dataset", str(dataset_path), "--image-root", str(tmp_path),
        "--output-root", str(tmp_path / "results"), "--allow-unverified"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["failures"] == 2
    report = Path(payload["comparison_report"]).read_text()
    assert "gemini" in report and "florence" in report and "setup_error" in report
