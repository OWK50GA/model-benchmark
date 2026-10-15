import math

import pytest
from PIL import Image
from pydantic import ValidationError

from fashionista_vision_lab.config import LabConfig, ModelConfig, resolve_config_path
from fashionista_vision_lab.models.base import AdapterNotImplementedError, ModelAdapter, UnsupportedCapabilityError
from fashionista_vision_lab.models.registry import ModelDescriptor, REGISTRY, create_adapter
from fashionista_vision_lab.schemas.outputs import (
    AttributeOutput, DetectionOutput, EmbeddingOutput, InferenceError, InferenceMetadata, InferenceResult,
)


def metadata():
    return InferenceMetadata(model_name="test-only", model_id="fixture", device="cpu", latency_seconds=0)


@pytest.mark.parametrize("capability,output", [
    ("detection", DetectionOutput(garments=[])),
    ("attributes", AttributeOutput()),
    ("embedding", EmbeddingOutput(vector=[1, 0], embedding_space="test-only", normalized=True)),
])
def test_result_serialization_preserves_output_and_raw(capability, output):
    result = InferenceResult(capability=capability, metadata=metadata(), output=output, raw_response={"test_only": True})
    assert InferenceResult.model_validate_json(result.model_dump_json()) == result


def test_result_requires_success_or_failure_not_both():
    with pytest.raises(ValidationError):
        InferenceResult(capability="detection", metadata=metadata())
    error = InferenceError(code="test-error", message="test-only")
    result = InferenceResult(capability="detection", metadata=metadata(), error=error)
    assert result.output is None
    with pytest.raises(ValidationError):
        InferenceResult(capability="detection", metadata=metadata(), error=error, output=DetectionOutput(garments=[]))
    with pytest.raises(ValidationError, match="match capability"):
        InferenceResult(capability="embedding", metadata=metadata(), output=AttributeOutput())


@pytest.mark.parametrize("vector", [[], [0, 0], [math.nan], [math.inf], [-math.inf]])
def test_invalid_embeddings_rejected(vector):
    with pytest.raises(ValidationError):
        EmbeddingOutput(vector=vector, embedding_space="test")


def test_normalized_claim_checked():
    with pytest.raises(ValidationError, match="unit norm"):
        EmbeddingOutput(vector=[2, 0], embedding_space="test", normalized=True)


def test_cost_requires_pricing_basis_and_currency():
    with pytest.raises(ValidationError):
        InferenceMetadata(**metadata().model_dump(exclude_none=True), estimated_cost=1)


def test_capabilities_fail_explicitly_without_inference():
    adapter = ModelAdapter(ModelConfig())
    with pytest.raises(UnsupportedCapabilityError):
        adapter.detect_garments(Image.new("RGB", (2, 2)))
    for name in ("fashionclip", "dinov3"):
        assert create_adapter(name, ModelConfig()).name == name
    assert create_adapter("gemini", ModelConfig()).name == "gemini"
    assert create_adapter("florence", ModelConfig()).name == "florence"
    with pytest.raises(ValueError, match="unknown model"):
        create_adapter("unknown", ModelConfig())


def test_registry_imports_only_requested_factory(monkeypatch):
    from fashionista_vision_lab.models import registry
    from types import SimpleNamespace

    imported = []

    def import_test_module(name):
        imported.append(name)
        return SimpleNamespace(Adapter=ModelAdapter)

    monkeypatch.setitem(REGISTRY, "test-only", ModelDescriptor("test-only", (), 1, "test", "test.module:Adapter"))
    monkeypatch.setattr(registry, "import_module", import_test_module)
    adapter = create_adapter("test-only", ModelConfig())
    assert isinstance(adapter, ModelAdapter)
    assert imported == ["test.module"]


def test_config_fingerprint_is_order_independent_and_sensitive():
    a = LabConfig(models={"a": ModelConfig(), "b": ModelConfig()})
    b = LabConfig(models={"b": ModelConfig(), "a": ModelConfig()})
    assert a.fingerprint() == b.fingerprint()
    b.seed = 3
    assert a.fingerprint() != b.fingerprint()


def test_config_paths_are_relative_to_config_file(tmp_path):
    assert resolve_config_path(tmp_path / "config/default.json", "../datasets") == tmp_path / "datasets"


def test_credentials_are_lazy_and_never_serialized(monkeypatch):
    monkeypatch.delenv("LAB_TEST_SECRET", raising=False)
    model = ModelConfig(credential_env="LAB_TEST_SECRET")
    with pytest.raises(ValueError, match="missing credential"):
        model.require_credential()
    monkeypatch.setenv("LAB_TEST_SECRET", "secret-value")
    assert model.require_credential() == "secret-value"
    assert "secret-value" not in model.model_dump_json()
    with pytest.raises(ValidationError):
        ModelConfig(api_key="secret-value")
