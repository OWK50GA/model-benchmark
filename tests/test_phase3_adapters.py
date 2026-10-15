import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from fashionista_vision_lab.config import ModelConfig
from fashionista_vision_lab.models.fashionclip import FashionCLIPAdapter
from fashionista_vision_lab.models.dinov3 import DINOv3Adapter
from fashionista_vision_lab.models.base import UnsupportedCapabilityError
from fashionista_vision_lab.schemas.common import BoundingBox


class Tensor:
    def __init__(self, values):
        self.array = np.asarray(values)

    def __getitem__(self, index):
        return Tensor(self.array[index])

    def to(self, *args, **kwargs):
        return self

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.array.tolist()


@pytest.fixture
def local_runtime(monkeypatch):
    calls = []

    class Processor:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("processor_load", model_id, kwargs))
            return cls()

        def __call__(self, images, return_tensors):
            calls.append(("image", images.size))
            return {"pixel_values": Tensor([1])}

        def to_dict(self):
            return {"size": 224, "test_only": True}

    class Model:
        config = SimpleNamespace(model_type="dinov3_vit")

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("model_load", model_id, kwargs))
            return cls()

        def to(self, device):
            return self

        def eval(self):
            calls.append(("eval",))
            return self

        def get_image_features(self, **kwargs):
            calls.append(("clip_features",))
            return Tensor([[3, 4]])

        def __call__(self, **kwargs):
            calls.append(("dino_features",))
            return SimpleNamespace(last_hidden_state=Tensor([[[3, 4], [99, 99], [-99, -99]]]))

    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), backends=SimpleNamespace(),
        manual_seed=lambda seed: calls.append(("seed", seed)), float32="float32", inference_mode=nullcontext)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoImageProcessor=Processor,
        AutoModel=Model, CLIPImageProcessor=Processor, CLIPModel=Model))
    monkeypatch.setattr("fashionista_vision_lab.models.embedding.version", lambda name: "fixture-version")
    return calls, Model


@pytest.mark.parametrize("adapter_class", [FashionCLIPAdapter, DINOv3Adapter])
def test_loading_pooling_normalization_crop_and_cleanup(local_runtime, adapter_class):
    calls, _ = local_runtime
    adapter = adapter_class(ModelConfig(model_id="fixture", revision="a" * 40))
    output = adapter.generate_embedding(Image.new("RGB", (20, 30)),
        BoundingBox(x_min=1, y_min=2, x_max=10, y_max=20))
    assert output.error is None, output.error
    assert output.output.vector == pytest.approx([0.6, 0.8])
    assert output.output.normalized
    assert ("image", (9, 18)) in calls
    load = next(call for call in calls if call[0] == "model_load")
    assert load[2]["trust_remote_code"] is False
    assert load[2]["local_files_only"] is True
    assert load[2]["torch_dtype"] == "float32"
    assert ("clip_features",) in calls if adapter.name == "fashionclip" else ("dino_features",) in calls
    adapter.generate_embedding(Image.new("RGB", (20, 30)))
    assert len([call for call in calls if call[0] == "model_load"]) == 1
    adapter.close()
    assert adapter._model is None


def test_checkpoint_revision_changes_space(local_runtime):
    outputs = []
    for revision in ("a" * 40, "b" * 40):
        outputs.append(FashionCLIPAdapter(ModelConfig(model_id="fixture", revision=revision))
                       .generate_embedding(Image.new("RGB", (10, 10))).output)
    assert outputs[0].embedding_space != outputs[1].embedding_space


def test_pinning_access_and_capability_failures(monkeypatch):
    adapter = DINOv3Adapter(ModelConfig(model_id="fixture"))
    assert "40-character" in adapter.generate_embedding(Image.new("RGB", (2, 2))).error.message
    monkeypatch.delenv("TEST_DINO_TOKEN", raising=False)
    with pytest.raises(UnsupportedCapabilityError):
        adapter.detect_garments(Image.new("RGB", (2, 2)))


def test_gated_access_missing_key_and_unsupported_architecture(local_runtime, monkeypatch):
    _, model_class = local_runtime
    monkeypatch.delenv("TEST_DINO_TOKEN", raising=False)
    adapter = DINOv3Adapter(ModelConfig(model_id="fixture", revision="a" * 40, credential_env="TEST_DINO_TOKEN"))
    assert "TEST_DINO_TOKEN" in adapter.generate_embedding(Image.new("RGB", (2, 2))).error.message
    model_class.config = SimpleNamespace(model_type="dinov3_convnext")
    adapter = DINOv3Adapter(ModelConfig(model_id="fixture", revision="a" * 40))
    assert "CLS" in adapter.generate_embedding(Image.new("RGB", (2, 2))).error.message


def test_zero_embedding_is_failure(local_runtime, monkeypatch):
    _, model_class = local_runtime
    monkeypatch.setattr(model_class, "get_image_features", lambda self, **kwargs: Tensor([[0, 0]]))
    result = FashionCLIPAdapter(ModelConfig(model_id="fixture", revision="a" * 40)).generate_embedding(Image.new("RGB", (2, 2)))
    assert result.error.code == "invalid_response_or_input"
    assert result.raw_response["vector"] == [0, 0]
