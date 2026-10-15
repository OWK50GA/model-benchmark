import json
from types import SimpleNamespace

import pytest
from PIL import Image

from fashionista_vision_lab.config import ModelConfig
from fashionista_vision_lab.models.gemini import GeminiAdapter, response_schema
from fashionista_vision_lab.models.florence import FlorenceAdapter, select_device
from fashionista_vision_lab.models.runtime import SetupError, crop_image
from fashionista_vision_lab.models.base import UnsupportedCapabilityError
from fashionista_vision_lab.schemas.common import BoundingBox


class Response:
    def __init__(self, text):
        self.text = text

    def model_dump(self, **kwargs):
        return {"model_version": "test-version", "usage_metadata": {
            "prompt_token_count": 100, "candidates_token_count": 20, "thoughts_token_count": 5}}


def stub_gemini(text, config=None):
    adapter = GeminiAdapter(config or ModelConfig(model_id="test-only"))
    calls = []

    def generate_content(**kwargs):
        calls.append(kwargs)
        return Response(text)

    adapter._client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content), close=lambda: None)
    return adapter, calls


def test_gemini_coordinate_conversion_usage_and_request():
    adapter, calls = stub_gemini('{"garments":[{"category":"shirt","box_2d":[100,200,900,800]}]}')
    result = adapter.detect_garments(Image.new("RGB", (200, 100)))
    assert result.error is None
    assert result.output.garments[0].bbox.model_dump() == dict(x_min=40, y_min=10, x_max=160, y_max=90)
    assert result.metadata.input_tokens == 100 and result.metadata.thinking_tokens == 5
    assert result.metadata.resolved_model_version == "test-version"
    assert calls[0]["model"] == "test-only"
    assert calls[0]["config"]["response_json_schema"] == response_schema("detection")
    assert result.raw_response["text"]


@pytest.mark.parametrize("text", ['not json', '{"garments":[{"category":"shirt","box_2d":[0,0,1001,900]}]}',
    '{"garments":[{"category":"shirt","box_2d":[100,100,0,0]}]}', '{"garments": null}', None])
def test_gemini_malformed_responses_preserve_raw_and_usage(text):
    adapter, _ = stub_gemini(text)
    result = adapter.detect_garments(Image.new("RGB", (20, 20)))
    assert result.error.code == "invalid_response_or_input"
    assert result.raw_response["text"] == text
    assert result.metadata.input_tokens == 100


def test_gemini_crops_without_sending_ground_truth_and_prices_usage():
    config = ModelConfig(model_id="test", input_cost_per_million=1, output_cost_per_million=2,
                         cost_currency="USD", cost_basis="test fixture pricing")
    adapter, calls = stub_gemini('{"category":"shirt","attributes":{"colors":null}}', config)
    result = adapter.extract_attributes(Image.new("RGB", (100, 100)), BoundingBox(x_min=10, y_min=20, x_max=50, y_max=70))
    assert result.error is None and result.output.attributes.colors is None
    assert calls[0]["contents"][1].size == (40, 50)
    assert result.metadata.estimated_cost == pytest.approx(0.00015)


def test_missing_config_key_and_provider_exception_are_safe(monkeypatch):
    assert GeminiAdapter(ModelConfig()).detect_garments(Image.new("RGB", (2, 2))).error.code == "setup_error"
    monkeypatch.delenv("LAB_MISSING_KEY", raising=False)
    adapter = GeminiAdapter(ModelConfig(model_id="test", credential_env="LAB_MISSING_KEY"))
    assert "LAB_MISSING_KEY" in adapter.detect_garments(Image.new("RGB", (2, 2))).error.message
    adapter, _ = stub_gemini("unused")

    def fail(**kwargs):
        raise RuntimeError("secret-token-do-not-print")

    adapter._client.models.generate_content = fail
    result = adapter.detect_garments(Image.new("RGB", (2, 2)))
    assert result.error.code == "inference_error"
    assert "secret-token" not in result.model_dump_json()


def test_florence_native_detection_and_caption(monkeypatch):
    adapter = FlorenceAdapter(ModelConfig(model_id="test-only"))
    monkeypatch.setattr(adapter, "prepare", lambda: None)
    monkeypatch.setattr(adapter, "request", lambda image, task: ({"text": "raw-test"}, {}))
    adapter._processor = SimpleNamespace(post_process_generation=lambda text, task, image_size: {
        "<OD>": {"labels": ["shirt", "person"], "bboxes": [[0, 0, 10, 10], [0, 0, 20, 20]]},
        "<MORE_DETAILED_CAPTION>": "Test-only caption"})
    result = adapter.detect_garments(Image.new("RGB", (20, 20)))
    assert result.error is None and len(result.output.garments) == 2
    assert result.raw_response["parsed"]["<OD>"]["labels"] == ["shirt", "person"]
    assert adapter.caption(Image.new("RGB", (20, 20))).output.text == "Test-only caption"
    with pytest.raises(UnsupportedCapabilityError):
        adapter.extract_attributes(Image.new("RGB", (20, 20)))


def test_florence_bad_postprocessing_preserves_text(monkeypatch):
    adapter = FlorenceAdapter(ModelConfig())
    monkeypatch.setattr(adapter, "prepare", lambda: None)
    monkeypatch.setattr(adapter, "request", lambda image, task: ({"text": "raw"}, {}))
    adapter._processor = SimpleNamespace(post_process_generation=lambda *a, **k: {"<OD>": {"labels": ["shirt"], "bboxes": []}})
    result = adapter.detect_garments(Image.new("RGB", (20, 20)))
    assert result.error is not None and result.raw_response["text"] == "raw"


def test_florence_requires_explicit_pinned_code_before_importing_torch():
    result = FlorenceAdapter(ModelConfig(model_id="microsoft/Florence-2-large")).detect_garments(Image.new("RGB", (2, 2)))
    assert result.error.code == "setup_error"
    assert "40-character" in result.error.message


def test_device_selection_and_fractional_crop():
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), backends=SimpleNamespace())
    assert select_device(torch, "auto") == "cpu"
    with pytest.raises(SetupError, match="unavailable"):
        select_device(torch, "cuda")
    region = BoundingBox(x_min=0.2, y_min=1.2, x_max=4.1, y_max=5.1)
    assert crop_image(Image.new("RGB", (10, 10)), region).size == (5, 5)


def test_real_sdk_accepts_request_types_when_installed():
    types = pytest.importorskip("google.genai.types")
    for task in ("detection", "attributes"):
        config = types.GenerateContentConfig(response_json_schema=response_schema(task), response_mime_type="application/json")
        assert config.response_json_schema


def test_real_gemini_sdk_with_mock_http_transport():
    genai = pytest.importorskip("google.genai")
    httpx = pytest.importorskip("httpx")
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            "candidates": [{"content": {"role": "model", "parts": [{"text": '{"garments":[]}'}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 4},
            "modelVersion": "fixture-model-version"})

    adapter = GeminiAdapter(ModelConfig(model_id="fixture-only"))
    adapter._client = genai.Client(api_key="test-only-no-real-key", http_options={
        "client_args": {"transport": httpx.MockTransport(respond)}})
    try:
        result = adapter.detect_garments(Image.new("RGB", (10, 10)))
    finally:
        adapter.close()
    assert result.error is None, result.error
    assert result.output.garments == []
    assert result.metadata.input_tokens == 12
    assert requests[0]["generationConfig"]["responseMimeType"] == "application/json"
    assert requests[0]["contents"][0]["parts"][1]["inlineData"]["mimeType"] == "image/png"


def test_florence_loader_and_generation_contract(monkeypatch):
    import sys
    from contextlib import nullcontext

    calls = {}

    class Tensor:
        def __init__(self, floating):
            self.floating = floating

        def is_floating_point(self):
            return self.floating

        def to(self, *args, **kwargs):
            calls.setdefault("tensor_moves", []).append((self.floating, args, kwargs))
            return self

    class Processor:
        def __call__(self, **kwargs):
            calls["processor"] = kwargs
            return {"input_ids": Tensor(False), "pixel_values": Tensor(True)}

        def batch_decode(self, ids, **kwargs):
            return ["fixture-native-output"]

        def post_process_generation(self, text, task, image_size):
            return {task: {"labels": ["shirt"], "bboxes": [[0, 0, 10, 10]]}}

    class Model:
        dtype = "float32"

        def to(self, device):
            calls["device"] = device
            return self

        def eval(self):
            calls["eval"] = True
            return self

        def generate(self, **kwargs):
            calls["generation"] = kwargs
            return [1]

    def load_processor(model_id, **kwargs):
        calls["processor_load"] = kwargs
        return Processor()

    def load_model(model_id, **kwargs):
        calls["model_load"] = kwargs
        return Model()

    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), backends=SimpleNamespace(),
        float16="float16", float32="float32", manual_seed=lambda value: calls.update(seed=value), inference_mode=nullcontext)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
        AutoProcessor=SimpleNamespace(from_pretrained=load_processor)))
    adapter = FlorenceAdapter(ModelConfig(model_id="microsoft/Florence-2-large", revision="a" * 40, trust_remote_code=True))
    result = adapter.detect_garments(Image.new("RGB", (20, 20)))
    assert result.error is None, result.error
    assert calls["model_load"]["local_files_only"] is True
    assert calls["model_load"]["code_revision"] == "a" * 40
    assert calls["model_load"]["torch_dtype"] == "float32"
    assert calls["generation"]["do_sample"] is False and calls["generation"]["num_beams"] == 3
    assert calls["tensor_moves"][0] == (False, ("cpu",), {})
    assert calls["tensor_moves"][1][2]["dtype"] == "float32"
    assert result.metadata.hardware == {"device": "cpu"}
