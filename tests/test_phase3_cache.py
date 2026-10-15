import json

import pytest

from fashionista_vision_lab.embedding_cache import EmbeddingCache
from fashionista_vision_lab.schemas.common import BoundingBox
from fashionista_vision_lab.schemas.outputs import EmbeddingOutput, InferenceError, InferenceMetadata, InferenceResult


def result():
    return InferenceResult(capability="embedding", metadata=InferenceMetadata(
        model_name="fixture", model_id="fixture", device="cpu", latency_seconds=10),
        output=EmbeddingOutput(vector=[1, 0], embedding_space="fixture"), raw_response={"fixture": True})


def test_cache_hit_preserves_origin_and_reports_lookup_latency(tmp_path):
    cache = EmbeddingCache(tmp_path, {"revision": "1"})
    key = cache.key(b"image", None)
    assert cache.get(key) is None
    cache.put(key, result())
    hit = cache.get(key)
    assert hit.output == result().output
    assert hit.metadata.cache_hit and hit.metadata.latency_seconds < 10
    assert hit.raw_response["original_metadata"]["latency_seconds"] == 10
    assert list(tmp_path.glob(".embedding-*")) == []


def test_key_changes_with_image_region_model_and_preprocessing(tmp_path):
    cache = EmbeddingCache(tmp_path, {"revision": "1", "preprocessing": "a"})
    region = BoundingBox(x_min=0, y_min=0, x_max=1, y_max=1)
    keys = {cache.key(b"image", None), cache.key(b"new", None), cache.key(b"image", region),
        EmbeddingCache(tmp_path, {"revision": "2", "preprocessing": "a"}).key(b"image", None),
        EmbeddingCache(tmp_path, {"revision": "1", "preprocessing": "b"}).key(b"image", None)}
    assert len(keys) == 5


def test_corrupt_entry_fails_and_existing_entry_not_overwritten(tmp_path):
    cache = EmbeddingCache(tmp_path, {})
    key = cache.key(b"image", None)
    cache.put(key, result())
    path = tmp_path / f"{key}.json"
    data = json.loads(path.read_text())
    data["result"]["output"]["vector"] = [0, 1]
    path.write_text(json.dumps(data))
    before = path.read_bytes()
    cache.put(key, result())
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="Invalid embedding cache"):
        cache.get(key)


def test_failures_are_not_cached(tmp_path):
    failure = InferenceResult(capability="embedding", metadata=result().metadata,
                              error=InferenceError(code="test", message="fixture"))
    cache = EmbeddingCache(tmp_path, {})
    cache.put(cache.key(b"image", None), failure)
    assert list(tmp_path.iterdir()) == []
