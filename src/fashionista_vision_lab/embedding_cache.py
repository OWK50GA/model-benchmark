"""Content-addressed, checked JSON cache; never load pickle or cache failures."""

import json
import os
import tempfile
from pathlib import Path
from time import perf_counter

from .artifacts import digest, json_digest
from .schemas.outputs import EmbeddingOutput, InferenceResult


class EmbeddingCache:
    def __init__(self, root: Path, signature: dict):
        self.root, self.signature = root, signature

    def key(self, data: bytes, region):
        return json_digest({"format": "embedding-cache-v1", "image_sha256": digest(data),
                            "region": region.model_dump() if region else None, "signature": self.signature})

    def get(self, key: str):
        start = perf_counter()
        path = self.root / f"{key}.json"
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["key"] != key or record["checksum"] != json_digest(record["result"]):
                raise ValueError("cache checksum/key mismatch")
            result = InferenceResult.model_validate(record["result"])
            if result.capability != "embedding" or result.error or not isinstance(result.output, EmbeddingOutput):
                raise ValueError("cache entry is not a successful embedding")
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Invalid embedding cache entry {path}; remove it or use --no-cache.") from exc
        original = result.metadata.model_dump(mode="json")
        result.metadata.cache_hit = True
        result.metadata.latency_seconds = perf_counter() - start
        result.raw_response = {"cache_key": key, "original_metadata": original, "source_response": result.raw_response}
        return result

    def put(self, key: str, result: InferenceResult):
        if result.error or not isinstance(result.output, EmbeddingOutput):
            return
        self.root.mkdir(parents=True, exist_ok=True)
        value = result.model_dump(mode="json")
        entry = {"key": key, "checksum": json_digest(value), "result": value}
        fd, temporary = tempfile.mkstemp(prefix=".embedding-", dir=self.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(entry, stream, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                # Publish a complete file without replacing an existing/concurrent entry.
                os.link(temporary, self.root / f"{key}.json")
            except FileExistsError:
                pass
        finally:
            os.unlink(temporary)
