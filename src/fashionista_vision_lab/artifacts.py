"""Exclusive run directories and separate raw/normalized JSONL artifacts."""

import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_digest(value) -> str:
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def environment():
    packages = {}
    for name in ("fashionista-vision-lab", "pydantic", "Pillow", "google-genai", "torch", "transformers", "timm", "einops"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    source_root = Path(__file__).parent
    sources = {str(path.relative_to(source_root)): digest(path.read_bytes())
               for path in sorted(source_root.rglob("*.py"))}
    return {"python": platform.python_version(), "platform": platform.platform(),
            "implementation_sha256": json_digest(sources),
            "machine": platform.machine(), "processor": platform.processor(), "packages": packages}


def write_json(path: Path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


class RunWriter:
    def __init__(self, root: Path, manifest: dict):
        root.mkdir(parents=True, exist_ok=True)
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:12]
        self.path = root / name
        self.path.mkdir()
        write_json(self.path / "manifest.json", {**manifest, "run_id": name,
                   "created_at": datetime.now(timezone.utc).isoformat(), "environment": environment()})
        # Presence of these files also describes a zero-record interrupted run.
        (self.path / "normalized.jsonl").touch(exist_ok=False)
        (self.path / "raw.jsonl").touch(exist_ok=False)

    def append(self, sample_id: str, image_id: str, result, score=None):
        data = result.model_dump(mode="json")
        raw = data.pop("raw_response")
        row = {"sample_id": sample_id, "image_id": image_id, "result": data, "score": score}
        for filename, value in (("normalized.jsonl", row), ("raw.jsonl", {"sample_id": sample_id, "response": raw})):
            with (self.path / filename).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")

    def finish(self, metrics):
        write_json(self.path / "metrics.json", metrics)
        from .reporting import render_report
        (self.path / "report.md").write_text(render_report([self.path]), encoding="utf-8")
