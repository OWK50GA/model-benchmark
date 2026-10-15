"""JSON configuration contains credential variable names, never credential values."""

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

from pydantic import Field

from .schemas.common import Identifier, Schema


class ModelConfig(Schema):
    model_id: Identifier | None = None
    revision: Identifier | None = None
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    credential_env: str | None = Field(default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")

    def require_credential(self) -> str:
        if self.credential_env is None:
            raise ValueError("no credential environment variable configured")
        value = os.environ.get(self.credential_env)
        if not value or not value.strip():
            raise ValueError(f"missing credential: set {self.credential_env}")
        return value


class LabConfig(Schema):
    schema_version: Literal["1.0"] = "1.0"
    seed: int = Field(default=42, ge=0, le=2**32 - 1, strict=True)
    dataset_path: Identifier = "../datasets/annotations/example.json"
    image_root: Identifier = "../datasets/images"
    results_dir: Identifier = "../results"
    models: dict[Identifier, ModelConfig] = Field(default_factory=dict)

    def fingerprint(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(canonical.encode()).hexdigest()


def load_config(path: Path) -> LabConfig:
    return LabConfig.model_validate_json(path.read_text(encoding="utf-8"))


def resolve_config_path(config_path: Path, value: str) -> Path:
    """All relative config paths are relative to the configuration file."""
    return (config_path.resolve().parent / value).resolve()
