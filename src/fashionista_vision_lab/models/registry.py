"""Descriptors can be inspected without importing optional SDKs or torch."""

from dataclasses import asdict, dataclass
from importlib import import_module

from ..config import ModelConfig
from ..schemas.outputs import Capability
from .base import AdapterNotImplementedError, ModelAdapter


@dataclass(frozen=True)
class ModelDescriptor:
    name: str
    planned_capabilities: tuple[Capability, ...]
    phase: int
    notes: str
    factory_path: str | None = None

    def as_dict(self) -> dict:
        return {**asdict(self), "implemented": self.factory_path is not None}


REGISTRY = {
    "gemini": ModelDescriptor("gemini", ("detection", "attributes"), 2,
                              "Configured API model ID required; capabilities depend on the selected model.",
                              "fashionista_vision_lab.models.gemini:GeminiAdapter"),
    "florence": ModelDescriptor("florence", ("detection", "caption"), 2,
                                "Native captioning/localization; structured garment attributes are not assumed.",
                                "fashionista_vision_lab.models.florence:FlorenceAdapter"),
    "fashionclip": ModelDescriptor("fashionclip", ("embedding",), 3,
                                   "Fashion similarity embeddings; physical identity must be evaluated.",
                                   "fashionista_vision_lab.models.fashionclip:FashionCLIPAdapter"),
    "dinov3": ModelDescriptor("dinov3", ("embedding",), 3,
                              "Visual embeddings; checkpoint access and license acceptance may be required.",
                              "fashionista_vision_lab.models.dinov3:DINOv3Adapter"),
}


def get_descriptor(name: str) -> ModelDescriptor:
    try:
        return REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"unknown model {name!r}; available: {', '.join(sorted(REGISTRY))}") from exc


def create_adapter(name: str, config: ModelConfig) -> ModelAdapter:
    descriptor = get_descriptor(name)
    if descriptor.factory_path is None:
        raise AdapterNotImplementedError(f"{name} adapter is planned for Phase {descriptor.phase}")
    module_name, class_name = descriptor.factory_path.split(":", 1)
    adapter = getattr(import_module(module_name), class_name)(config)
    if not isinstance(adapter, ModelAdapter):
        raise TypeError("registered factory must return a ModelAdapter")
    return adapter
