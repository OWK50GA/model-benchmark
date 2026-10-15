"""Minimal capability-specific interfaces for independently usable adapters."""

from PIL import Image

from ..config import ModelConfig
from ..schemas.common import BoundingBox
from ..schemas.outputs import Capability, InferenceResult


class UnsupportedCapabilityError(ValueError):
    pass


class AdapterNotImplementedError(RuntimeError):
    pass


class ModelAdapter:
    """Adapters override only their supported methods and return validated results.

    Input images must already be EXIF-oriented. Region coordinates refer to that
    full image. Constructors should store config; load weights on first inference.
    """

    name = "base"
    capabilities: frozenset[Capability] = frozenset()

    def __init__(self, config: ModelConfig):
        self.config = config

    def require_capability(self, capability: Capability) -> None:
        if capability not in self.capabilities:
            raise UnsupportedCapabilityError(f"{self.name} does not support {capability}")

    def _unimplemented(self, capability: Capability) -> InferenceResult:
        self.require_capability(capability)
        raise AdapterNotImplementedError(f"{self.name}: {capability} has no implementation")

    def detect_garments(self, image: Image.Image) -> InferenceResult:
        return self._unimplemented("detection")

    def extract_attributes(self, image: Image.Image, region: BoundingBox | None = None) -> InferenceResult:
        return self._unimplemented("attributes")

    def generate_embedding(self, image: Image.Image, region: BoundingBox | None = None) -> InferenceResult:
        return self._unimplemented("embedding")
