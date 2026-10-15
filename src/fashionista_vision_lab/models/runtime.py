"""Shared adapter mechanics, independent of benchmark policy."""

import math
from time import perf_counter

from PIL import Image

from ..schemas.common import BoundingBox
from ..schemas.outputs import InferenceError, InferenceMetadata, InferenceResult
from .base import ModelAdapter


class SetupError(ValueError):
    """An actionable configuration/dependency error safe to display verbatim."""


def crop_image(image: Image.Image, region: BoundingBox | None) -> Image.Image:
    if region is None:
        return image.convert("RGB")
    if not region.fits(*image.size):
        raise ValueError("region exceeds image dimensions")
    # Explicit outward rounding preserves all pixels touched by the annotation.
    return image.crop((math.floor(region.x_min), math.floor(region.y_min),
                       math.ceil(region.x_max), math.ceil(region.y_max))).convert("RGB")


class RuntimeAdapter(ModelAdapter):
    device = "not_loaded"

    def metadata(self, start: float, **usage) -> InferenceMetadata:
        return InferenceMetadata(model_name=self.name, model_id=self.config.model_id or "unconfigured",
                                 model_revision=self.config.revision, device=self.device,
                                 hardware=getattr(self, "hardware", {}),
                                 latency_seconds=perf_counter() - start, **usage)

    def failure(self, task, start, code, message, raw=None, **usage) -> InferenceResult:
        return InferenceResult(capability=task, metadata=self.metadata(start, **usage),
                               error=InferenceError(code=code, message=message), raw_response=raw)

    def invoke(self, image, task, region=None) -> InferenceResult:
        self.require_capability(task)
        start = perf_counter()
        raw, usage = None, {}
        try:
            prepared = crop_image(image, region)
            self.prepare()
            raw, usage = self.request(prepared, task)
            output = self.normalize(raw, task, prepared.size)
            return InferenceResult(capability=task, metadata=self.metadata(start, **usage),
                                   output=output, raw_response=raw)
        except SetupError as exc:
            return self.failure(task, start, "setup_error", str(exc), raw, **usage)
        except (ValueError, TypeError, KeyError, IndexError):
            return self.failure(task, start, "invalid_response_or_input",
                                "Input region or model output failed validation; inspect raw response.", raw, **usage)
        except Exception as exc:
            # Provider exceptions can contain request URLs/keys. Persist type only.
            return self.failure(task, start, "inference_error",
                                f"{type(exc).__name__}: inference failed; check access, model ID, quota and device resources.", raw, **usage)
