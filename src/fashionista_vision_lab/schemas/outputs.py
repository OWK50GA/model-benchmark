"""Normalized inference outputs, failures and metadata; scoring lives separately."""

from typing import Literal

from pydantic import Field, JsonValue, model_validator

from .common import BoundingBox, Identifier, NonNegativeFloat, Probability, Schema
from .dataset import GarmentAttributes

Capability = Literal["detection", "attributes", "embedding", "caption"]


class CaptionOutput(Schema):
    text: Identifier


class Detection(Schema):
    category: Identifier | None = None
    bbox: BoundingBox | None = None
    confidence: Probability | None = None


class DetectionOutput(Schema):
    garments: list[Detection]


class AttributeOutput(Schema):
    category: Identifier | None = None
    attributes: GarmentAttributes = Field(default_factory=GarmentAttributes)
    # Field paths (e.g. attributes.colors) with explanations for uncertain outputs.
    uncertainty: dict[str, Identifier] = Field(default_factory=dict)


class EmbeddingOutput(Schema):
    vector: list[float] = Field(min_length=1)
    embedding_space: Identifier
    normalized: bool = False

    @model_validator(mode="after")
    def meaningful_vector(self) -> "EmbeddingOutput":
        import math

        norm = math.hypot(*self.vector)
        if not math.isfinite(norm) or norm == 0:
            raise ValueError("embedding must have a finite, non-zero norm")
        if self.normalized and not math.isclose(norm, 1.0, abs_tol=1e-5):
            raise ValueError("normalized embedding must have unit norm")
        return self


class InferenceMetadata(Schema):
    model_name: Identifier
    model_id: Identifier
    model_revision: Identifier | None = None
    device: Identifier
    hardware: dict[str, str] = Field(default_factory=dict)
    latency_seconds: NonNegativeFloat
    cache_hit: bool = False
    input_tokens: int | None = Field(default=None, ge=0, strict=True)
    output_tokens: int | None = Field(default=None, ge=0, strict=True)
    thinking_tokens: int | None = Field(default=None, ge=0, strict=True)
    resolved_model_version: Identifier | None = None
    estimated_cost: NonNegativeFloat | None = None
    cost_currency: Identifier | None = None
    cost_basis: Identifier | None = None

    @model_validator(mode="after")
    def cost_is_explained(self) -> "InferenceMetadata":
        if self.estimated_cost is not None and (self.cost_currency is None or self.cost_basis is None):
            raise ValueError("estimated costs require currency and pricing basis")
        return self


class InferenceError(Schema):
    code: Identifier
    message: Identifier


class InferenceResult(Schema):
    schema_version: Literal["1.0"] = "1.0"
    capability: Capability
    metadata: InferenceMetadata
    output: DetectionOutput | AttributeOutput | EmbeddingOutput | CaptionOutput | None = None
    error: InferenceError | None = None
    raw_response: JsonValue = None

    @model_validator(mode="after")
    def valid_outcome(self) -> "InferenceResult":
        if (self.output is None) == (self.error is None):
            raise ValueError("exactly one of output or error is required")
        expected = {"detection": DetectionOutput, "attributes": AttributeOutput, "embedding": EmbeddingOutput, "caption": CaptionOutput}
        if self.output is not None and not isinstance(self.output, expected[self.capability]):
            raise ValueError("output type must match capability")
        return self
