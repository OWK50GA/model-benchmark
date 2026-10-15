"""Shared validation primitives. Unknown fields are errors, not silent typos."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
NonNegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class BoundingBox(Schema):
    """Continuous pixel coordinates: [x_min, y_min, x_max, y_max].

    Origin is the top-left of the EXIF-oriented image. Maxima are exclusive;
    a full image box is (0, 0, width, height). No implicit rounding or clipping.
    """

    x_min: NonNegativeFloat
    y_min: NonNegativeFloat
    x_max: NonNegativeFloat
    y_max: NonNegativeFloat

    @model_validator(mode="after")
    def positive_area(self) -> "BoundingBox":
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("bounding box must have positive width and height")
        return self

    def fits(self, width: int, height: int) -> bool:
        return self.x_max <= width and self.y_max <= height
