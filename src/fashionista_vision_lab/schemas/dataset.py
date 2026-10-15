"""Annotations distinguish physical identity, visual resemblance, and provenance."""

from pathlib import PurePosixPath, PureWindowsPath
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .common import BoundingBox, Identifier, Schema

Viewpoint = Literal["front", "back", "side", "detail", "unknown"]
ImageContext = Literal["product", "selfie", "flat_lay", "full_outfit", "other", "unknown"]


class GarmentAttributes(Schema):
    """None means unannotated/unknown; [] means a reviewed absence of values.

    Material observations describe visible appearance, not inferred composition.
    Taxonomy values are dataset-defined, enabling new labels without code edits.
    """

    subcategory: Identifier | None = None
    colors: list[Identifier] | None = None
    patterns: list[Identifier] | None = None
    visible_material_characteristics: list[Identifier] | None = None
    silhouette: Identifier | None = None
    distinctive_details: list[Identifier] | None = None

    @field_validator("colors", "patterns", "visible_material_characteristics", "distinctive_details")
    @classmethod
    def no_duplicate_labels(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and len(value) != len(set(value)):
            raise ValueError("attribute labels must be unique")
        return value


class GarmentAnnotation(Schema):
    annotation_id: Identifier
    identity_id: Identifier | None = None
    appearance_group: Identifier | None = None
    category: Identifier | None = None
    attributes: GarmentAttributes = Field(default_factory=GarmentAttributes)
    bbox: BoundingBox | None = None
    viewpoint: Viewpoint = "unknown"
    verification: Literal["example", "unverified", "verified"] = "unverified"
    verified_by: Identifier | None = None
    notes: Identifier | None = None

    @model_validator(mode="after")
    def require_reviewer(self) -> "GarmentAnnotation":
        if self.verification == "verified" and self.verified_by is None:
            raise ValueError("verified annotations require verified_by")
        if self.verification != "verified" and self.verified_by is not None:
            raise ValueError("verified_by is only valid for verified annotations")
        return self


class ImageAnnotation(Schema):
    image_id: Identifier
    file_path: Identifier
    width: int = Field(gt=0, strict=True)
    height: int = Field(gt=0, strict=True)
    context: ImageContext = "unknown"
    split: Literal["train", "validation", "test", "unspecified"] = "unspecified"
    # False distinguishes partial labels from a reviewed image with no garments.
    garments_complete: bool = False
    garments: list[GarmentAnnotation] = Field(default_factory=list)

    @field_validator("file_path")
    @classmethod
    def portable_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (path.is_absolute() or PureWindowsPath(value).drive or "\\" in value
                or ".." in path.parts or ":" in value or "\x00" in value
                or value == "." or value.endswith("/")):
            raise ValueError("file_path must be a relative POSIX file path without traversal")
        return value

    @model_validator(mode="after")
    def boxes_within_image(self) -> "ImageAnnotation":
        for garment in self.garments:
            if garment.bbox is not None and not garment.bbox.fits(self.width, self.height):
                raise ValueError(f"box for {garment.annotation_id} exceeds image dimensions")
        return self


class Dataset(Schema):
    schema_version: Literal["1.0"] = "1.0"
    dataset_id: Identifier
    description: Identifier
    synthetic: bool = False
    images: list[ImageAnnotation] = Field(min_length=1)

    @model_validator(mode="after")
    def consistent_records(self) -> "Dataset":
        for label, values in (
            ("image_id", [image.image_id for image in self.images]),
            ("file_path", [image.file_path for image in self.images]),
            ("annotation_id", [g.annotation_id for image in self.images for g in image.garments]),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {label}")
        for image in self.images:
            for garment in image.garments:
                if self.synthetic and garment.verification != "example":
                    raise ValueError("synthetic datasets must mark every annotation as example")
                if not self.synthetic and garment.verification == "example":
                    raise ValueError("example annotations require synthetic=true")
        return self
