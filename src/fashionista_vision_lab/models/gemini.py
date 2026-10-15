"""Google Gen AI SDK adapter. Importing this module does not import the SDK."""

import json

from pydantic import Field

from ..schemas.common import Identifier, Schema
from ..schemas.outputs import AttributeOutput, Detection, DetectionOutput
from ..schemas.common import BoundingBox
from .runtime import RuntimeAdapter, SetupError

PROMPT_VERSION = "gemini-garments-v1"
DETECTION_PROMPT = """Identify every visible garment, including separate items worn together.
Exclude people, skin, furniture and background objects. Return JSON with garments.
For each garment return category (a short singular English clothing name) and
box_2d=[ymin,xmin,ymax,xmax], integer coordinates normalized to 0..1000.
Return an empty garments list when no garments are visible. Do not invent hidden items."""
ATTRIBUTE_PROMPT = """Describe the single target garment in this image. Return JSON only.
Use short lowercase English labels. category is a singular garment type.
attributes contains subcategory, colors, patterns, visible_material_characteristics,
silhouette and distinctive_details. Describe visible evidence only, never guess
fiber composition, brand, physical identity or hidden details. Use null for unknown
fields and [] only for an observed absence. uncertainty maps uncertain field paths
to short explanations. If several garments make the target ambiguous, leave
uncertain fields null and explain the ambiguity."""


class BoxPrediction(Schema):
    category: Identifier
    box_2d: list[int] = Field(min_length=4, max_length=4)


class GeminiDetections(Schema):
    garments: list[BoxPrediction]


def response_schema(task: str) -> dict:
    """Small provider schema avoids sending Pydantic-only validation constraints."""
    if task == "detection":
        return {"type": "object", "required": ["garments"], "properties": {
            "garments": {"type": "array", "items": {"type": "object",
                "required": ["category", "box_2d"], "properties": {
                    "category": {"type": "string"},
                    "box_2d": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}}}}}}
    fields = {}
    for field in ("subcategory", "colors", "patterns", "visible_material_characteristics", "silhouette", "distinctive_details"):
        fields[field] = ({"type": ["string", "null"]} if field in ("subcategory", "silhouette") else
                         {"type": ["array", "null"], "items": {"type": "string"}})
    return {"type": "object", "required": ["category", "attributes"], "properties": {
        "category": {"type": ["string", "null"]},
        "attributes": {"type": "object", "properties": fields, "required": list(fields)},
        "uncertainty": {"type": "object", "additionalProperties": {"type": "string"}}}}


class GeminiAdapter(RuntimeAdapter):
    name = "gemini"
    capabilities = frozenset({"detection", "attributes"})
    device = "hosted"

    def __init__(self, config):
        super().__init__(config)
        self._client = None

    def prepare(self):
        if not self.config.model_id:
            raise SetupError("Set models.gemini.model_id to an available Gemini API model ID.")
        if self._client is None:
            try:
                key = self.config.require_credential()
            except ValueError as exc:
                raise SetupError(str(exc)) from exc
            try:
                from google import genai
            except ImportError as exc:
                raise SetupError("Install the Gemini extra: pip install -e '.[gemini]'") from exc
            self._client = genai.Client(api_key=key, http_options={"timeout": int(self.config.timeout_seconds * 1000)})

    def request(self, image, task):
        response = self._client.models.generate_content(
            model=self.config.model_id,
            contents=[DETECTION_PROMPT if task == "detection" else ATTRIBUTE_PROMPT, image],
            config={"temperature": self.config.temperature, "seed": self.seed, "max_output_tokens": self.config.max_output_tokens,
                    "response_mime_type": "application/json", "response_json_schema": response_schema(task)},
        )
        body = response.model_dump(mode="json", exclude_none=True)
        raw = {"prompt_version": PROMPT_VERSION,
               "prompt": DETECTION_PROMPT if task == "detection" else ATTRIBUTE_PROMPT,
               "response_schema": response_schema(task), "body": body, "text": response.text}
        metadata = body.get("usage_metadata", {})
        usage = {"input_tokens": metadata.get("prompt_token_count"),
                 "output_tokens": metadata.get("candidates_token_count"),
                 "thinking_tokens": metadata.get("thoughts_token_count"),
                 "resolved_model_version": body.get("model_version")}
        if (self.config.input_cost_per_million is not None and usage["input_tokens"] is not None
                and usage["output_tokens"] is not None):
            # Configured rates must include the provider's thinking-token pricing policy.
            usage.update(estimated_cost=(usage["input_tokens"] * self.config.input_cost_per_million
                         + (usage["output_tokens"] + (usage["thinking_tokens"] or 0))
                         * self.config.output_cost_per_million) / 1_000_000,
                         cost_currency=self.config.cost_currency, cost_basis=self.config.cost_basis)
        return raw, usage

    def normalize(self, raw, task, size):
        data = json.loads(raw["text"])
        if task == "attributes":
            return AttributeOutput.model_validate(data)
        parsed = GeminiDetections.model_validate(data)
        width, height = size
        garments = []
        for item in parsed.garments:
            if any(v < 0 or v > 1000 for v in item.box_2d):
                raise ValueError("normalized coordinates outside 0..1000")
            y1, x1, y2, x2 = item.box_2d
            garments.append(Detection(category=item.category, bbox=BoundingBox(
                x_min=x1 * width / 1000, y_min=y1 * height / 1000,
                x_max=x2 * width / 1000, y_max=y2 * height / 1000)))
        return DetectionOutput(garments=garments)

    def detect_garments(self, image):
        return self.invoke(image, "detection")

    def extract_attributes(self, image, region=None):
        return self.invoke(image, "attributes", region)

    def close(self):
        if self._client is not None:
            self._client.close()
            self._client = None
