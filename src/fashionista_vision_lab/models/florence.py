"""Microsoft's original Florence-2 checkpoint loader and native task outputs."""

import re

from ..schemas.common import BoundingBox
from ..schemas.outputs import CaptionOutput, Detection, DetectionOutput
from .runtime import RuntimeAdapter, SetupError

TASKS = {"detection": "<OD>", "caption": "<MORE_DETAILED_CAPTION>"}


def select_device(torch, requested):
    available = {"cpu": True, "cuda": torch.cuda.is_available(),
                 "mps": hasattr(torch.backends, "mps") and torch.backends.mps.is_available()}
    if requested == "auto":
        return "cuda" if available["cuda"] else "cpu"
    if not available[requested]:
        raise SetupError(f"Requested device {requested} is unavailable; use cpu or auto.")
    return requested


class FlorenceAdapter(RuntimeAdapter):
    name = "florence"
    capabilities = frozenset({"detection", "caption"})

    def __init__(self, config):
        super().__init__(config)
        self._model = self._processor = self._torch = None

    def prepare(self):
        if self._model is not None:
            return
        if not self.config.model_id:
            raise SetupError("Set models.florence.model_id (Microsoft checkpoint: microsoft/Florence-2-large).")
        if not self.config.trust_remote_code or not re.fullmatch(r"[0-9a-fA-F]{40}", self.config.revision or ""):
            raise SetupError("Original Florence weights need reviewed custom code: set trust_remote_code=true and revision to a 40-character commit hash.")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:
            raise SetupError("Install the Florence extra with a suitable PyTorch build: pip install -e '.[florence]'") from exc
        self.device = select_device(torch, self.config.device)
        torch.manual_seed(self.seed)
        self.hardware = {"device": self.device}
        if self.device == "cuda":
            self.hardware["gpu_name"] = torch.cuda.get_device_name(0)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        options = dict(revision=self.config.revision, code_revision=self.config.revision,
                       trust_remote_code=True, local_files_only=not self.config.allow_downloads)
        if self.config.credential_env:
            try:
                options["token"] = self.config.require_credential()
            except ValueError as exc:
                raise SetupError(str(exc)) from exc
        try:
            processor = AutoProcessor.from_pretrained(self.config.model_id, **options)
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id, torch_dtype=dtype, attn_implementation="eager", **options)
            model = model.to(self.device).eval()
        except OSError as exc:
            raise SetupError("Florence weights/code could not be loaded. Check the revision, cache and access; allow_downloads defaults to false.") from exc
        self._model, self._processor, self._torch = model, processor, torch

    def request(self, image, task):
        prompt = TASKS[task]
        inputs = self._processor(text=prompt, images=image, return_tensors="pt")
        inputs = {key: value.to(device=self.device, dtype=self._model.dtype)
                  if value.is_floating_point() else value.to(self.device)
                  for key, value in inputs.items()}
        with self._torch.inference_mode():
            ids = self._model.generate(**inputs, max_new_tokens=self.config.max_output_tokens,
                                       num_beams=3, do_sample=False)
        text = self._processor.batch_decode(ids, skip_special_tokens=False)[0]
        # Preserve text before parsing, so malformed post-processing remains debuggable.
        return {"task_prompt": prompt, "text": text}, {}

    def normalize(self, raw, task, size):
        parsed = self._processor.post_process_generation(raw["text"], task=TASKS[task], image_size=size)
        raw["parsed"] = parsed
        value = parsed[TASKS[task]]
        if task == "caption":
            return CaptionOutput(text=value)
        labels, boxes = value["labels"], value["bboxes"]
        if len(labels) != len(boxes):
            raise ValueError("Florence label and box counts differ")
        garments = []
        for label, coords in zip(labels, boxes):
            if len(coords) != 4:
                raise ValueError("invalid Florence box")
            box = BoundingBox(**dict(zip(("x_min", "y_min", "x_max", "y_max"), coords)))
            if not box.fits(*size):
                raise ValueError("Florence box outside image")
            # Keep all native OD labels. Dataset scoring exposes non-garment false positives.
            garments.append(Detection(category=label, bbox=box))
        return DetectionOutput(garments=garments)

    def detect_garments(self, image):
        return self.invoke(image, "detection")

    def caption(self, image, region=None):
        return self.invoke(image, "caption", region)

    def close(self):
        self._model = self._processor = None
        if self._torch is not None and self.device == "cuda":
            self._torch.cuda.empty_cache()
