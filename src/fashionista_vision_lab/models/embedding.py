"""Pinned local image encoders. No torch/Transformers imports until inference."""

import math
import re
from importlib.metadata import version
from pathlib import Path

from ..artifacts import json_digest
from ..schemas.outputs import EmbeddingOutput
from .florence import select_device
from .runtime import RuntimeAdapter, SetupError


class ImageEmbeddingAdapter(RuntimeAdapter):
    capabilities = frozenset({"embedding"})
    pooling = ""

    def __init__(self, config):
        super().__init__(config)
        self._model = self._processor = self._torch = None

    def prepare(self):
        if self._model is not None:
            return
        if not self.config.model_id or not re.fullmatch(r"[0-9a-fA-F]{40}", self.config.revision or ""):
            raise SetupError(f"Set models.{self.name}.model_id and revision to a full 40-character checkpoint commit hash.")
        if Path(self.config.model_id).is_dir():
            raise SetupError("Use a pinned Hugging Face repository ID; mutable local model directories are not supported by the embedding cache contract.")
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel, CLIPImageProcessor, CLIPModel
        except ImportError as exc:
            raise SetupError(f"Install .[{self.name}] in an embedding environment (Transformers >=4.56,<5; separate from Florence).") from exc
        self.device = select_device(torch, self.config.device)
        torch.manual_seed(self.seed)
        self.hardware = {"device": self.device}
        if self.device == "cuda":
            self.hardware["gpu_name"] = torch.cuda.get_device_name(0)
        options = dict(revision=self.config.revision, trust_remote_code=False,
                       local_files_only=not self.config.allow_downloads)
        if self.config.credential_env:
            try:
                options["token"] = self.config.require_credential()
            except ValueError as exc:
                raise SetupError(str(exc)) from exc
        try:
            processor_class, model_class = ((CLIPImageProcessor, CLIPModel) if self.name == "fashionclip"
                                          else (AutoImageProcessor, AutoModel))
            processor = processor_class.from_pretrained(self.config.model_id, use_fast=False, **options)
            model = model_class.from_pretrained(self.config.model_id, torch_dtype=torch.float32, **options)
            if self.name == "dinov3" and model.config.model_type != "dinov3_vit":
                raise SetupError("This DINOv3 adapter uses ViT CLS features; choose a dinov3_vit checkpoint, not ConvNeXt.")
            model = model.to(self.device).eval()
        except OSError as exc:
            raise SetupError(f"Cannot load {self.name}: check revision, local cache, downloads and checkpoint access/license approval.") from exc
        self._processor, self._model, self._torch = processor, model, torch
        self._space_spec = {"adapter": self.name, "version": "embedding-v1", "model_id": self.config.model_id,
            "revision": self.config.revision, "pooling": self.pooling, "dtype": "float32", "normalization": "l2",
            "processor": processor.to_dict(), "transformers": version("transformers"),
            "preprocessing": "EXIF-RGB; outward rounded crop; pinned slow processor"}

    def request(self, image, task):
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            if self.name == "fashionclip":
                features = self._model.get_image_features(**inputs)
            else:
                # Explicitly excludes register/patch tokens; never average all tokens.
                features = self._model(**inputs).last_hidden_state[:, 0, :]
        vector = features[0].detach().float().cpu().tolist()
        return {"vector": vector, "space_spec": self._space_spec}, {}

    def normalize(self, raw, task, size):
        checked = EmbeddingOutput(vector=raw["vector"], embedding_space="pending")
        norm = math.hypot(*checked.vector)
        spec = {**raw["space_spec"], "dimensions": len(checked.vector)}
        return EmbeddingOutput(vector=[v / norm for v in checked.vector], normalized=True,
                               embedding_space=json_digest(spec))

    def generate_embedding(self, image, region=None):
        return self.invoke(image, "embedding", region)

    def close(self):
        self._model = self._processor = None
        if self._torch is not None and self.device == "cuda":
            self._torch.cuda.empty_cache()
