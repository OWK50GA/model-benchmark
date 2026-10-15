"""DINOv3 ViT: final CLS token, normalized for cosine retrieval."""

from .embedding import ImageEmbeddingAdapter


class DINOv3Adapter(ImageEmbeddingAdapter):
    name = "dinov3"
    pooling = "last_hidden_state_cls"
