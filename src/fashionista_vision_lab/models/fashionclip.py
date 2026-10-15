"""FashionCLIP 2.0: projected CLIP image features, normalized for cosine retrieval."""

from .embedding import ImageEmbeddingAdapter


class FashionCLIPAdapter(ImageEmbeddingAdapter):
    name = "fashionclip"
    pooling = "clip_projected_image_features"
