import pytest

from fashionista_vision_lab.benchmarks.identity import Observation, neighbors, score_identity
from fashionista_vision_lab.schemas.outputs import EmbeddingOutput


def observation(sample, identity, vector, viewpoint="front", group=None, image=None, pixels=None):
    return Observation(sample_id=sample, image_id=image or sample, identity_id=identity,
        file_path=f"{sample}.jpg", viewpoint=viewpoint, appearance_group=group, image_digest=pixels,
        embedding=EmbeddingOutput(vector=vector, embedding_space="fixture") if vector else None)


def test_retrieval_cross_view_and_hard_negatives():
    rows = [observation("a-front", "a", [1, 0], group="shirts"),
            observation("a-back", "a", [0.9, 0.1], "back", "shirts"),
            observation("b-front", "b", [0.99, 0.01], group="shirts")]
    metrics, queries, pairs = score_identity(rows, 0.8)
    assert metrics["retrieval"]["all_views"]["eligible_queries"] == 2
    assert metrics["retrieval"]["all_views"]["recall_at_1"] == 0
    assert metrics["retrieval"]["all_views"]["recall_at_5"] == 1
    assert metrics["matching"]["hard_negatives"] == 2
    assert metrics["matching"]["hard_negative_false_positive_rate"] == 1
    assert metrics["matching"]["tp"] == 1
    assert len(pairs) == 3
    assert neighbors(rows, "a-front", 1)[0]["sample_id"] == "b-front"
    assert neighbors(rows, "a-front", 1, cross_view=True)[0]["sample_id"] == "a-back"


def test_failed_embeddings_do_not_improve_pair_false_positive_rate():
    rows = [observation("a1", "a", [1, 0]), observation("a2", "a", None, "back"),
            observation("b1", "b", None)]
    metrics, _, pairs = score_identity(rows, 0.5)
    assert metrics["retrieval"]["all_views"]["eligible_queries"] == 2
    assert metrics["retrieval"]["all_views"]["recall_at_5"] == 0
    assert metrics["matching"]["false_positive_rate"] is None
    assert metrics["matching"]["tn"] == 0
    assert metrics["matching"]["failed_different_pairs"] == 2
    assert metrics["matching"]["same_pair_recall_including_failures"] == 0
    assert all(p["outcome"] is None for p in pairs)


def test_self_same_image_and_identical_content_excluded():
    a = observation("a", "same", [1, 0], image="image1", pixels="identical")
    same_image = observation("b", "same", [1, 0], image="image1")
    duplicate = observation("c", "same", [1, 0], pixels="identical")
    assert neighbors([a, same_image, duplicate], "a") == []
    a.crop_digest, duplicate.crop_digest = "crop", "crop"
    duplicate.image_digest = "other"
    assert neighbors([a, duplicate], "a") == []


def test_no_positive_or_known_view_yields_na():
    rows = [observation("a", "a", [1, 0], "unknown"), observation("b", "b", [0, 1], "unknown")]
    metrics, _, _ = score_identity(rows, 0.5)
    assert metrics["retrieval"]["all_views"]["recall_at_1"] is None
    assert metrics["retrieval"]["cross_view"]["eligible_queries"] == 0
    assert metrics["matching"]["false_positive_rate"] == 0


def test_mixed_spaces_dimensions_duplicate_ids_rejected():
    a, b = observation("a", "a", [1, 0]), observation("b", "a", [1, 0, 0])
    with pytest.raises(ValueError, match="spaces or dimensions"):
        score_identity([a, b], 0.5)
    b.embedding = EmbeddingOutput(vector=[1, 0], embedding_space="another")
    with pytest.raises(ValueError, match="spaces or dimensions"):
        neighbors([a, b], "a")
    with pytest.raises(ValueError, match="duplicate"):
        score_identity([a, a], 0.5)


def test_ties_are_deterministic_and_threshold_is_inclusive():
    rows = [observation("q", "same", [1, 0]), observation("z", "same", [0, 1]),
            observation("a", "different", [0, -1])]
    assert [m["sample_id"] for m in neighbors(rows, "q")] == ["a", "z"]
    metrics, _, _ = score_identity(rows, 0)
    assert metrics["matching"]["tp"] == 1
    assert metrics["matching"]["fp"] == 1


def test_recall5_only_counts_positive_within_five():
    rows = [observation("q", "a", [1, 0]), observation("positive", "a", [-1, 0])]
    rows += [observation(f"negative{i}", f"b{i}", [1, i * 0.01]) for i in range(6)]
    metrics, queries, _ = score_identity(rows, 0.5)
    q = next(row for row in queries if row["sample_id"] == "q" and row["mode"] == "all_views")
    assert not q["hit_at_5"] and len(q["top5"]) == 5


@pytest.mark.parametrize("threshold", [-1.1, 1.1, float("nan")])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        score_identity([], threshold)


def test_cosine_is_stable_for_large_valid_vectors():
    rows = [observation("a", "a", [1e300, 1e300]), observation("b", "a", [1e300, 1e300])]
    assert neighbors(rows, "a")[0]["cosine_similarity"] == pytest.approx(1)
