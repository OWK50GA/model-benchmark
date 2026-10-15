"""Identity retrieval and threshold matching. Similarity is not physical identity."""

import math
from typing import Literal

from pydantic import Field

from ..schemas.common import Identifier, Schema
from ..schemas.outputs import EmbeddingOutput


class IdentityProtocol(Schema):
    version: Literal["identity-v1"] = "identity-v1"
    threshold: float = Field(ge=-1, le=1)
    split: Literal["train", "validation", "test", "unspecified"] = "test"
    allow_unverified: bool = False


class Observation(Schema):
    sample_id: Identifier
    image_id: Identifier
    identity_id: Identifier
    appearance_group: Identifier | None = None
    viewpoint: str = "unknown"
    file_path: Identifier
    image_digest: str | None = None
    crop_digest: str | None = None
    embedding: EmbeddingOutput | None = None


def distinct_images(a: Observation, b: Observation) -> bool:
    return (a.image_id != b.image_id
            and not (a.image_digest and a.image_digest == b.image_digest)
            and not (a.crop_digest and a.crop_digest == b.crop_digest))


def validate_spaces(observations):
    if len({o.sample_id for o in observations}) != len(observations):
        raise ValueError("duplicate observation IDs")
    spaces = {(o.embedding.embedding_space, len(o.embedding.vector)) for o in observations if o.embedding}
    if len(spaces) > 1:
        raise ValueError("Cannot compare different embedding spaces or dimensions")


def similarities(query: Observation, gallery: list[Observation]):
    from sklearn.metrics.pairwise import cosine_similarity

    if query.embedding is None or not gallery:
        return []
    # Stable pre-normalization avoids squared-norm overflow for valid large vectors.
    def unit(embedding):
        norm = math.hypot(*embedding.vector)
        return [value / norm for value in embedding.vector]

    scores = cosine_similarity([unit(query.embedding)], [unit(g.embedding) for g in gallery])[0]
    return sorted([(g, max(-1.0, min(1.0, float(score)))) for g, score in zip(gallery, scores)],
                  key=lambda item: (-item[1], item[0].sample_id))


def neighbors(observations, sample_id, k=5, cross_view=False):
    if k <= 0:
        raise ValueError("k must be positive")
    validate_spaces(observations)
    query = next((o for o in observations if o.sample_id == sample_id), None)
    if query is None:
        raise ValueError(f"unknown sample ID: {sample_id}")
    if query.embedding is None:
        raise ValueError("query embedding failed or is missing")
    gallery = [g for g in observations if g.embedding and distinct_images(query, g)
               and (not cross_view or (query.viewpoint != "unknown" and g.viewpoint != "unknown"
                                       and query.viewpoint != g.viewpoint))]
    return [{"sample_id": g.sample_id, "image_id": g.image_id, "file_path": g.file_path,
             "cosine_similarity": score, "identity_id": g.identity_id,
             "same_identity_label": g.identity_id == query.identity_id, "viewpoint": g.viewpoint}
            for g, score in similarities(query, gallery)[:k]]


def score_identity(observations: list[Observation], threshold: float):
    if not -1 <= threshold <= 1:
        raise ValueError("cosine threshold must be between -1 and 1")
    validate_spaces(observations)
    queries = []
    retrieval = {}
    for mode in ("all_views", "cross_view"):
        eligible, hits1, hits5, failed, no_positive = 0, 0, 0, 0, 0
        for q in observations:
            gallery = [g for g in observations if distinct_images(q, g)
                       and (mode == "all_views" or (q.viewpoint != "unknown" and g.viewpoint != "unknown"
                                                   and q.viewpoint != g.viewpoint))]
            positives = [g for g in gallery if g.identity_id == q.identity_id]
            if not positives:
                no_positive += 1
                queries.append({"sample_id": q.sample_id, "mode": mode, "eligible": False,
                                "reason": "no_distinct_positive_in_gallery"})
                continue
            eligible += 1
            failed += q.embedding is None
            ranked = similarities(q, [g for g in gallery if g.embedding])
            hit1 = any(g.identity_id == q.identity_id for g, _ in ranked[:1])
            hit5 = any(g.identity_id == q.identity_id for g, _ in ranked[:5])
            hits1 += hit1
            hits5 += hit5
            queries.append({"sample_id": q.sample_id, "mode": mode, "eligible": True,
                "query_failed": q.embedding is None, "gallery_size": len(gallery),
                "gallery_failed": sum(g.embedding is None for g in gallery),
                "positive_gallery_size": len(positives), "hit_at_1": hit1, "hit_at_5": hit5,
                "top5": [{"sample_id": g.sample_id, "cosine": score} for g, score in ranked[:5]]})
        retrieval[mode] = {"eligible_queries": eligible, "failed_queries": failed, "queries_without_positive": no_positive,
            "recall_at_1": hits1 / eligible if eligible else None,
            "recall_at_5": hits5 / eligible if eligible else None}

    pairs = []
    totals = dict(tp=0, fp=0, tn=0, fn=0, same_pairs=0, different_pairs=0,
                  failed_same_pairs=0, failed_different_pairs=0, hard_negatives=0,
                  scored_hard_negatives=0, hard_negative_false_positives=0)
    same_scores, different_scores = [], []
    for index, a in enumerate(observations):
        remaining = [b for b in observations[index + 1:] if distinct_images(a, b)]
        scored = {b.sample_id: score for b, score in similarities(a, [b for b in remaining if b.embedding])}
        for b in remaining:
            same = a.identity_id == b.identity_id
            hard = not same and bool(a.appearance_group) and a.appearance_group == b.appearance_group
            totals["same_pairs" if same else "different_pairs"] += 1
            totals["hard_negatives"] += hard
            score = scored.get(b.sample_id)
            outcome = None
            if score is None:
                totals["failed_same_pairs" if same else "failed_different_pairs"] += 1
            else:
                matched = score >= threshold
                outcome = ("tp" if matched else "fn") if same else ("fp" if matched else "tn")
                totals[outcome] += 1
                (same_scores if same else different_scores).append(score)
                totals["scored_hard_negatives"] += hard
                totals["hard_negative_false_positives"] += hard and matched
            pairs.append({"a": a.sample_id, "b": b.sample_id, "same_identity": same,
                          "hard_negative": hard, "cosine": score, "outcome": outcome})
    negative_scored = totals["fp"] + totals["tn"]
    matching = {**totals, "threshold": threshold,
        "false_positive_rate": totals["fp"] / negative_scored if negative_scored else None,
        "same_pair_recall_including_failures": totals["tp"] / totals["same_pairs"] if totals["same_pairs"] else None,
        "hard_negative_false_positive_rate": totals["hard_negative_false_positives"] / totals["scored_hard_negatives"] if totals["scored_hard_negatives"] else None,
        "mean_same_cosine": sum(same_scores) / len(same_scores) if same_scores else None,
        "mean_different_cosine": sum(different_scores) / len(different_scores) if different_scores else None}
    return {"retrieval": retrieval, "matching": matching}, queries, pairs
