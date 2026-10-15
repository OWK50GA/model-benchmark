"""One-to-one category-aware detection matching; not a COCO AP implementation."""

from ..schemas.common import BoundingBox


def label(value, aliases=None):
    value = " ".join(value.casefold().split()) if value else None
    return (aliases or {}).get(value, value)


def iou(a: BoundingBox, b: BoundingBox) -> float:
    intersection = max(0, min(a.x_max, b.x_max) - max(a.x_min, b.x_min)) * max(
        0, min(a.y_max, b.y_max) - max(a.y_min, b.y_min))
    union = (a.x_max - a.x_min) * (a.y_max - a.y_min) + (b.x_max - b.x_min) * (b.y_max - b.y_min) - intersection
    return intersection / union


def score_detection(truth, predictions, mode="boxes", threshold=0.5, aliases=None):
    """Maximum-cardinality bipartite matching, deterministic IoU-first traversal.

    In category mode, match category counts only. Box mode additionally requires
    IoU >= threshold; missing predicted boxes cannot match. Duplicate predictions
    can match only once. Mean IoU is descriptive for the chosen matching, not an
    optimized-IoU assignment or average precision.
    """
    if mode not in ("boxes", "categories") or not 0 < threshold <= 1:
        raise ValueError("invalid detection scoring settings")
    edges = {}
    overlaps = {}
    for p, prediction in enumerate(predictions):
        candidates = []
        for g, garment in enumerate(truth):
            if garment.category is None or label(prediction.category, aliases) != label(garment.category, aliases):
                continue
            overlap = iou(garment.bbox, prediction.bbox) if garment.bbox and prediction.bbox else None
            if mode == "boxes" and (overlap is None or overlap < threshold):
                continue
            candidates.append((g, overlap or 0))
            overlaps[p, g] = overlap
        edges[p] = [g for g, _ in sorted(candidates, key=lambda v: (-v[1], v[0]))]
    owners = {}

    def augment(p, visited):
        for g in edges[p]:
            if g in visited:
                continue
            visited.add(g)
            if g not in owners or augment(owners[g], visited):
                owners[g] = p
                return True
        return False

    for p in edges:
        augment(p, set())
    matches = [{"annotation_id": truth[g].annotation_id, "prediction_index": p,
                "iou": overlaps[p, g] if mode == "boxes" else None} for g, p in sorted(owners.items())]
    return {"tp": len(matches), "fp": len(predictions) - len(matches), "fn": len(truth) - len(matches),
            "matches": matches,
            "missed": [g.annotation_id for i, g in enumerate(truth) if i not in owners],
            "incorrect_predictions": [p for p in range(len(predictions)) if p not in owners.values()]}


def aggregate_detection(scores):
    totals = {key: sum(row[key] for row in scores) for key in ("tp", "fp", "fn")}
    tp, fp, fn = (totals[k] for k in ("tp", "fp", "fn"))
    overlaps = [m["iou"] for row in scores for m in row["matches"] if m["iou"] is not None]
    return {**totals, "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "mean_matched_iou": sum(overlaps) / len(overlaps) if overlaps else None,
            "matched_iou_count": len(overlaps)}
