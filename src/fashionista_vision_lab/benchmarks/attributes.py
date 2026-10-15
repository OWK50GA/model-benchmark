"""Attribute accuracy on known annotations, independent of detection quality."""

from .detection import label


def values(garment):
    return {"category": garment.category, **garment.attributes.model_dump()}


def canonical(value, aliases=None):
    if isinstance(value, list):
        return sorted({label(v) for v in value})
    return label(value, aliases)


def score_attributes(truth, prediction, aliases=None):
    expected = values(truth)
    actual = values(prediction) if prediction else {key: None for key in expected}
    scores = {}
    for field, target in expected.items():
        result = actual[field]
        if target is None:
            scores[field] = {"eligible": 0, "correct": 0, "missing": 0, "incorrect": 0,
                             "unsupported_values": 0, "unscored_prediction": int(result is not None)}
            continue
        target = canonical(target, aliases if field == "category" else None)
        result = canonical(result, aliases if field == "category" else None)
        missing = result is None or (isinstance(target, list) and bool(target) and result == [])
        correct = result == target
        extra = len(set(result or []) - set(target)) if isinstance(target, list) else int(not missing and not correct)
        scores[field] = {"eligible": 1, "correct": int(correct), "missing": int(missing),
                         "incorrect": int(not correct and not missing), "unsupported_values": extra,
                         "unscored_prediction": 0}
    return scores


def aggregate_attributes(scores):
    fields = {}
    for row in scores:
        for field, counts in row.items():
            target = fields.setdefault(field, {key: 0 for key in counts})
            for key, count in counts.items():
                target[key] += count
    for counts in fields.values():
        counts["accuracy"] = counts["correct"] / counts["eligible"] if counts["eligible"] else None
    return {"per_attribute": fields, "category_accuracy": fields.get("category", {}).get("accuracy")}
