import pytest

from fashionista_vision_lab.benchmarks.detection import iou, score_detection, aggregate_detection
from fashionista_vision_lab.benchmarks.attributes import score_attributes, aggregate_attributes
from fashionista_vision_lab.schemas.common import BoundingBox
from fashionista_vision_lab.schemas.dataset import GarmentAnnotation, GarmentAttributes
from fashionista_vision_lab.schemas.outputs import Detection, AttributeOutput


def box(x=0, right=10):
    return BoundingBox(x_min=x, y_min=0, x_max=right, y_max=10)


def garment(name="g", category="shirt", bbox=None, **attributes):
    return GarmentAnnotation(annotation_id=name, category=category, bbox=bbox,
                             attributes=GarmentAttributes(**attributes))


def test_iou_geometry():
    assert iou(box(), box()) == 1
    assert iou(box(), box(10, 20)) == 0
    assert iou(box(), box(5, 15)) == pytest.approx(1/3)


def test_duplicate_and_wrong_categories_are_false_positives():
    scores = score_detection([garment(bbox=box())], [Detection(category="shirt", bbox=box()),
        Detection(category="shirt", bbox=box()), Detection(category="dress", bbox=box())])
    assert (scores["tp"], scores["fp"], scores["fn"]) == (1, 2, 0)
    assert len(scores["incorrect_predictions"]) == 2
    assert aggregate_detection([scores])["precision"] == pytest.approx(1/3)


def test_maximum_matching_avoids_greedy_under_count():
    # Prediction 0 can match either truth; prediction 1 can only match truth 0.
    truth = [garment("a", bbox=box()), garment("b", bbox=box(5, 15))]
    predictions = [Detection(category="shirt", bbox=box(2, 12)), Detection(category="shirt", bbox=box())]
    assert score_detection(truth, predictions, threshold=0.5)["tp"] == 2


def test_missing_boxes_category_mode_and_aliases():
    truth = [garment(bbox=box())]
    prediction = [Detection(category=" T-SHIRT ")]
    assert score_detection(truth, prediction, aliases={"t-shirt": "shirt"})["tp"] == 0
    result = score_detection(truth, prediction, mode="categories", aliases={"t-shirt": "shirt"})
    assert result["tp"] == 1
    assert aggregate_detection([result])["mean_matched_iou"] is None


def test_empty_and_failed_prediction_denominators():
    empty = aggregate_detection([score_detection([], [])])
    assert empty["precision"] is None and empty["recall"] is None
    failed = score_detection([garment(bbox=box())], [])
    assert failed["missed"] == ["g"]
    assert aggregate_detection([failed])["recall"] == 0


def test_attributes_unknown_missing_incorrect_and_extra_values():
    truth = garment(colors=["blue"], patterns=["solid"], distinctive_details=[])
    pred = AttributeOutput(category="dress", attributes=GarmentAttributes(
        colors=["blue", "red"], patterns=None, distinctive_details=[], silhouette="straight"))
    scores = score_attributes(truth, pred)
    assert scores["category"]["incorrect"] == 1
    assert scores["colors"]["unsupported_values"] == 1
    assert scores["patterns"]["missing"] == 1
    assert scores["distinctive_details"]["correct"] == 1
    assert scores["silhouette"]["eligible"] == 0
    assert scores["silhouette"]["unscored_prediction"] == 1
    assert aggregate_attributes([scores])["per_attribute"]["silhouette"]["accuracy"] is None


def test_attribute_lists_are_sets_and_failure_is_not_excluded():
    truth = garment(colors=["blue", "white"])
    pred = AttributeOutput(category="SHIRT", attributes=GarmentAttributes(colors=["WHITE", "Blue"]))
    good, failed = score_attributes(truth, pred), score_attributes(truth, None)
    summary = aggregate_attributes([good, failed])
    assert summary["category_accuracy"] == 0.5
    assert summary["per_attribute"]["colors"]["accuracy"] == 0.5
    assert summary["per_attribute"]["colors"]["missing"] == 1
