import pandas as pd
import pytest

from src.t14_tools import (
    build_class_map,
    load_inputs,
    map_classes,
    normalise_boxes,
    validate_predictions,
)

FIXTURE_DIR = "fixtures/t14_auto_annotation"


def test_wrong_alias_mapping_must_fail_loudly():
    """RED TEST: Map sai COCO 'car' (id=2) sang 'pedestrian' không được phép lọt qua."""
    data = load_inputs(FIXTURE_DIR)
    bad_alias = {"car": "pedestrian"}

    with pytest.raises(ValueError, match="duplicate|missing|contradicts|same taxonomy"):
        cmap = build_class_map(data["label_map"], data["taxonomy"], alias=bad_alias)
        mapped = map_classes(data["preds"], cmap)
        normed, _ = normalise_boxes(mapped, data["images"])
        invalid = validate_predictions(normed, data["taxonomy"], data["images"])
        assert invalid.empty


def test_unknown_prediction_class_id_raises_error():
    """Class ID lạ không có trong model label_map phải bị chặn ngay."""
    data = load_inputs(FIXTURE_DIR)
    cmap = build_class_map(data["label_map"], data["taxonomy"])

    bad_preds = pd.DataFrame(
        [{"image_id": data["images"].iloc[0]["image_id"], "class_id": 99, "x": 10, "y": 10, "w": 20, "h": 20, "conf": 0.9}]
    )
    with pytest.raises(ValueError, match="not in the model label_map"):
        map_classes(bad_preds, cmap)


def test_unmapped_target_not_in_taxonomy_raises_error():
    """Alias trỏ tới một tên lớp không tồn tại trong taxonomy phải báo lỗi."""
    data = load_inputs(FIXTURE_DIR)
    with pytest.raises(ValueError, match="not in the taxonomy"):
        build_class_map(data["label_map"], data["taxonomy"], alias={"car": "truck"})


def test_valid_fixture_mapping_passes_validation():
    """Happy path: Map chuẩn cả 4 lớp của fixture (0->pedestrian, 2->car, 3->motorbike, 11->sign) phải pass."""
    data = load_inputs(FIXTURE_DIR)
    cmap = build_class_map(data["label_map"], data["taxonomy"])

    assert cmap == {0: "pedestrian", 2: "car", 3: "motorbike", 11: "sign"}

    mapped = map_classes(data["preds"], cmap)
    assert set(mapped["class"].unique()) == {"pedestrian", "car", "motorbike", "sign"}

    normed, _ = normalise_boxes(mapped, data["images"])
    invalid = validate_predictions(normed, data["taxonomy"], data["images"])
    assert invalid.empty