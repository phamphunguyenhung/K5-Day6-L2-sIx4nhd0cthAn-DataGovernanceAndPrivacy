"""T14 tests: adapter, geometry, review loop and the end-to-end pipeline. None of them needs gt/ (not distributed)."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.common.context import RunContext
from src.pipeline import decide
from src.t14_tools import (add_suspicion, build_class_map, export_coco, export_yolo, load_inputs, map_classes,
                           normalise_boxes, parse_override, roundtrip_check, validate_predictions)

ROOT = Path(__file__).resolve().parents[1]
FX = ROOT / "fixtures" / "t14_auto_annotation"


def run_main(out: Path, pipeline: str | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "src.main", "--config", "configs/lab.yaml", "--track", "t14", "--input", str(FX),
           "--output", str(out), "--run-name", "poc", "--offline", "--seed", "42"]
    if pipeline:
        cmd += ["--pipeline", pipeline]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ------------------------------------------------------------------ adapter (success + failure)
def test_class_map_converts_coco_ids_to_taxonomy_names():
    d = load_inputs(FX)
    cm = build_class_map(d["label_map"], d["taxonomy"])
    # model ids are COCO-style and DIFFER from taxonomy ids (taxonomy: car=0, model: person=0)
    assert cm == {0: "pedestrian", 2: "car", 3: "motorbike", 11: "sign"}
    mapped = map_classes(d["preds"], cm)
    assert set(mapped["class"]) <= set(d["taxonomy"].values())


def test_unmapped_model_class_fails_loudly():
    d = load_inputs(FX)
    with pytest.raises(ValueError, match="no alias"):
        build_class_map({**d["label_map"], 7: "truck"}, d["taxonomy"])          # a van/truck has no taxonomy class
    cm = build_class_map(d["label_map"], d["taxonomy"])
    with pytest.raises(ValueError, match="not in the model label_map"):
        map_classes(pd.DataFrame({"class_id": [0, 99]}), cm)


def test_wrong_alias_would_still_be_a_valid_file_but_is_caught_by_validator_rules():
    """Failure mode #1: a wrong map exports 'valid' rows. The adapter must refuse an alias outside the taxonomy."""
    d = load_inputs(FX)
    with pytest.raises(ValueError, match="not in the taxonomy"):
        build_class_map(d["label_map"], d["taxonomy"], {"person": "bicycle"})


@pytest.mark.parametrize("alias,why", [
    ({"car": "pedestrian"}, "contradicts the taxonomy"),      # COCO 2=car relabelled as pedestrian
    ({"person": "car"}, "map to the same taxonomy class"),    # person and car both end up as car
])
def test_wrong_alias_inside_the_taxonomy_is_refused(alias, why):
    """Failure mode #1, the silent variant: the wrong target IS a taxonomy class, so every exported row is 'valid'."""
    d = load_inputs(FX)
    with pytest.raises(ValueError, match=why):
        build_class_map(d["label_map"], d["taxonomy"], alias)


def test_validator_alone_cannot_see_a_wrong_class_map():
    """Why the guard lives in the adapter: once mapped, a swapped class is a perfectly valid label."""
    d = load_inputs(FX)
    swapped = {0: "car", 2: "pedestrian", 3: "motorbike", 11: "sign"}           # person<->car swapped by hand
    boxes, _ = normalise_boxes(map_classes(d["preds"], swapped), d["images"])
    assert validate_predictions(boxes, d["taxonomy"], d["images"]).empty


def test_validator_flags_unknown_class_and_bad_geometry():
    imgs = pd.DataFrame({"image_id": ["a"], "width": [100], "height": [50]})
    pred = pd.DataFrame({"image_id": ["a", "a", "a", "a", "zz"], "class": ["car", "truck", "car", "car", "car"],
                         "x": [1.0, 1.0, -2.0, 95.0, 1.0], "y": [1.0] * 5, "w": [5.0, 5.0, 5.0, 10.0, 5.0], "h": [5.0, 5.0, 0.0, 5.0, 5.0]})
    bad = validate_predictions(pred, {0: "car"}, imgs)
    assert bad["problem"].tolist() == ["unknown_class;", "non_positive_size;negative_origin;", "out_of_bounds;", "unknown_image;"]


# ------------------------------------------------------------------ suspicion score + review queue
def _scored(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["image_id", "class", "x", "y", "w", "h", "conf", "clipped"])
    return add_suspicion(df, thr=0.25)


def test_suspicion_ranks_small_and_uncertain_boxes_above_confident_large_ones():
    s = _scored([
        ("a", "car", 0, 0, 80, 60, 0.95, False),            # large + confident
        ("b", "sign", 0, 0, 10, 10, 0.95, False),           # small (100 px2 < 150)
        ("c", "car", 0, 0, 80, 60, 0.50, False),            # mid confidence
        ("d", "sign", 0, 0, 10, 10, 0.50, False),           # small + mid confidence
        ("e", "car", 0, 0, 80, 60, 0.20, True),             # below threshold + clipped
    ])
    assert s["score"].tolist() == [0.05, 0.35, 0.5, 0.8, 0.533333]
    assert s["reason"].tolist() == ["low_priority", "small", "mid_conf", "mid_conf;small", "below_threshold;clipped"]
    assert s["score"].idxmax() == 3 and s["score"].idxmin() == 0


def test_suspicion_flags_overlapping_boxes_of_different_classes():
    s = _scored([
        ("a", "motorbike", 10, 10, 40, 40, 0.9, False),
        ("a", "pedestrian", 12, 12, 40, 40, 0.9, False),    # IoU ~0.82 with the motorbike
        ("a", "car", 200, 100, 40, 40, 0.9, False),
        ("b", "pedestrian", 10, 10, 40, 40, 0.9, False),    # same place, other image
    ])
    assert s["confusion"].tolist() == [True, True, False, False]
    assert "class_confusion" in s.loc[0, "reason"] and s.loc[0, "score"] > s.loc[2, "score"]


def test_review_queue_is_top_budget_by_score_and_enriches_small_boxes(tmp_path):
    assert run_main(tmp_path).returncode == 0
    q = pd.read_csv(tmp_path / "review_queue.csv")
    boxes = pd.read_csv(tmp_path / "boxes_all.csv")
    assert list(q.columns[:5]) == ["rank", "item_id", "score", "reason", "suggested_action"]
    assert len(q) == 30 and q["rank"].tolist() == list(range(1, 31))
    assert q["item_id"].is_unique and set(q["item_id"]) <= set(boxes["box_id"])
    assert q["score"].is_monotonic_decreasing
    assert q["score"].min() >= boxes[~boxes["box_id"].isin(q["item_id"])]["score"].max()   # nothing better was left out
    in_queue = boxes["box_id"].isin(q["item_id"])
    assert boxes[in_queue]["small"].mean() > 3 * boxes["small"].mean()                     # queue targets the hard slice


# ------------------------------------------------------------------ exporter QA
def test_roundtrip_passes_on_clean_export_and_fails_when_a_label_file_is_tampered(tmp_path):
    tax = {0: "car", 1: "pedestrian"}
    imgs = pd.DataFrame({"image_id": ["a", "b"], "path": ["images/a.jpg", "images/b.jpg"], "width": [320, 320], "height": [180, 180]})
    pred = pd.DataFrame({"image_id": ["a", "a"], "class": ["car", "pedestrian"], "x": [10.0, 100.5], "y": [20.0, 30.25],
                         "w": [50.0, 12.5], "h": [40.0, 30.0], "conf": [0.9, 0.6]})
    export_yolo(pred, tax, imgs, tmp_path / "yolo")
    export_coco(pred, tax, imgs, tmp_path / "coco.json")
    assert (tmp_path / "yolo" / "labels" / "b.txt").read_text(encoding="utf-8") == ""     # image without boxes still gets a file
    rt = roundtrip_check(pred, tax, imgs, tmp_path / "yolo", tmp_path / "coco.json")
    assert rt["yolo_ok"] and rt["coco_ok"] and rt["n_boxes"] == 2
    label = tmp_path / "yolo" / "labels" / "a.txt"
    label.write_text(label.read_text(encoding="utf-8").replace("0 ", "1 ", 1), encoding="utf-8")   # car -> pedestrian
    rt = roundtrip_check(pred, tax, imgs, tmp_path / "yolo", tmp_path / "coco.json")
    assert not rt["yolo_ok"] and rt["coco_ok"]


# ------------------------------------------------------------------ geometry
def test_normalise_clips_out_of_bounds_and_drops_empty_boxes():
    imgs = pd.DataFrame({"image_id": ["a"], "path": ["images/a.jpg"], "width": [100], "height": [50]})
    df = pd.DataFrame({"image_id": ["a"] * 3, "x": [90.0, -5.0, 200.0], "y": [10.0, 10.0, 10.0],
                       "w": [20.0, 10.0, 10.0], "h": [10.0, 10.0, 10.0]})
    out, dropped = normalise_boxes(df, imgs)
    assert dropped == 1                                              # fully outside -> no area
    assert out["clipped"].tolist() == [True, True]
    assert out.loc[0, "w"] == 10.0 and out.loc[1, "x"] == 0.0 and out.loc[1, "w"] == 5.0
    assert validate_predictions(out.assign(**{"class": "car"}), {0: "car"}, imgs).empty


def test_parse_override_formats_and_errors():
    names = {"car", "pedestrian"}
    assert parse_override("car", names) == {"class": "car"}
    assert parse_override("1,2,3,4", names) == {"box": (1.0, 2.0, 3.0, 4.0)}
    assert parse_override("pedestrian, 1,2,3,4", names) == {"class": "pedestrian", "box": (1.0, 2.0, 3.0, 4.0)}
    for bad in ("", "van", "1,2,3", "car,a,b,c,d"):
        with pytest.raises(ValueError):
            parse_override(bad, names)


# ------------------------------------------------------------------ end to end
def test_pipeline_end_to_end_is_valid_reproducible_and_holds_without_review(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    assert run_main(a).returncode == 0
    assert run_main(b).returncode == 0
    pred = pd.read_csv(a / "predictions.csv")
    assert list(pred.columns[:7]) == ["image_id", "class", "x", "y", "w", "h", "conf"]
    assert set(pred["class"]) <= {"car", "pedestrian", "motorbike", "sign"}
    qa = json.loads((a / "qa_report.json").read_text(encoding="utf-8"))
    assert qa["n_invalid"] == 0 and qa["yolo_ok"] and qa["coco_ok"]
    dec = json.loads((a / "decision.json").read_text(encoding="utf-8"))
    assert (dec["status"], dec["reason_code"]) == ("HOLD", "NOT_REVIEWED")     # valid file != reviewed file
    assert sha(a / "predictions.csv") == sha(b / "predictions.csv")            # seed 42 -> same bytes
    assert len(pd.read_csv(a / "review_queue.csv")) == 30


def test_baseline_exports_out_of_bounds_boxes_that_the_poc_fixes(tmp_path):
    base, poc = tmp_path / "base", tmp_path / "poc"
    assert run_main(base, "src.baseline_t14").returncode == 0
    assert run_main(poc).returncode == 0
    nb = json.loads((base / "qa_report.json").read_text(encoding="utf-8"))["n_invalid"]
    npoc = json.loads((poc / "qa_report.json").read_text(encoding="utf-8"))["n_invalid"]
    assert nb > 0 and npoc == 0


def _reviewed_from_queue(out: Path, edit_box: str = "10,10,20,20") -> pd.DataFrame:
    q = pd.read_csv(out / "review_queue.csv", dtype=str).fillna("")
    q["reviewer"], q["reviewer_decision"] = "tester", "approve"
    q.loc[0, "reviewer_decision"] = "reject"
    q.loc[1, ["reviewer_decision", "override_value"]] = ["edit", edit_box]
    return q


def test_review_loop_applies_reject_and_edit_logs_audit_and_reaches_go(tmp_path):
    out = tmp_path / "poc"
    assert run_main(out).returncode == 0
    q = _reviewed_from_queue(out)
    rejected, edited = q.loc[0, "item_id"], q.loc[1, "item_id"]
    q.to_csv(out / "review_queue_reviewed.csv", index=False)
    assert run_main(out).returncode == 0
    pred = pd.read_csv(out / "predictions.csv")
    assert rejected not in set(pred["box_id"])                                  # deleted by the reviewer
    row = pred[pred["box_id"] == edited].iloc[0]
    assert (row["x"], row["y"], row["w"], row["h"]) == (10.0, 10.0, 20.0, 20.0)  # edited box
    diff = pd.read_csv(out / "correction_diff.csv")
    assert len(diff) == 30 and set(diff["decision"]) == {"approve", "reject", "edit"}
    actions = {json.loads(l)["action"] for l in (out / "audit_log.jsonl").read_text(encoding="utf-8").splitlines()}
    assert {"review:reject", "review:edit", "review:approve"} <= actions
    dec = json.loads((out / "decision.json").read_text(encoding="utf-8"))
    assert (dec["status"], dec["reason_code"]) == ("GO", "REVIEWED_AND_VALID")


def test_partial_review_stays_on_hold(tmp_path):
    out = tmp_path / "poc"
    assert run_main(out).returncode == 0
    _reviewed_from_queue(out).head(3).to_csv(out / "review_queue_reviewed.csv", index=False)
    assert run_main(out).returncode == 0
    assert json.loads((out / "decision.json").read_text(encoding="utf-8"))["reason_code"] == "REVIEW_INCOMPLETE"


@pytest.mark.parametrize("mutate,why", [
    (lambda q: q.assign(reviewer=""), "decision without reviewer"),
    (lambda q: q.assign(item_id=["zz#9"] * len(q)), "item outside the queue"),
    (lambda q: q.assign(reviewer_decision="maybe"), "unknown decision value"),
])
def test_bad_review_file_is_rejected_with_exit_2(tmp_path, mutate, why):
    out = tmp_path / "poc"
    assert run_main(out).returncode == 0
    mutate(_reviewed_from_queue(out)).to_csv(out / "review_queue_reviewed.csv", index=False)
    assert run_main(out).returncode == 2, why


def test_missed_box_addition_needs_a_reviewer_and_is_exported(tmp_path):
    out = tmp_path / "poc"
    assert run_main(out).returncode == 0
    add = pd.DataFrame([{"image_id": "aa_00", "class": "car", "x": 5, "y": 5, "w": 30, "h": 20, "reviewer": "", "note": "van missed"}])
    add.to_csv(out / "review_additions.csv", index=False)
    assert run_main(out).returncode == 2                                        # owner is mandatory
    add.loc[0, "reviewer"] = "tester"
    add.to_csv(out / "review_additions.csv", index=False)
    assert run_main(out).returncode == 0
    pred = pd.read_csv(out / "predictions.csv")
    assert (pred["source"] == "human").sum() == 1


# ------------------------------------------------------------------ pipeline decide() unit tests
@pytest.fixture
def run_context_for_decide(tmp_path):
    import numpy as np
    return RunContext(
        track="t14",
        input_dir=tmp_path / "in",
        output_dir=tmp_path / "out",
        config={"owner": "Bui Van Thinh (Lead)", "params": {"min_review_fraction": 0.8}},
        seed=42,
        rng=np.random.default_rng(42),
        stats={"queue_size": 30, "n_reviewed": 30},
    )


@pytest.mark.parametrize("qa_override,case_desc", [
    ({"n_invalid": 1, "yolo_ok": True, "coco_ok": True}, "invalid_label_detected"),
    ({"n_invalid": 0, "yolo_ok": False, "coco_ok": True}, "yolo_roundtrip_failed"),
    ({"n_invalid": 0, "yolo_ok": True, "coco_ok": False}, "coco_roundtrip_failed"),
    ({"n_invalid": 3, "yolo_ok": False, "coco_ok": False}, "both_invalid_and_roundtrip_failed"),
])
def test_decide_rework_on_export_qa_failed(run_context_for_decide, qa_override, case_desc):
    base_qa = {
        "n_exported": 50,
        "n_invalid": 0,
        "invalid_label_rate": 0.0,
        "yolo_ok": True,
        "coco_ok": True,
        "yolo_max_err_px": 0.01,
        "coco_max_err_px": 0.01,
        "tol_px": 0.05,
        "n_boxes": 50,
    }
    qa = {**base_qa, **qa_override}
    decision = decide(run_context_for_decide, qa)
    assert decision.status == "REWORK", case_desc
    assert decision.canonical == "REWORK"
    assert decision.reason_code == "EXPORT_QA_FAILED"
    assert "Invalid labels or YOLO/COCO round-trip mismatch" in decision.reason
    assert decision.owner == "Bui Van Thinh (Lead)"
    assert decision.scope == "t14"
    assert any("qa_report.json" in ev for ev in decision.evidence)


def test_decide_rework_precedence_over_hold_and_go(tmp_path):
    import numpy as np
    # Trường hợp 1: n_reviewed = 0 (bình thường là HOLD / NOT_REVIEWED)
    ctx_unreviewed = RunContext(
        track="t14",
        input_dir=tmp_path,
        output_dir=tmp_path,
        config={"owner": "tester"},
        seed=42,
        rng=np.random.default_rng(42),
        stats={"queue_size": 30, "n_reviewed": 0},
    )
    bad_qa = {"n_invalid": 2, "yolo_ok": True, "coco_ok": True}
    dec_unreviewed = decide(ctx_unreviewed, bad_qa)
    assert dec_unreviewed.status == "REWORK"
    assert dec_unreviewed.reason_code == "EXPORT_QA_FAILED"

    # Trường hợp 2: n_reviewed = 30 / 30 (bình thường là GO)
    ctx_reviewed = RunContext(
        track="t14",
        input_dir=tmp_path,
        output_dir=tmp_path,
        config={"owner": "tester"},
        seed=42,
        rng=np.random.default_rng(42),
        stats={"queue_size": 30, "n_reviewed": 30},
    )
    dec_reviewed = decide(ctx_reviewed, bad_qa)
    assert dec_reviewed.status == "REWORK"
    assert dec_reviewed.reason_code == "EXPORT_QA_FAILED"

