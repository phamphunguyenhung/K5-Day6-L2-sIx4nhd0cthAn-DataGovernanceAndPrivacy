"""Tests for tools/t14seg/evaluate.py. ALL data here is SYNTHETIC (not gold, not real images)."""
import numpy as np
import pytest

from tools.t14seg import evaluate as ev
from tools.t14seg.evaluate import GAP, EvalError, Sample


def synth(stem="s0", shift=0, h=60, w=80, gap=False):
    """SYNTHETIC: sky top, road bottom, one car rectangle (instance 1)."""
    lab = np.full((h, w), 0, np.uint8)
    lab[:20] = 10
    inst = np.zeros((h, w), np.int32)
    lab[30:50, 20 + shift:50 + shift] = 13
    inst[30:50, 20 + shift:50 + shift] = 1
    if gap:
        lab[:, 0:4] = GAP
    return Sample(stem, lab, inst, 1)


def sets(n=5, **kw):
    return {f"s{i}": synth(f"s{i}", **kw) for i in range(n)}


def test_perfect_pred_scores_one():
    g = sets()
    rep = ev.eval_gold(g, g, "heldout")
    m = rep["metrics"]
    assert m["mIoU"]["value"] == pytest.approx(1.0)
    assert m["instF1@0.5"]["value"] == 1.0 and m["instF1@0.75"]["value"] == 1.0
    assert m["boundaryF"]["value"] == 1.0 and m["gap_rate"]["value"] == 0.0
    assert rep["warning"]  # n=5 < 30


def test_known_iou_and_instance_threshold():
    g, p = sets(), sets(shift=5)  # car shifted 5 of 30 px: IoU = 25/35
    rep = ev.eval_gold(p, g, "heldout")
    m = rep["metrics"]
    assert m["instF1@0.5"]["value"] == 1.0
    assert m["instF1@0.75"]["value"] == 0.0
    assert rep["per_class_IoU"]["car"] == pytest.approx(25 / 35, abs=1e-6)
    assert m["mIoU"]["value"] < 1.0


def test_gap_rate_baseline_diff_and_seed():
    g = sets()
    pred, base = sets(), sets(shift=5)
    r1 = ev.eval_gold(pred, g, "heldout", baseline=base)
    r2 = ev.eval_gold(pred, g, "heldout", baseline=base)
    assert r1 == r2  # deterministic seed 42
    assert r1["metrics"]["mIoU"]["diff"] > 0
    assert ev.eval_gold(sets(gap=True), g, "heldout")["metrics"]["gap_rate"]["value"] == pytest.approx(4 / 80)


def test_refuse_name_size_split():
    g = sets()
    with pytest.raises(EvalError, match="names differ"):
        ev.eval_gold({k + "x": v for k, v in g.items()}, g, "heldout")
    bad = sets()
    bad["s0"] = synth("s0", h=61)
    with pytest.raises(EvalError, match="size mismatch"):
        ev.eval_gold(bad, g, "heldout")
    with pytest.raises(EvalError, match="heldout"):
        ev.eval_gold(g, g, "tune")
    with pytest.raises(EvalError, match="tune thresholds"):
        ev.eval_gold(g, g, "heldout", tuned_stems=["s1.jpg"])


def test_proxy_is_labelled_and_no_accuracy_claim():
    rep = ev.eval_proxy(sets(), sets(shift=5), {"s0": 1.5})
    assert ev.PROXY_LABEL in rep["note"] and rep["label"] == ev.PROXY_LABEL
    assert "metrics" not in rep and 0 < rep["agreement_mIoU_mean"] < 1
    assert ev.PROXY_LABEL in ev.format_report(rep)


def test_iaa():
    rep = ev.eval_iaa(sets(), sets(shift=5))
    assert rep["mode"] == "iaa" and rep["metrics"]["mIoU"]["value"] < 1.0


def test_coco_render_and_png_roundtrip(tmp_path):
    import cv2
    coco = {"images": [{"id": 1, "file_name": "a.jpg", "width": 40, "height": 30}],
            "categories": [{"id": 14, "name": "car"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 14,
                             "segmentation": [[5, 5, 25, 5, 25, 20, 5, 20]]}]}
    s = ev.render_coco(coco)["a"]
    assert (s.labels == 13).sum() > 200 and s.inst.max() == 1 and s.labels[0, 39] == GAP
    cv2.imwrite(str(tmp_path / "a.png"), s.labels)
    cv2.imwrite(str(tmp_path / "a_inst.png"), s.inst.astype(np.uint16))
    t = ev.load_set(tmp_path)["a"]
    assert (t.labels == s.labels).all() and (t.inst == s.inst).all()


def test_gap_gate_hard_and_dropped_rate():
    g = sets()
    ok = ev.eval_gold(g, g, "heldout")
    assert ok["gap_gate"] == "PASS" and ok["dropped_polygon_rate"] == 0.0
    p = sets()
    p["s0"].n_dropped = 4  # 4 dropped vs 5 kept
    r = ev.eval_gold(p, g, "heldout")
    assert r["dropped_polygon_rate"] == pytest.approx(4 / 9)
    assert r["metrics"]["instRecall@0.5"]["value"] == 1.0
    bad = ev.eval_gold(sets(gap=True), g, "heldout")
    assert bad["gap_gate"] == "FAIL" and "FAIL" in ev.format_report(bad)
    # gap outside gold-valid area still fails the gate
    h = sets()
    h["s1"].labels[0, 0] = GAP
    assert ev.eval_gold(h, g, "heldout")["gap_gate"] == "FAIL"
