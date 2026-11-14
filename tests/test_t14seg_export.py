"""Validator tests for tools/t14seg/cvat_export.py (synthetic files, no images, no network)."""
import json
import zipfile
from pathlib import Path

import pytest

from tools.t14seg.cvat_export import validate_coco, validate_cvat_xml

SQ = [10.0, 10.0, 50.0, 10.0, 50.0, 40.0, 10.0, 40.0]  # 40x30 = 1200 px^2


FULL = [0.0, 0.0, 100.0, 0.0, 100.0, 80.0, 0.0, 80.0]


def _coco(tmp: Path, name="c.zip", ann_ids=(1, 2), cat_name="car", pts=SQ) -> Path:
    anns = [{"id": i, "image_id": 1, "category_id": 1, "segmentation": [pts], "area": 1200.0,
             "bbox": [10.0, 10.0, 40.0, 30.0], "iscrowd": 0} for i in ann_ids]
    d = {"images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 80}], "annotations": anns,
         "categories": [{"id": 1, "name": cat_name, "supercategory": ""}]}
    p = tmp / name
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("annotations/instances_default.json", json.dumps(d))
    return p


def _xml(tmp: Path, name="x.zip", label="car", pts="10,10;50,10;50,40;10,40") -> Path:
    x = ('<?xml version="1.0"?><annotations><version>1.1</version><meta><task><size>1</size><labels><label><name>car</name>'
         '<type>polygon</type><attributes><attribute><name>ai_draft</name><mutable>False</mutable><input_type>checkbox</input_type>'
         '</attribute></attributes></label></labels></task></meta>'
         f'<image id="0" name="a.jpg" width="100" height="80"><polygon label="{label}" source="auto" occluded="0" points="{pts}" z_order="0">'
         '<attribute name="ai_draft">true</attribute></polygon></image></annotations>')
    p = tmp / name
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("annotations.xml", x)
    return p


def test_valid_files_pass(tmp_path):
    rc = validate_coco(_coco(tmp_path, ann_ids=(1, 2)), task_images=["a.jpg"])
    assert rc.ok, rc.errors
    rx = validate_cvat_xml(_xml(tmp_path), task_images=["a.jpg"])
    assert rx.ok, rx.errors


def test_duplicate_ids_rejected(tmp_path):
    r = validate_coco(_coco(tmp_path, ann_ids=(1, 1)))
    assert not r.ok and any("duplicate annotation id" in e for e in r.errors)


def test_unknown_label_rejected(tmp_path):
    r = validate_cvat_xml(_xml(tmp_path, label="dragon"))
    assert not r.ok and any("not declared" in e for e in r.errors)
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps([{"name": "car", "attributes": []}]), encoding="utf-8")
    r2 = validate_coco(_coco(tmp_path, "c2.zip", cat_name="dragon"), labels_json=labels)
    assert not r2.ok and any("not declared in labels.json" in e for e in r2.errors)


def test_point_outside_image_rejected(tmp_path):
    r = validate_cvat_xml(_xml(tmp_path, pts="10,10;500,10;50,40"))
    assert not r.ok and any("outside image" in e for e in r.errors)
    r2 = validate_coco(_coco(tmp_path, pts=[10.0, 10.0, 500.0, 10.0, 50.0, 40.0]))
    assert not r2.ok and any("outside image" in e for e in r2.errors)


def test_missing_task_image_rejected(tmp_path):
    r = validate_cvat_xml(_xml(tmp_path), task_images=["other.jpg"])
    assert not r.ok and any("not in the task images" in e for e in r.errors)


def test_gap_is_caught(tmp_path):
    # SQ covers 1200 of 8000 px: gap must FAIL; full-frame polygon must pass with gap == 0
    r = validate_coco(_coco(tmp_path, ann_ids=(1,)), max_gap_pct=0.0)
    assert not r.ok and any("GAP FAIL" in e for e in r.errors) and r.stats["gap_violations"] == ["a.jpg"]
    rx = validate_cvat_xml(_xml(tmp_path), max_gap_pct=0.0)
    assert not rx.ok and any("GAP FAIL" in e for e in rx.errors)
    full = validate_cvat_xml(_xml(tmp_path, "f.zip", pts="0,0;100,0;100,80;0,80"), max_gap_pct=0.0)
    assert full.ok, full.errors
    # a thin 1px slit left uncovered is still a gap
    slit = validate_cvat_xml(_xml(tmp_path, "s.zip", pts="0,0;100,0;100,78;0,78"), max_gap_pct=0.0)
    assert not slit.ok
