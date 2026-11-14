"""Export the local segmentation draft (tools/t14_seg_local.py) as files you can IMPORT into CVAT by hand.

LOCAL FILES ONLY: this script never talks to a CVAT server and never uploads anything.

    python -m tools.t14_export_cvat                      # reads out/seg_local, writes out/seg_local/cvat_import
    python -m tools.t14_export_cvat --eps 0.8            # polygon simplification in px (0 = exact contours)

Produces (out/seg_local/cvat_import/):
  labels.json                  label spec to paste in "Constructor -> Raw" when creating the CVAT task
  images/                      the dev images, same file names the annotations refer to (local copy, PII risk: do not publish)
  coco_1.0.zip                 annotations/instances_default.json        -> CVAT: Upload annotations -> "COCO 1.0"
  cvat_images_1.1.zip          annotations.xml (polygons, source="auto") -> CVAT: Upload annotations -> "CVAT 1.1"
  export_report.csv/.json      gap / mismatch measured on the RENDERED simplified polygons, plus counts

Every shape is an AI draft (source="auto", attribute ai_draft=true, provenance in the sidecar). A human must review
and fix it in CVAT before it is called a label. Import order in CVAT: create task with images/ -> import ONE of the zips.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from xml.sax.saxutils import quoteattr

import cv2
import numpy as np

from tools.t14_seg_local import GAP, NAMES, REAL, polygons, render

ROOT = Path(__file__).resolve().parents[1]
COLORS = {k: "#%02x%02x%02x" % tuple(int(v) for v in np.random.RandomState(100 + k).randint(60, 230, 3)) for k in NAMES}


def simplified(lab, imap, eps):
    return polygons(lab, imap, min_px=1, eps=eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "out" / "seg_local"))
    ap.add_argument("--eps", type=float, default=0.8)
    a = ap.parse_args()
    src = Path(a.src)
    info = json.load(open(src / "run_info.json", encoding="utf-8"))
    draft = json.load(open(src / "polys" / "coco_segmentation_draft.json", encoding="utf-8"))
    out = src / "cvat_import"
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)

    used = sorted({int(a_["category_id"]) - 1 for a_ in draft["annotations"]})
    cats = [{"id": k + 1, "name": NAMES[k], "supercategory": ""} for k in used]
    labels = [{"name": NAMES[k], "color": COLORS[k], "type": "polygon",
               "attributes": [{"name": "ai_draft", "mutable": False, "input_type": "checkbox", "default_value": "true",
                               "values": ["true"]},
                              {"name": "run_id", "mutable": False, "input_type": "text", "default_value": info["run_id"],
                               "values": [info["run_id"]]}]} for k in used]
    json.dump(labels, open(out / "labels.json", "w", encoding="utf-8"), indent=1)

    images, anns, xml_img, rep = [], [], [], []
    aid = 1
    for i, im in enumerate(draft["images"], 1):
        iid = Path(im["file_name"]).stem
        W, H = im["width"], im["height"]
        shutil.copy2(REAL / "images" / im["file_name"], out / "images" / im["file_name"])
        lab = cv2.imread(str(src / "labels" / f"{iid}.png"), cv2.IMREAD_UNCHANGED)
        imap = cv2.imread(str(src / "labels" / f"{iid}_inst.png"), cv2.IMREAD_UNCHANGED)
        polys = sorted(simplified(lab, imap, a.eps), key=lambda t: -cv2.contourArea(np.array(t[1], np.float32).reshape(-1, 2)))
        rend = render(polys, H, W, None)
        gap = float((rend == GAP).mean())
        diff = float((rend != lab).mean())
        images.append({"id": i, "file_name": im["file_name"], "width": W, "height": H})
        shapes = []
        for z, (cid, pts, is_inst) in enumerate(polys):  # z_order = rank by area: bigger = lower, matches the render
            area = float(cv2.contourArea(np.array(pts, np.float32).reshape(-1, 2)))
            xs, ys = pts[0::2], pts[1::2]
            anns.append({"id": aid, "image_id": i, "category_id": cid + 1, "segmentation": [[round(v, 1) for v in pts]],
                         "area": round(area, 1), "bbox": [round(min(xs), 1), round(min(ys), 1),
                                                          round(max(xs) - min(xs), 1), round(max(ys) - min(ys), 1)],
                         "iscrowd": 0, "attributes": {"ai_draft": True, "run_id": info["run_id"],
                                                     "shape_source": "instance" if is_inst else "semantic"}})
            aid += 1
            p = ";".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
            shapes.append(f'    <polygon label={quoteattr(NAMES[cid])} source="auto" occluded="0" points="{p}" z_order="{z}">\n'
                          f'      <attribute name="ai_draft">true</attribute>\n      <attribute name="run_id">{info["run_id"]}</attribute>\n'
                          f'    </polygon>')
        xml_img.append(f'  <image id="{i - 1}" name={quoteattr(im["file_name"])} width="{W}" height="{H}">\n'
                       + "\n".join(shapes) + "\n  </image>")
        rep.append({"image_id": iid, "polygons": len(polys), "points": sum(len(p) // 2 for _, p, _i in polys), "instance_polys": sum(1 for *_x, i in polys if i),
                    "gap_render_pct": round(100 * gap, 4), "render_vs_label_diff_pct": round(100 * diff, 4)})

    coco = {"info": {"description": "AI draft, unreviewed", "version": "1.0", "year": 2026},
            "licenses": [{"id": 1, "name": "CC-BY-4.0 (UA-DETRAC via Day02)", "url": ""}],
            "images": images, "annotations": anns, "categories": cats}
    xml_labels = "\n".join(f'      <label>\n        <name>{NAMES[k]}</name>\n        <color>{COLORS[k]}</color>\n        <type>polygon</type>\n'
                           f'        <attributes>\n          <attribute><name>ai_draft</name><mutable>False</mutable><input_type>checkbox</input_type>'
                           f'<default_value>true</default_value><values>true</values></attribute>\n'
                           f'          <attribute><name>run_id</name><mutable>False</mutable><input_type>text</input_type>'
                           f'<default_value>{info["run_id"]}</default_value><values>{info["run_id"]}</values></attribute>\n'
                           f'        </attributes>\n      </label>' for k in used)
    xml = ('<?xml version="1.0" encoding="utf-8"?>\n<annotations>\n  <version>1.1</version>\n  <meta>\n    <task>\n'
           f'      <name>t14-seg-draft-{info["run_id"]}</name>\n      <size>{len(images)}</size>\n      <mode>annotation</mode>\n'
           f'      <labels>\n{xml_labels}\n      </labels>\n    </task>\n  </meta>\n' + "\n".join(xml_img) + "\n</annotations>\n")

    with zipfile.ZipFile(out / "coco_1.0.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("annotations/instances_default.json", json.dumps(coco))
    with zipfile.ZipFile(out / "cvat_images_1.1.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("annotations.xml", xml)

    with open(out / "export_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rep[0].keys()))
        w.writeheader()
        w.writerows(rep)
    summ = {"run_id": info["run_id"], "eps_px": a.eps, "images": len(images), "annotations": len(anns),
            "labels": [NAMES[k] for k in used], "provenance": "ai_draft_unreviewed",
            "max_gap_render_pct": max(r["gap_render_pct"] for r in rep),
            "mean_gap_render_pct": round(sum(r["gap_render_pct"] for r in rep) / len(rep), 4),
            "max_render_vs_label_diff_pct": max(r["render_vs_label_diff_pct"] for r in rep),
            "zip_sha256": {n: hashlib.sha256((out / n).read_bytes()).hexdigest() for n in ("coco_1.0.zip", "cvat_images_1.1.zip")},
            "cvat_upload_done": False}
    json.dump(summ, open(out / "export_report.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
