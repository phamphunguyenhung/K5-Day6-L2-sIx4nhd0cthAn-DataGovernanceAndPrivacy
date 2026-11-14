"""T14 real-data extension (Lane B): local segmentation DRAFT on the dev images. No CVAT, no network.

Pipeline per image: SegFormer-B2 (Cityscapes) -> full_stuff_map (background closed edge-to-edge)
-> overlay SegFormer thing pixels -> overlay YOLO-seg instances -> export label PNG, overlay at opacity 1.0,
COCO polygons and a gap report measured on the RENDERED polygons (the thing that matters at opacity 100%).

Rule being checked: with opacity pulled to max, regions must meet edge-to-edge; only pixels that belong to
no class may stay open.

Needs the external cvat tools on disk (read-only, not modified):
    set CVAT_TOOLS_SRC=<learning>/cvat/src
    python -m tools.t14_seg_local --limit 1            # smoke
    python -m tools.t14_seg_local                      # all split=dev images

Every box/polygon is `ai_draft_unreviewed` until a human reviews it. Output goes to out/seg_local (gitignored).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data" / "real" / "day02_traffic"

# COCO (yolo seg) -> Cityscapes id of the segmentation label set
COCO_TO_CS = {0: 11, 2: 13, 3: 17, 5: 15, 7: 14, 1: 18, 9: 6, 11: 7}
THING_IDS = (5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18)  # pole, light, sign, person..bicycle
NAMES = {0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence", 5: "pole", 6: "traffic light",
         7: "traffic sign", 8: "vegetation", 9: "terrain", 10: "sky", 11: "pedestrian", 12: "rider", 13: "car",
         14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle"}
GAP = 255


def _import_cvat():
    src = os.environ.get("CVAT_TOOLS_SRC")
    if not src or not Path(src).is_dir():
        sys.exit("Set CVAT_TOOLS_SRC to the folder containing cvat_tools (e.g. <learning>/cvat/src).")
    sys.path.insert(0, src)
    from cvat_tools.core import paths
    from cvat_tools.segmentation.labelmap import full_stuff_map
    from cvat_tools.segmentation.layers import z_order
    from cvat_tools.segmentation.onnx_seg import OnnxSeg
    return paths, full_stuff_map, z_order, OnnxSeg


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def instance_masks(yolo, img, conf, imgsz=1280):
    """-> list of (cs_id, conf, bool mask HxW) for mapped classes only; unmapped classes counted as OOD."""
    H, W = img.shape[:2]
    r = yolo.predict(img, conf=conf, imgsz=imgsz, device="cpu", verbose=False)[0]
    out, ood = [], 0
    if r.masks is None:
        return out, ood
    for m, c, s in zip(r.masks.data.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()):
        if c not in COCO_TO_CS:
            ood += 1
            continue
        mk = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        out.append((COCO_TO_CS[c], float(s), mk))
    return out, ood


def compose(cm, full_stuff_map, inst):
    """Edge-to-edge label map: closed background, SegFormer things, then instance masks on top."""
    lab = full_stuff_map(cm)
    th = np.isin(cm, THING_IDS)
    lab = np.where(th, cm, lab).astype(np.uint8)  # things win even where the stuff map filled them through
    imap = np.zeros(lab.shape, np.uint16)  # 0 = no instance (semantic-only region), k = YOLO instance index
    for k, (cs, _, mk) in enumerate(sorted(inst, key=lambda t: -int(t[2].sum())), 1):  # big first, small on top
        lab[mk] = cs
        imap[mk] = k
    return lab, imap


def polygons(lab, imap=None, min_px=1, eps=0.0):
    """One whole-region polygon per connected component (outer contour, no holes), one per INSTANCE for things
    (touching cars stay separate). -> [(cs_id, flat_pts, is_instance)]."""
    res = []
    key = lab.astype(np.int32)
    if imap is not None:
        key = np.where(imap > 0, 1000 + imap.astype(np.int32), key)
    for k in np.unique(key):
        if k == GAP:
            continue
        m = (key == k).astype(np.uint8)
        cid = int(lab[m.astype(bool)][0])
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cs:
            if cv2.contourArea(c) < min_px:
                continue
            a = (cv2.approxPolyDP(c, eps, True) if eps > 0 else c).reshape(-1, 2)
            if len(a) >= 3:
                res.append((cid, [float(v) for v in a.ravel()], bool(k >= 1000)))
    return res


def render(polys, H, W, z_order):
    """Painter's algorithm like an annotation tool at opacity 100%. Big area first: a region nested in a hole is
    always smaller than its parent, so this reproduces the label map exactly (z_order is NOT used for drawing)."""
    canvas = np.full((H, W), GAP, np.uint8)
    for cid, pts, _ in sorted(polys, key=lambda t: -cv2.contourArea(np.array(t[1], np.float32).reshape(-1, 2))):
        cv2.fillPoly(canvas, [np.round(np.array(pts, np.float32).reshape(-1, 2)).astype(np.int32)], int(cid))
    return canvas


def palette():
    rng = np.random.RandomState(7)
    p = rng.randint(40, 230, (256, 3)).astype(np.uint8)
    p[GAP] = (0, 0, 0)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--yolo", default="yolo26s-seg.pt")
    ap.add_argument("--imgsz", type=int, default=1280, help="YOLO-seg input size; 640 missed half the cars in dense scenes")
    ap.add_argument("--out", default=str(ROOT / "out" / "seg_local"))
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    np.random.seed(a.seed)

    paths, full_stuff_map, z_order, OnnxSeg = _import_cvat()
    from ultralytics import YOLO
    wdir = Path(paths.COPILOT)
    seg = OnnxSeg("b2")
    yolo = YOLO(str(wdir / a.yolo))

    rows = [r for r in csv.DictReader(open(REAL / "images.csv", encoding="utf-8")) if r["split"] == a.split]
    rows.sort(key=lambda r: r["image_id"])
    if a.limit:
        rows = rows[:a.limit]
    man = {r["file_id"]: r for r in csv.DictReader(open(REAL / "manifest.csv", encoding="utf-8"))}

    out = Path(a.out)
    for d in ("labels", "overlay", "polys"):
        (out / d).mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:8]
    pal = palette()
    coco = {"images": [], "annotations": [], "categories": [{"id": k + 1, "name": v} for k, v in NAMES.items()]}
    stats, aid = [], 1
    t_all = time.time()
    for i, r in enumerate(rows, 1):
        iid = r["image_id"]
        p = REAL / r["path"].replace("\\", "/")
        h = sha256(p)
        exp = man.get(iid, {}).get("sha256")
        if exp and exp != h:
            sys.exit(f"SHA256 mismatch for {iid}")
        img = cv2.imread(str(p))
        H, W = img.shape[:2]
        t0 = time.time()
        cm, info = seg.class_map(img)
        inst, ood = instance_masks(yolo, img, a.conf, a.imgsz)
        lab, imap = compose(cm, full_stuff_map, inst)
        polys = polygons(lab, imap)
        rend = render(polys, H, W, z_order)
        gap_label = float((lab == GAP).mean())
        gap_render = float((rend == GAP).mean())
        diff = float((rend != lab).mean())  # polygon simplification / hole-fill mismatch vs label map
        cv2.imwrite(str(out / "labels" / f"{iid}.png"), lab)
        cv2.imwrite(str(out / "labels" / f"{iid}_inst.png"), imap)
        cv2.imwrite(str(out / "overlay" / f"{iid}.jpg"), pal[rend])
        coco["images"].append({"id": i, "file_name": f"{iid}.jpg", "width": W, "height": H, "sha256": h})
        for cid, pts, is_inst in polys:
            area = float(cv2.contourArea(np.array(pts, np.float32).reshape(-1, 2)))
            xs, ys = pts[0::2], pts[1::2]
            coco["annotations"].append({
                "id": aid, "image_id": i, "category_id": cid + 1, "segmentation": [pts], "area": area,
                "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)], "iscrowd": 0,
                "attributes": {"provenance": "ai_draft_unreviewed", "run_id": run_id,
                               "source_model": f"segformer-b2-cityscapes+{a.yolo}", "image_sha256": h,
                               "review_status": "unreviewed", "shape_source": "instance" if is_inst else "semantic"}})
            aid += 1
        stats.append({"image_id": iid, "sec": round(time.time() - t0, 2), "n_polys": len(polys),
                      "n_instances": len(inst), "ood_dropped": ood, "gap_label_pct": round(100 * gap_label, 3),
                      "gap_render_pct": round(100 * gap_render, 3), "render_vs_label_diff_pct": round(100 * diff, 3)})
        print(f"[{i}/{len(rows)}] {iid} {stats[-1]['sec']}s polys={len(polys)} gap_render={stats[-1]['gap_render_pct']}%", flush=True)

    json.dump(coco, open(out / "polys" / "coco_segmentation_draft.json", "w", encoding="utf-8"))
    with open(out / "gap_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
        w.writeheader()
        w.writerows(stats)
    json.dump({"run_id": run_id, "images": len(rows), "seed": a.seed, "split": a.split,
               "wall_sec": round(time.time() - t_all, 1), "provenance": "ai_draft_unreviewed",
               "max_gap_render_pct": max(s["gap_render_pct"] for s in stats),
               "mean_gap_render_pct": round(sum(s["gap_render_pct"] for s in stats) / len(stats), 4)},
              open(out / "run_info.json", "w", encoding="utf-8"), indent=1)
    print("done", out)


if __name__ == "__main__":
    main()
