"""Layering cho segmentation draft: z_order gan-xa, giai chong lan, occluded/truncated.

Quy uoc (khop cvat_tools.segmentation.layers): polygon NGUYEN VUNG (khong khoet lo), vat the nam tren nen
(nen am, vat the z >= 0), cung nhom vat the thi vat the GAN camera (y day lon hon) nam tren.

assign_layers(instances, H, W) -> list[dict(label, polygon_or_mask, polygon, z_order, occluded, truncated,
                                            occlusion_ratio, ...)]

Instance vao: {"label": str, "mask": bool HxW}  hoac  {"label": str, "polygon": [x0,y0,x1,y1,...]}, tuy chon
"conf", "id". Mask CHONG LAN duoc phep (do model sinh ra); ham tu giai chong lan theo thu tu gan-xa.

Che khuat: ratio_i = |mask_i ∩ hop(mask cac instance GAN HON)| / |mask_i|  (tinh tren mask goc).
mode="visible" (mac dinh): vung tra ra = phan nhin thay (roi rac nhau), lap lo kin neu lap khong de len
    instance xa hon -> painter z tang dan tai tao dung ban do chiem.
mode="full": tra mask goc (chong lan), z_order quyet dinh ai de len ai.

Chay do: CVAT_TOOLS_SRC=<learning>/cvat/src python -m tools.t14seg.layering --measure
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

# Cung bang voi cvat_tools LAYER_ORDER cho nhom nen; vat the (z=0) duoc xep hang gan-xa.
_STUFF_Z: dict[str, int] = {
    "sky": -9, "building": -8, "vegetation": -8, "terrain": -8, "wall": -8, "fence": -8,
    "sidewalk": -7, "pole": -2, "road": -1,
}
_ALIASES = {"person": "pedestrian", "traffic_light": "traffic light", "traffic_sign": "traffic sign"}


def _canon(label: str) -> str:
    return _ALIASES.get(label, label)


def _poly_to_mask(flat: Sequence[float], H: int, W: int) -> np.ndarray:
    import cv2
    m = np.zeros((H, W), np.uint8)
    pts = np.round(np.asarray(flat, np.float32).reshape(-1, 2)).astype(np.int32)
    if len(pts) >= 3:
        cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def _mask_to_polys(mask: np.ndarray, min_area: float) -> list[list[float]]:
    """Duong vien ngoai moi thanh phan lien thong (khong lo)."""
    import cv2
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cs:
        if cv2.contourArea(c) >= min_area and len(c) >= 3:
            out.append([float(v) for v in c.reshape(-1, 2).ravel()])
    return out


def _fill_safe_holes(vis: np.ndarray, farther_union: np.ndarray) -> np.ndarray:
    """Lap lo kin cua vung nhin thay, TRU lo chua pixel cua instance XA HON (painter se de nham len no)."""
    import cv2
    h = vis.astype(np.uint8)
    inv = (1 - h).astype(np.uint8)
    n, lab = cv2.connectedComponents(inv, connectivity=4)
    if n <= 2:
        return vis.copy()
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist())
    out = vis.copy()
    for k in range(1, n):
        if k in border:
            continue
        hole = lab == k
        if farther_union[hole].any():
            continue
        out |= hole
    return out


def depth_key(mask: np.ndarray) -> tuple[float, float]:
    """(y day vung 99.5%, y trong tam): lon hon = gan camera hon."""
    ys = np.nonzero(mask)[0]
    if ys.size == 0:
        return (-1.0, -1.0)
    return (float(np.percentile(ys, 99.5)), float(ys.mean()))


def assign_layers(instances: Sequence[dict[str, Any]], H: int, W: int, *, mode: str = "visible",
                  order: str = "depth", occ_thresh: float = 0.30, trunc_thresh_px: int = 3,
                  min_area: float = 4.0) -> list[dict[str, Any]]:
    """Gan z_order/occluded/truncated + giai chong lan. order: 'depth' (y day lon = gan) | 'area' (dien tich nho = tren).

    Tra list sap theo z_order tang dan (thu tu ve painter). Instance bi cat nhieu manh -> nhieu entry cung 'id'.
    """
    if mode not in ("visible", "full"):
        raise ValueError(mode)
    recs = []
    for i, ins in enumerate(instances):
        has_mask = ins.get("mask") is not None
        m = np.asarray(ins["mask"], bool) if has_mask else _poly_to_mask(ins["polygon"], H, W)
        if not m.any():
            continue
        label = _canon(ins["label"])
        recs.append({"id": ins.get("id", i), "label": label, "mask": m, "has_mask": has_mask,
                     "conf": float(ins.get("conf", 1.0)), "area": int(m.sum()), "key": depth_key(m)})

    things = [r for r in recs if r["label"] not in _STUFF_Z]
    if order == "depth":   # xa -> gan
        things.sort(key=lambda r: r["key"])
    elif order == "area":  # lon (xa) -> nho (gan)
        things.sort(key=lambda r: -r["area"])
    else:
        raise ValueError(order)
    for rank, r in enumerate(things):
        r["z"] = rank
    for r in recs:
        if r["label"] in _STUFF_Z:
            r["z"] = _STUFF_Z[r["label"]]

    ordered = sorted(recs, key=lambda r: (r["z"], r["key"]))
    # hop luy ke tu gan -> xa de tinh khuat va lo an toan
    nearer = np.zeros((H, W), bool)
    farther_all = np.zeros((H, W), bool)
    for r in ordered:
        farther_all |= r["mask"]
    out: list[dict[str, Any]] = []
    # khuat: chi tinh trong nhom cung loai tang (vat the voi vat the; nen voi vat the tren no)
    for idx, r in enumerate(ordered):
        above = [o["mask"] for o in ordered[idx + 1:]]
        cover = np.zeros((H, W), bool)
        for a in above:
            cover |= a
        ratio = float((r["mask"] & cover).sum() / r["area"])
        vis = r["mask"] & ~cover
        if mode == "visible":
            below = np.zeros((H, W), bool)
            for b in ordered[:idx]:
                below |= b["mask"] & ~r["mask"]
            res = _fill_safe_holes(vis, below)
        else:
            res = r["mask"]
        m = r["mask"]
        edge = int(m[0].sum() + m[-1].sum() + m[:, 0].sum() + m[:, -1].sum())
        truncated = edge >= trunc_thresh_px
        parts = _mask_to_polys(res, min_area)
        if not parts:
            parts = [[]]
        for p in parts:
            ent: dict[str, Any] = {
                "id": r["id"], "label": r["label"], "polygon": p, "z_order": int(r["z"]),
                "occluded": ratio >= occ_thresh, "truncated": truncated,
                "occlusion_ratio": round(ratio, 4), "conf": r["conf"],
            }
            ent["polygon_or_mask"] = res if r["has_mask"] else p
            out.append(ent)
    return out


def render_painter(layers: Sequence[dict[str, Any]], H: int, W: int, label_ids: dict[str, int]) -> np.ndarray:
    """Painter z tang dan (stable) -> canvas id (0 = trong). Doc lap voi viec ban render dung dien tich."""
    import cv2
    canvas = np.zeros((H, W), np.int32)
    for k, e in enumerate(sorted(layers, key=lambda e: e["z_order"]), 1):
        if len(e["polygon"]) >= 6:
            pts = np.round(np.asarray(e["polygon"], np.float32).reshape(-1, 2)).astype(np.int32)
            cv2.fillPoly(canvas, [pts], k)
    return canvas


# ------------------------------------------------------------------ tests
def _rect(H: int, W: int, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_near_car_on_top_and_occlusion() -> None:
    H, W = 100, 100
    far = _rect(H, W, 10, 10, 60, 50)     # day y=50
    near = _rect(H, W, 30, 30, 90, 80)    # day y=80 -> gan hon
    L = assign_layers([{"label": "car", "mask": far}, {"label": "car", "mask": near}], H, W)
    by = {tuple(sorted({e["id"]})): e for e in L}
    assert by[(1,)]["z_order"] > by[(0,)]["z_order"]
    assert by[(1,)]["occlusion_ratio"] == 0.0 and not by[(1,)]["occluded"]
    assert by[(0,)]["occluded"] and 0.2 < by[(0,)]["occlusion_ratio"] < 0.5
    # polygon xa khong bi giao voi gan (visible), painter z tang dan khong de mat xe gan
    canvas = render_painter(L, H, W, {})
    zs = sorted(L, key=lambda e: e["z_order"])
    assert canvas[60, 60] == zs.index(by[(1,)]) + 1  # vung chong lan thuoc xe gan


def test_truncated_and_stuff_z() -> None:
    H, W = 50, 50
    L = assign_layers([{"label": "car", "mask": _rect(H, W, 0, 10, 20, 30)},
                       {"label": "road", "mask": _rect(H, W, 0, 30, 50, 50)}], H, W)
    car = next(e for e in L if e["label"] == "car")
    road = next(e for e in L if e["label"] == "road")
    assert car["truncated"] and road["z_order"] == -1 and car["z_order"] >= 0


def test_painter_matches_depth_ownership() -> None:
    H, W = 80, 80
    ms = [_rect(H, W, 5, 5, 40, 40), _rect(H, W, 20, 20, 60, 60), _rect(H, W, 35, 30, 75, 75)]
    L = assign_layers([{"label": "car", "mask": m, "id": i} for i, m in enumerate(ms)], H, W)
    canvas = render_painter(L, H, W, {})
    byk = {k: e for k, e in enumerate(sorted(L, key=lambda e: e["z_order"]), 1)}
    ownership = np.zeros((H, W), int)
    for k in sorted(byk, key=lambda k: byk[k]["z_order"]):
        ownership[ms[byk[k]["id"]]] = byk[k]["id"] + 1
    got = np.zeros((H, W), int)
    for k, e in byk.items():
        got[canvas == k] = e["id"] + 1
    assert (got != ownership).mean() < 0.005


# ------------------------------------------------------------------ do tren drive_030 (proxy, khong co gold)
def _measure(images: list[str], out_dir: str) -> None:  # pragma: no cover
    import json
    import os
    import sys
    import time
    from pathlib import Path
    import cv2
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    import torch
    torch.set_num_threads(2)
    from tools import t14_seg_local as T
    paths, *_ = T._import_cvat()
    from ultralytics import YOLO
    yolo = YOLO(str(Path(paths.COPILOT) / "yolo26s-seg.pt"))
    names = {v: k for k, v in T.NAMES.items()}
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    res = {}
    for name in images:
        img = cv2.imread(str(T.REAL / "images" / f"{name}.jpg"))
        H, W = img.shape[:2]
        t0 = time.time()
        inst, _ = T.instance_masks(yolo, img, 0.25, 1280)
        insts = [{"label": T.NAMES[cs], "mask": mk, "conf": c, "id": i} for i, (cs, c, mk) in enumerate(inst)
                 if T.NAMES[cs] in ("car", "truck", "bus", "motorcycle", "bicycle")]
        vehicles = len(insts)
        masks = [i["mask"] for i in insts]

        def overlap_pairs(ms: list[np.ndarray], frac: float = 0.01) -> int:
            n = 0
            for a in range(len(ms)):
                for b in range(a + 1, len(ms)):
                    inter = int((ms[a] & ms[b]).sum())
                    if inter and inter >= frac * min(ms[a].sum(), ms[b].sum()):
                        n += 1
            return n

        pairs_before = overlap_pairs(masks)
        row: dict[str, Any] = {"n_vehicle_instances": vehicles, "overlap_pairs_before": pairs_before}
        for order in ("depth", "area"):
            L = assign_layers(insts, H, W, mode="visible", order=order)
            vis_masks: dict[Any, np.ndarray] = {}
            for e in L:
                if len(e["polygon"]) >= 6:
                    m = _poly_to_mask(e["polygon"], H, W)
                    vis_masks[e["id"]] = vis_masks.get(e["id"], np.zeros((H, W), bool)) | m
            row[f"overlap_pairs_after_{order}"] = overlap_pairs(list(vis_masks.values()))
            ids = {e["id"] for e in L}
            ratios = {e["id"]: e["occlusion_ratio"] for e in L}
            row[f"occluded_gt50_{order}"] = sum(1 for v in ratios.values() if v > 0.5)
            row[f"occluded_flag_{order}"] = sum(1 for e in {e["id"]: e for e in L}.values() if e["occluded"])
            row[f"n_ids_{order}"] = len(ids)
            # painter z tang dan phai tai tao ownership (moi pixel -> instance gan nhat dang phu no)
            Lf = assign_layers(insts, H, W, mode="full", order=order)
            canvas = render_painter(Lf, H, W, {})
            rank = {id(e): k for k, e in enumerate(sorted(Lf, key=lambda e: e["z_order"]), 1)}
            idmap = np.zeros(canvas.max() + 1, np.int32)
            for e in Lf:
                idmap[rank[id(e)]] = e["id"] + 1
            row[f"_canvas_{order}"] = idmap[canvas]
            # tham chieu: chu ky pixel-owner theo gan-xa tu mask goc
            if order == "depth":
                z = {e["id"]: e["z_order"] for e in Lf}
                own = np.zeros((H, W), np.int32)
                for i in sorted(z, key=lambda i: z[i]):
                    own[insts[i]["mask"]] = i + 1
                row["painter_vs_ownership_diff_px_depth"] = int((row["_canvas_depth"] != own).sum())
        inst_union = np.zeros((H, W), bool)
        for m in masks:
            inst_union |= m
        d = row["_canvas_depth"] != row["_canvas_area"]
        row["render_diff_depth_vs_area_pct_of_image"] = round(100 * float(d.mean()), 4)
        row["render_diff_depth_vs_area_pct_of_vehicle_union"] = round(100 * float(d.sum() / max(1, inst_union.sum())), 3)
        # so pixel ma cars-nearer-top khac cars-bigger-first: chi o vung chong lan
        ov = np.zeros((H, W), np.int32)
        for m in masks:
            ov += m
        row["overlap_region_px"] = int((ov > 1).sum())
        row["render_diff_pct_of_overlap_region"] = round(100 * float(d.sum() / max(1, (ov > 1).sum())), 2)
        row.pop("_canvas_depth"), row.pop("_canvas_area")
        row["sec"] = round(time.time() - t0, 1)
        res[name] = row
        print(name, json.dumps(row), flush=True)
    json.dump(res, open(od / "layering_measure.json", "w"), indent=1)


if __name__ == "__main__":  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--images", default="drive_030")
    ap.add_argument("--out", default="out/opt/layering")
    a = ap.parse_args()
    if a.measure:
        _measure(a.images.split(","), a.out)
    else:
        test_near_car_on_top_and_occlusion()
        test_truncated_and_stuff_z()
        test_painter_matches_depth_ownership()
        print("ok")
