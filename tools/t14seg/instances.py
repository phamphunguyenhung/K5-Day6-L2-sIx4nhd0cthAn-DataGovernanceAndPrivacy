"""Instance layer for the T14 segmentation draft: YOLO-seg full pass + optional SAHI-style tiles + night enhance.

detect_instances(img_bgr, cfg) -> (instances, ood)
  instances: list[(cs_id, conf, mask_bool HxW)]  (COCO->Cityscapes as t14_seg_local.COCO_TO_CS)
  ood: number of raw detections (full pass + tiles, before merge) of classes that do not map
Benchmark (5 fixed images, proxy metrics only):  python -m tools.t14seg.instances bench [config names]
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

COCO_TO_CS: dict[int, int] = {0: 11, 2: 13, 3: 17, 5: 15, 7: 14, 1: 18, 9: 6, 11: 7}  # mirror of t14_seg_local
WEIGHTS_DIR = Path(r"D:\06-Learning\AIA-VINGROUP\learning\tools\cvat-seg-copilot")
Inst = tuple[int, float, Any]
_MODELS: dict[str, Any] = {}


@dataclass(frozen=True)
class InstCfg:
    model: str = "yolo26s-seg"
    imgsz: int = 1280
    conf: float = 0.25
    iou: float = 0.7              # YOLO internal NMS
    tile: bool = False
    tile_size: int = 320          # tile side in original px
    tile_overlap: float = 0.25
    tile_y: tuple[float, float] = (0.15, 0.75)  # only tile this vertical band (far field), fractions of H
    tile_imgsz: int = 960
    merge_iou: float = 0.5        # class-aware mask IoU for cross-tile NMS (containment uses max(thr, 0.8))
    night_thr: float = 70.0       # mean_gray below -> enhance
    night_clahe: float = 3.0
    night_gamma: float = 0.7
    threads: int = 2
    min_area: int = 16
    keep_conf: float = 0.0        # drop 'blurry' instances: conf below this (0 = off)
    keep_area: int = 0            # drop instances smaller than this many px (0 = off)
    keep_solid: float = 0.0       # drop if largest connected component < this fraction of mask (torn mask)


def _model(name: str, threads: int):
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(threads)
    if name not in _MODELS:
        _MODELS[name] = YOLO(str(WEIGHTS_DIR / f"{name}.pt"))
    return _MODELS[name]


def enhance_night(img_bgr, cfg: InstCfg):
    """CLAHE on L + gamma; only when mean_gray < cfg.night_thr. -> (img, applied)."""
    import cv2
    import numpy as np
    if float(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).mean()) >= cfg.night_thr:
        return img_bgr, False
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=cfg.night_clahe, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    lut = (255.0 * (np.arange(256) / 255.0) ** cfg.night_gamma).astype(np.uint8)
    return cv2.LUT(out, lut), True


def _predict(model, img, cfg: InstCfg, imgsz: int) -> list[tuple[int, float, Any]]:
    """-> list of (coco_cls, conf, mask_bool at img size)."""
    r = model.predict(img, conf=cfg.conf, iou=cfg.iou, imgsz=imgsz, device="cpu", verbose=False, retina_masks=True)[0]
    if r.masks is None:
        return []
    ms = r.masks.data.cpu().numpy() > 0.5
    cl = r.boxes.cls.cpu().numpy().astype(int)
    sc = r.boxes.conf.cpu().numpy()
    return [(int(c), float(s), m) for m, c, s in zip(ms, cl, sc)]


def _tile_boxes(H: int, W: int, cfg: InstCfg) -> list[tuple[int, int, int, int]]:
    y0, y1 = int(cfg.tile_y[0] * H), int(cfg.tile_y[1] * H)
    ts = cfg.tile_size
    step = max(1, int(ts * (1 - cfg.tile_overlap)))
    ys = list(range(y0, max(y0 + 1, y1 - ts + 1), step)) or [y0]
    if ys[-1] + ts < y1:
        ys.append(max(y0, y1 - ts))
    xs = list(range(0, max(1, W - ts + 1), step)) or [0]
    if xs[-1] + ts < W:
        xs.append(max(0, W - ts))
    return [(x, y, min(W, x + ts), min(H, y + ts)) for y in ys for x in xs]


def _nms_class_aware(dets: list[tuple[int, float, Any]], thr: float) -> list[tuple[int, float, Any]]:
    """Greedy by conf; drop same-class det when mask IoU >= thr or containment (inter/min area) >= max(thr, .8)."""
    import numpy as np
    dets = sorted(dets, key=lambda d: -d[1])
    keep: list[tuple[int, float, Any]] = []
    areas: list[int] = []
    for c, s, m in dets:
        a = int(m.sum())
        ok = True
        for (kc, _, km), ka in zip(keep, areas):
            if kc != c:
                continue
            inter = int(np.logical_and(m, km).sum())
            if inter and (inter / (a + ka - inter) >= thr or inter / max(1, min(a, ka)) >= max(thr, 0.8)):
                ok = False
                break
        if ok:
            keep.append((c, s, m))
            areas.append(a)
    return keep


def drop_blurry(insts: list[Inst], cfg: InstCfg) -> tuple[list[Inst], dict[str, int]]:
    """Remove unsure instances (low conf / tiny / torn mask). Dropped pixels are NOT repainted here: the caller's
    compose() paints instances over the closed full_stuff_map, so they fall back to the background beneath (no hole).
    Kept masks are reduced to their largest connected component. -> (kept, stats)."""
    import cv2
    import numpy as np
    kept: list[Inst] = []
    st = {"low_conf": 0, "tiny": 0, "torn": 0, "empty": 0, "released_px": 0}
    for c, s, m in insts:
        a = int(m.sum())
        why = None
        if a == 0:
            why = "empty"
        elif s < cfg.keep_conf:
            why = "low_conf"
        elif a < cfg.keep_area:
            why = "tiny"
        else:
            n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
            big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            if stats[big, cv2.CC_STAT_AREA] / a < cfg.keep_solid:
                why = "torn"
            elif n > 2:
                m = lab == big
        if why:
            st[why] += 1
            st["released_px"] += a
        else:
            kept.append((c, s, m))
    return kept, st


def detect_instances(img_bgr, cfg: InstCfg | None = None) -> tuple[list[Inst], int]:
    import numpy as np
    cfg = cfg or InstCfg()
    H, W = img_bgr.shape[:2]
    img, _ = enhance_night(img_bgr, cfg)
    model = _model(cfg.model, cfg.threads)
    raw = _predict(model, img, cfg, cfg.imgsz)
    if cfg.tile:
        for x0, y0, x1, y1 in _tile_boxes(H, W, cfg):
            for c, s, m in _predict(model, img[y0:y1, x0:x1], cfg, cfg.tile_imgsz):
                ys, xs = np.where(m)
                if len(ys) == 0:
                    continue
                # drop detections cut by an interior tile border (neighbour tile / full pass sees them whole)
                if ((xs.min() == 0 and x0 > 0) or (xs.max() == x1 - x0 - 1 and x1 < W)
                        or (ys.min() == 0 and y0 > 0) or (ys.max() == y1 - y0 - 1 and y1 < H)):
                    continue
                full = np.zeros((H, W), bool)
                full[y0:y1, x0:x1] = m
                raw.append((c, s, full))
    ood = sum(1 for c, _, _ in raw if c not in COCO_TO_CS)
    mapped = [(COCO_TO_CS[c], s, m) for c, s, m in raw if c in COCO_TO_CS and int(m.sum()) >= cfg.min_area]
    out = _nms_class_aware(mapped, cfg.merge_iou)
    if cfg.keep_conf or cfg.keep_area or cfg.keep_solid:
        out, _ = drop_blurry(out, cfg)
    return out, ood


# ----------------------------------------------------------------- proxy metrics / benchmark
def _union(insts: list[Inst], shape):
    import numpy as np
    u = np.zeros(shape, bool)
    for _, _, m in insts:
        u |= m
    return u


def agreement(a: list[Inst], b: list[Inst]) -> dict[str, float]:
    """Proxy stability: union-mask IoU and mean best-match same-class instance IoU (a->b and b->a averaged)."""
    import numpy as np
    if not a and not b:
        return {"union_iou": 1.0, "match_iou": 1.0}
    shape = (a or b)[0][2].shape
    ua, ub = _union(a, shape), _union(b, shape)
    u = float(np.logical_and(ua, ub).sum() / max(1, np.logical_or(ua, ub).sum()))

    def best(x, y):
        v = []
        for c, _, m in x:
            bi = 0.0
            for c2, _, m2 in y:
                if c2 == c:
                    i = np.logical_and(m, m2).sum()
                    if i:
                        bi = max(bi, float(i / np.logical_or(m, m2).sum()))
            v.append(bi)
        return float(np.mean(v)) if v else 1.0
    return {"union_iou": u, "match_iou": 0.5 * (best(a, b) + best(b, a))}


CONFIGS: dict[str, InstCfg] = {
    "s_1280_c25": InstCfg(),
    "s_960_c25": InstCfg(imgsz=960),
    "s_1536_c25": InstCfg(imgsz=1536),
    "m_1280_c25": InstCfg(model="yolo26m-seg"),
    "l_1280_c25": InstCfg(model="yolo26l-seg"),
    "s_1280_c15": InstCfg(conf=0.15),
    "s_1280_c35": InstCfg(conf=0.35),
    "s_1280_c20_tile": InstCfg(conf=0.20, tile=True),
    "m_1280_c20_tile": InstCfg(model="yolo26m-seg", conf=0.20, tile=True),
    "s_1280_c20_tile_clean": InstCfg(conf=0.20, tile=True, keep_conf=0.35, keep_area=40, keep_solid=0.8),
}
IMAGES = ["drive_001", "drive_002", "drive_010", "drive_030"]  # dev only; drive_057 is EVAL, never used to tune


def bench(names: list[str] | None = None) -> dict[str, Any]:
    import json
    import cv2
    root = Path(__file__).resolve().parents[2]
    out = root / "out" / "opt" / "instances"
    out.mkdir(parents=True, exist_ok=True)
    res: dict[str, Any] = {}
    store: dict[str, dict[str, list[Inst]]] = {}
    for cn in names or list(CONFIGS):
        cfg = CONFIGS[cn]
        rows = []
        for im in IMAGES:
            img = cv2.imread(str(root / "data/real/day02_traffic/images" / f"{im}.jpg"))
            t = time.perf_counter()
            inst, ood = detect_instances(img, cfg)
            dt = time.perf_counter() - t
            store.setdefault(cn, {})[im] = inst
            rows.append({"img": im, "n": len(inst), "small": sum(int(m.sum()) < 400 for _, _, m in inst),
                         "ood": ood, "px": int(_union(inst, img.shape[:2]).sum()), "sec": round(dt, 2), "night": enhance_night(img, cfg)[1]})
        res[cn] = rows
        (out / "bench.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(cn, [(r["n"], r["small"], r["sec"]) for r in rows], flush=True)
    ref = "m_1280_c20_tile" if "m_1280_c20_tile" in store else next(iter(store))
    agr = {}
    for cn in store:
        v = [agreement(store[cn][im], store[ref][im]) for im in IMAGES]
        agr[cn] = {"vs": ref, "union_iou": round(sum(x["union_iou"] for x in v) / len(v), 3),
                   "match_iou": round(sum(x["match_iou"] for x in v) / len(v), 3)}
    res["_agreement"] = agr
    (out / "bench.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(agr, indent=1))
    return res


def _self_test() -> None:
    import numpy as np
    m = np.zeros((20, 20), bool)
    m[2:10, 2:10] = True
    m2 = m.copy()
    m3 = np.zeros_like(m)
    m3[15:19, 15:19] = True
    k = _nms_class_aware([(13, .9, m), (13, .5, m2), (13, .4, m3), (11, .3, m2)], 0.5)
    assert [(c, s) for c, s, _ in k] == [(13, .9), (13, .4), (11, .3)]
    assert len(_tile_boxes(1000, 1920, InstCfg())) >= 1
    dark = np.full((50, 50, 3), 20, np.uint8)
    assert enhance_night(dark, InstCfg())[1] and not enhance_night(dark + 200, InstCfg())[1]
    assert abs(agreement([(13, 1, m)], [(13, 1, m2)])["match_iou"] - 1) < 1e-9
    print("self_test ok")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        bench(sys.argv[2:] or None)
    else:
        _self_test()
