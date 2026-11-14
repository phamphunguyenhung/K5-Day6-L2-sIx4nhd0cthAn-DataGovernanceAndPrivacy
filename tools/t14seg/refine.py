"""SAM refinement of YOLO instance masks (box prompt from YOLO mask bbox), with safety gates + cache.

Instance = (cs_id, conf, mask_bool HxW). refine_instances() returns the same type, same order/length.
Safety: a SAM mask replaces the YOLO one only if it passes the gates; otherwise the YOLO mask is kept.
Metrics here are PROXY (agreement with YOLO), not correctness: no gold.
CLI: python -m tools.t14seg.refine --selftest | --bench mobilesam|sam2
Weights: env COPILOT_DIR or learning/tools/cvat-seg-copilot.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Instance = Tuple[int, float, np.ndarray]
ROOT = Path(__file__).resolve().parents[2]
_MODELS: Dict[str, object] = {}
WEIGHTS = {"mobilesam": "mobile_sam.pt", "sam2": "sam2.1_b.pt"}


@dataclass
class RefineCfg:
    backend: str = "mobilesam"
    threads: int = 2
    enabled: bool = True
    # policy: which instances go to SAM
    conf_lo: float = 0.0          # SAM only if conf >= conf_lo
    conf_hi: float = 0.80         # high-conf clean YOLO masks kept unless large/rough
    min_box_px: int = 24          # min box side; SAM on tiny objects is noise
    big_area_frac: float = 0.01   # instances above this image fraction always eligible
    rough_thresh: float = 1.12    # boundary roughness above this => eligible even if conf high
    # safety gates (SAM mask rejected if violated)
    min_iou: float = 0.60         # IoU(sam, yolo)
    max_box_ratio: float = 1.5    # sam area / box area (overflow)
    area_ratio: Tuple[float, float] = (0.6, 1.5)  # sam area / yolo area
    box_margin: float = 0.08      # sam mask must stay within box expanded by this fraction
    # blur/uncertainty gates -> 'drop' (instance removed from output; closure fills it with the background below)
    drop_enabled: bool = True
    clear_conf: float = 0.35      # YOLO mask is "clear" only if conf >= this ...
    clear_main_frac: float = 0.85  # ... and largest connected component holds >= this share of its pixels
    clear_fill: float = 0.20      # ... and mask area / box area >= this
    sam_main_frac: float = 0.90   # SAM mask torn (fragmented) below this
    sam_rough_max: float = 1.35   # SAM mask too jagged above this
    cache_dir: Optional[Path] = None
    dropped: List[int] = field(default_factory=list)  # input indices dropped in the last call
    stats: Dict[str, int] = field(default_factory=dict)


def _weights_dir() -> Path:
    e = os.environ.get("COPILOT_DIR")
    if e:
        return Path(e)
    for p in ROOT.parents:
        d = p / "tools" / "cvat-seg-copilot"
        if d.exists():
            return d
    raise FileNotFoundError("tools/cvat-seg-copilot (set COPILOT_DIR)")


def _load(backend: str, threads: int):
    if backend not in _MODELS:
        import torch
        torch.set_num_threads(threads)
        from ultralytics import SAM
        _MODELS[backend] = SAM(str(_weights_dir() / WEIGHTS[backend]))
    return _MODELS[backend]


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def boundary_roughness(mask: np.ndarray) -> float:
    """Contour length / contour length of Gaussian-smoothed mask. 1.0 = smooth; larger = jagged. Proxy only."""
    import cv2
    bb = mask_bbox(mask)
    if bb is None or mask.sum() < 16:
        return 1.0
    pad = 6
    c = mask.astype(np.uint8)[max(0, bb[1] - pad):bb[3] + pad, max(0, bb[0] - pad):bb[2] + pad]

    def per(x: np.ndarray) -> float:
        cs, _ = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return float(sum(cv2.arcLength(k, True) for k in cs))

    s = (cv2.GaussianBlur(c.astype(np.float32), (0, 0), 2.0) > 0.5).astype(np.uint8)
    ps = per(s)
    return per(c) / ps if ps > 0 else 1.0


def passes_gates(sam: np.ndarray, yolo: np.ndarray, box: Tuple[int, int, int, int], cfg: RefineCfg) -> str:
    """'' if SAM mask acceptable, else rejection reason."""
    sa, ya = int(sam.sum()), int(yolo.sum())
    if sa == 0:
        return "empty"
    bw, bh = box[2] - box[0], box[3] - box[1]
    if sa > cfg.max_box_ratio * bw * bh:
        return "overflow_box"
    r = sa / max(ya, 1)
    if not (cfg.area_ratio[0] <= r <= cfg.area_ratio[1]):
        return "area_ratio"
    mx, my = cfg.box_margin * bw, cfg.box_margin * bh
    sb = mask_bbox(sam)
    if sb is None or sb[0] < box[0] - mx or sb[1] < box[1] - my or sb[2] > box[2] + mx or sb[3] > box[3] + my:
        return "outside_box"
    if mask_iou(sam, yolo) < cfg.min_iou:
        return "low_iou"
    if main_frac(sam) < cfg.sam_main_frac:
        return "torn"
    if boundary_roughness(sam) > cfg.sam_rough_max:
        return "jagged"
    return ""


def main_frac(mask: np.ndarray) -> float:
    """Share of mask pixels in its largest 8-connected component (1.0 = one solid blob)."""
    import cv2
    bb = mask_bbox(mask)
    if bb is None:
        return 0.0
    c = mask.astype(np.uint8)[bb[1]:bb[3], bb[0]:bb[2]]
    n, _, st, _ = cv2.connectedComponentsWithStats(c, connectivity=8)
    return float(st[1:, cv2.CC_STAT_AREA].max() / max(int(c.sum()), 1)) if n > 1 else 0.0


def yolo_is_clear(conf: float, mask: np.ndarray, box: Tuple[int, int, int, int], cfg: RefineCfg) -> bool:
    bw, bh = box[2] - box[0], box[3] - box[1]
    return (conf >= cfg.clear_conf and mask.sum() >= cfg.clear_fill * bw * bh
            and main_frac(mask) >= cfg.clear_main_frac)


def eligible(conf: float, mask: np.ndarray, box: Tuple[int, int, int, int], cfg: RefineCfg) -> bool:
    """Policy: SAM for mid-confidence, large, or rough instances; skip tiny and clean high-conf ones."""
    if min(box[2] - box[0], box[3] - box[1]) < cfg.min_box_px or conf < cfg.conf_lo:
        return False
    if conf < cfg.conf_hi:
        return True
    if mask.sum() >= cfg.big_area_frac * mask.size:
        return True
    return boundary_roughness(mask) >= cfg.rough_thresh


def sam_mask(img_bgr: np.ndarray, box: Tuple[int, int, int, int], cfg: RefineCfg) -> np.ndarray:
    """Local-crop SAM with box prompt -> bool mask in full-image coordinates."""
    import cv2
    H, W = img_bgr.shape[:2]
    m = int(max(box[2] - box[0], box[3] - box[1]) * 0.25) + 8
    x0, y0, x1, y1 = max(0, box[0] - m), max(0, box[1] - m), min(W, box[2] + m), min(H, box[3] + m)
    crop = np.ascontiguousarray(img_bgr[y0:y1, x0:x1])
    res = _load(cfg.backend, cfg.threads)(crop, bboxes=[[box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0]],
                                          device="cpu", verbose=False)
    out = np.zeros((H, W), bool)
    if not res or res[0].masks is None or len(res[0].masks.data) == 0:
        return out
    mk = res[0].masks.data[0].cpu().numpy().astype(np.uint8)
    if mk.shape != crop.shape[:2]:
        mk = cv2.resize(mk, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    out[y0:y1, x0:x1] = mk > 0
    return out


def _key(h: str, box: Sequence[int], backend: str) -> str:
    return hashlib.sha256(f"{h}|{backend}|{tuple(int(v) for v in box)}".encode()).hexdigest()[:32]


def _cache_get(cfg: RefineCfg, key: str, shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if not cfg.cache_dir:
        return None
    p = Path(cfg.cache_dir) / f"{key}.npy"
    if not p.exists():
        return None
    return np.unpackbits(np.load(p))[: shape[0] * shape[1]].reshape(shape).astype(bool)


def _cache_put(cfg: RefineCfg, key: str, mask: np.ndarray) -> None:
    if cfg.cache_dir:
        Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)
        np.save(Path(cfg.cache_dir) / f"{key}.npy", np.packbits(mask.reshape(-1)))


def refine_instances(img_bgr: np.ndarray, instances: Sequence[Instance], cfg: Optional[RefineCfg] = None,
                     img_sha256: Optional[str] = None) -> List[Instance]:
    """Same-typed output, same order. Counters in cfg.stats: eligible/accepted/skipped/rejected_<why>/cache_hit."""
    cfg = cfg or RefineCfg()
    st = cfg.stats
    if not cfg.enabled:
        return list(instances)
    h = img_sha256 or hashlib.sha256(np.ascontiguousarray(img_bgr).tobytes()).hexdigest()
    out: List[Instance] = []
    cfg.dropped = []

    def drop(i: int, why: str) -> None:
        cfg.dropped.append(i)
        st["dropped_" + why] = st.get("dropped_" + why, 0) + 1

    for i, (cs, conf, mk) in enumerate(instances):
        box = mask_bbox(mk)
        if box is None:
            if cfg.drop_enabled:
                drop(i, "empty_yolo")
            else:
                out.append((cs, conf, mk))
            continue
        clear = yolo_is_clear(conf, mk, box, cfg)
        if not eligible(conf, mk, box, cfg):
            st["skipped"] = st.get("skipped", 0) + 1
            if cfg.drop_enabled and not clear:
                drop(i, "yolo_unclear")
            else:
                out.append((cs, conf, mk))
            continue
        st["eligible"] = st.get("eligible", 0) + 1
        key = _key(h, box, cfg.backend)
        sm = _cache_get(cfg, key, mk.shape)
        if sm is None:
            sm = sam_mask(img_bgr, box, cfg)
            _cache_put(cfg, key, sm)
        else:
            st["cache_hit"] = st.get("cache_hit", 0) + 1
        why = passes_gates(sm, mk, box, cfg)
        if not why:
            st["accepted"] = st.get("accepted", 0) + 1
            out.append((cs, conf, sm))
            continue
        st["rejected_" + why] = st.get("rejected_" + why, 0) + 1
        if cfg.drop_enabled and not clear:
            drop(i, "both_unclear")
        else:
            out.append((cs, conf, mk))  # SAM rejected, YOLO clear enough -> keep YOLO
    return out


# ---------------------------------------------------------------- tests / bench
def _selftest() -> None:
    a = np.zeros((100, 100), bool)
    a[20:60, 20:60] = True
    cfg = RefineCfg()
    box = mask_bbox(a)
    assert box == (20, 20, 60, 60)
    assert passes_gates(a, a, box, cfg) == ""
    assert passes_gates(np.zeros_like(a), a, box, cfg) == "empty"
    assert passes_gates(np.ones_like(a), a, box, cfg) == "overflow_box"
    sh = np.zeros_like(a)
    sh[20:60, 45:85] = True
    assert passes_gates(sh, a, box, cfg) in ("low_iou", "outside_box")
    assert abs(mask_iou(a, a) - 1.0) < 1e-9
    frag = np.zeros((100, 100), bool)
    frag[20:30, 20:30] = True
    frag[50:60, 50:60] = True
    assert main_frac(frag) < 0.6 and main_frac(a) == 1.0
    assert yolo_is_clear(0.9, a, box, cfg) and not yolo_is_clear(0.2, a, box, cfg)
    assert not yolo_is_clear(0.9, frag, mask_bbox(frag), cfg)
    off = RefineCfg(enabled=False)
    assert refine_instances(np.zeros((100, 100, 3), np.uint8), [(13, .9, a)], off)[0][2] is a
    print("selftest ok")


def _bench(backend: str, names: Sequence[str], out_dir: Path) -> dict:
    import json
    import time
    import cv2
    from ultralytics import YOLO
    coco_to_cs = {0: 11, 2: 13, 3: 17, 5: 15, 7: 14, 1: 18}
    yolo = YOLO(str(_weights_dir() / "yolo26s-seg.pt"))
    rows: List[dict] = []
    summary: dict = {}
    cfg = RefineCfg(backend=backend)  # bench calls sam_mask directly (ungated); gates evaluated afterwards
    for n in names:
        img = cv2.imread(str(ROOT / "data/real/day02_traffic/images" / f"{n}.jpg"))
        H, W = img.shape[:2]
        r = yolo.predict(img, conf=0.25, imgsz=1280, device="cpu", verbose=False)[0]
        if r.masks is None:
            continue
        for m, c, s in zip(r.masks.data.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()):
            if c not in coco_to_cs:
                continue
            mk = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            box = mask_bbox(mk)
            if box is None or min(box[2] - box[0], box[3] - box[1]) < 24:
                continue
            t = time.perf_counter()
            sm = sam_mask(img, box, cfg)
            dt = time.perf_counter() - t
            ya, sa = int(mk.sum()), int(sm.sum())
            ba = (box[2] - box[0]) * (box[3] - box[1])
            rows.append(dict(img=n, cs=int(coco_to_cs[c]), conf=float(s), area_frac=ya / (H * W), t=dt,
                             iou=mask_iou(sm, mk), empty=sa == 0, overflow=sa > 1.5 * ba, area_ratio=sa / max(ya, 1),
                             rough_y=boundary_roughness(mk), rough_s=boundary_roughness(sm) if sa else None,
                             gate=passes_gates(sm, mk, box, RefineCfg())))
    if rows:
        io = np.array([r["iou"] for r in rows])
        tt = np.array([r["t"] for r in rows])
        ok = [r for r in rows if r["rough_s"] is not None]
        gates: Dict[str, int] = {}
        for r in rows:
            gates[r["gate"] or "ok"] = gates.get(r["gate"] or "ok", 0) + 1
        summary = dict(backend=backend, n=len(rows), t_mean=float(tt.mean()), t_med=float(np.median(tt)),
                       iou_mean=float(io.mean()), iou_med=float(np.median(io)), iou_lt06=int((io < .6).sum()),
                       empty=int(sum(r["empty"] for r in rows)), overflow=int(sum(r["overflow"] for r in rows)),
                       rough_yolo=float(np.mean([r["rough_y"] for r in ok])),
                       rough_sam=float(np.mean([r["rough_s"] for r in ok])), gates=gates)
        for nm, g in (("hi", [r for r in rows if r["conf"] >= .8]), ("lo", [r for r in rows if r["conf"] < .8])):
            if g:
                summary[f"iou_conf_{nm}"] = float(np.mean([r["iou"] for r in g]))
                summary[f"n_conf_{nm}"] = len(g)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"bench_{backend}.json").write_text(json.dumps(dict(summary=summary, rows=rows), indent=1))
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--bench", choices=list(WEIGHTS))
    ap.add_argument("--imgs", default="drive_001,drive_002,drive_010,drive_030,drive_057")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    if a.bench:
        print(_bench(a.bench, a.imgs.split(","), ROOT / "out" / "opt" / "sam_refine"))
