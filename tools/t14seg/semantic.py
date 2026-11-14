"""Semantic layer (SegFormer-B2 Cityscapes ONNX) used ONLY for background; things come from instances.

Public API
    semantic_labels(img_bgr, cfg) -> (class_map uint8 HxW Cityscapes ids, info dict)
    thing_residual(cm, instance_union_mask, ids=THING_IDS) -> bool mask of thing pixels with no instance
    apply_residual(cm, residual, mode, ...) -> (stuff_label_map, kept_semantic_only_mask, stats)
    PRESETS: named input policies (see `python -m tools.t14seg.semantic --help`)

cfg keys (all optional): preset | policy (inputspec policy dict) | ensemble (list of presets, softmax mean)
    threads (default 2), with_conf (bool, adds info["conf"] uint8 max-prob map), cvat_src (else env CVAT_TOOLS_SRC)
Heavy imports (onnxruntime, cvat_tools) are lazy. Proxy metrics only: no gold exists.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

# Cityscapes ids
STUFF_BG = (0, 1, 2, 3, 4, 8, 9, 10)          # road sidewalk building wall fence vegetation terrain sky
SMALL_FIXTURES = (5, 6, 7)                    # pole, traffic light, traffic sign (thin/small background-ish objects)
THING_IDS = (11, 12, 13, 14, 15, 16, 17, 18)  # person rider car truck bus train motorcycle bicycle
UNLABELED = 255

PRESETS: Dict[str, dict] = {
    "ls640": {"long_side": 640, "enhance": "none"},
    "ls1024": {"long_side": 1024, "enhance": "none"},          # model_inputs.yaml default
    "ls1536": {"long_side": 1536, "enhance": "none"},
    "ls1024_dark": {"long_side": 1024, "enhance": "gamma_clahe"},
    "ls1024_gamma": {"long_side": 1024, "enhance": "gamma"},
    "tile512": {"long_side": 1024, "enhance": "none", "tile": {"size": 512, "overlap": 0.25}},
    "tile768": {"long_side": 1536, "enhance": "none", "tile": {"size": 768, "overlap": 0.25}},
}
DEFAULT_PRESET = "ls1024_dark"  # enhance only fires when mean gray < dark_thresh, so identical to ls1024 on bright frames

_SESS: Dict[Tuple[str, int], Any] = {}
_INPUTSPEC: Any = None


def _cvat() -> Any:
    global _INPUTSPEC
    src = os.environ.get("CVAT_TOOLS_SRC")
    if src and src not in sys.path:
        sys.path.insert(0, src)
    from cvat_tools.segmentation.onnx_seg import OnnxSeg  # noqa: WPS433 lazy
    return OnnxSeg


def _seg(threads: int) -> Any:
    key = ("b2", threads)
    if key not in _SESS:
        _SESS[key] = _cvat()("b2", threads=threads)
    return _SESS[key]


def _softmax(lg: np.ndarray) -> np.ndarray:
    e = np.exp(lg - lg.max(0, keepdims=True))
    return e / e.sum(0, keepdims=True)


def semantic_labels(img_bgr: np.ndarray, cfg: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
    """SegFormer-B2 class map at original resolution, via the input contract (inputspec).
    Returns (uint8 HxW Cityscapes ids, info{seconds, policy, in_hw, enhance, [conf]})."""
    cfg = cfg or {}
    seg = _seg(int(cfg.get("threads", 2)))
    presets: List[str] = list(cfg.get("ensemble") or [])
    t0 = time.time()
    info: Dict[str, Any] = {}
    if presets:  # softmax-mean over policies (costs sum of times)
        acc, metas = None, []
        for p in presets:
            lg, meta = seg.logits(img_bgr, PRESETS[p])
            pr = _softmax(lg)
            acc = pr if acc is None else acc + pr
            metas.append({"preset": p, "in_hw": meta["in_hw"], "tiles": len(meta["tiles"] or [1])})
        probs = acc / len(presets)
        info["policy"] = {"ensemble": metas}
    else:
        name = cfg.get("preset", DEFAULT_PRESET)
        pol = cfg.get("policy") or PRESETS[name]
        lg, meta = seg.logits(img_bgr, pol)
        probs = _softmax(lg) if cfg.get("with_conf") else None
        info["policy"] = {"preset": name if not cfg.get("policy") else "custom", **pol}
        info["in_hw"], info["enhance"] = meta["in_hw"], meta["enhance"]
        cm = lg.argmax(0).astype(np.uint8)
    if probs is not None:
        cm = probs.argmax(0).astype(np.uint8)
        if cfg.get("with_conf"):
            top2 = np.partition(probs, -2, axis=0)[-2:]
            info["conf"] = (top2[1] * 255).astype(np.uint8)  # max prob
            info["margin"] = ((top2[1] - top2[0]) * 255).astype(np.uint8)  # top1 - top2
    info["s"] = round(time.time() - t0, 3)
    return cm, info


def thing_residual(cm: np.ndarray, instance_union_mask: np.ndarray, ids: Iterable[int] = THING_IDS) -> np.ndarray:
    """bool HxW: pixels SegFormer calls a 'thing' (default person..bicycle) that no instance covers."""
    return np.isin(cm, tuple(ids)) & ~instance_union_mask.astype(bool)


def apply_residual(cm: np.ndarray, residual: np.ndarray, mode: str = "hybrid", min_area: int = 150,
                   conf: Optional[np.ndarray] = None, min_conf: int = 200, min_extent: int = 6
                   ) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Decide what to do with residual thing pixels.
    mode 'flag'  : keep the semantic thing label, return all residual as semantic_only (reviewer must check)
         'drop'  : reassign every residual pixel to nearest background class (no gap)
         'hybrid': keep components with area>=min_area, short side>=min_extent and (if conf given) mean conf>=min_conf
                   as semantic_only; drop the rest to nearest background
    Returns (label_map where kept residual keeps its thing id and all else is background-closed,
             semantic_only bool mask, stats)."""
    import cv2

    cm_bg = np.where(np.isin(cm, THING_IDS), UNLABELED, cm).astype(np.uint8)
    # background used for filling dropped pixels: nearest known background (things/unknown are fill targets)
    known = np.isin(cm_bg, STUFF_BG + SMALL_FIXTURES)
    keep = np.zeros(residual.shape, bool)
    comps = 0
    if mode in ("flag", "hybrid"):
        if mode == "flag":
            keep = residual.copy()
        else:
            n, lab, st, _ = cv2.connectedComponentsWithStats(residual.astype(np.uint8), connectivity=8)
            for i in range(1, n):
                a, w, h = st[i, cv2.CC_STAT_AREA], st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
                if a < min_area or min(w, h) < min_extent:
                    continue
                m = lab == i
                if conf is not None and float(conf[m].mean()) < min_conf:
                    continue
                keep |= m
    comps = int(cv2.connectedComponents(keep.astype(np.uint8), connectivity=8)[0] - 1) if keep.any() else 0
    fill = (~known) & ~keep  # includes dropped residual and any other unknown pixel
    out = cm_bg.copy()
    if fill.any() and known.any():
        _, lbl = cv2.distanceTransformWithLabels((~known).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(known)
        table = np.zeros(int(lbl.max()) + 1, np.uint8)
        table[lbl[ys, xs]] = cm_bg[ys, xs]
        out = np.where(fill, table[lbl], out).astype(np.uint8)
    out = np.where(keep, cm, out).astype(np.uint8)
    stats = {"mode": mode, "residual_px": int(residual.sum()), "kept_px": int(keep.sum()),
             "dropped_px": int(residual.sum() - keep.sum()), "kept_components": comps}
    return out, keep, stats


# ---------------------------------------------------------------- proxy metrics (no gold)
def agreement_miou(a: np.ndarray, b: np.ndarray) -> float:
    """mIoU between two label maps over classes present in either (policy-vs-policy agreement, NOT accuracy)."""
    ious = []
    for c in np.union1d(np.unique(a), np.unique(b)):
        x, y = a == c, b == c
        u = (x | y).sum()
        if u:
            ious.append((x & y).sum() / u)
    return float(np.mean(ious))


def boundary_stats(cm: np.ndarray) -> dict:
    """Edge smoothness proxies: 8-connected components (total / <0.1% of image) and label-boundary pixel fraction."""
    import cv2

    H, W = cm.shape
    n_tot = n_small = 0
    for c in np.unique(cm):
        n, _, st, _ = cv2.connectedComponentsWithStats((cm == c).astype(np.uint8), connectivity=8)
        n_tot += n - 1
        n_small += int((st[1:, cv2.CC_STAT_AREA] < 0.001 * H * W).sum())
    edge = (cm[:, 1:] != cm[:, :-1]).sum() + (cm[1:, :] != cm[:-1, :]).sum()
    return {"components": n_tot, "small_components": n_small, "edge_frac": round(float(edge) / (H * W), 4)}


def class_fractions(cm: np.ndarray) -> Dict[int, float]:
    cnt = np.bincount(cm.ravel(), minlength=19)[:19]
    return {i: round(float(v) / cm.size, 4) for i, v in enumerate(cnt) if v}


# ---------------------------------------------------------------- bench
IMGS = ("drive_001", "drive_002", "drive_010", "drive_030", "drive_057")
TUNE_IMGS = IMGS[:4]  # drive_057 is EVAL split: reported only, never used to choose policy/threshold


def _bench(out_dir: Path, with_yolo: bool) -> dict:
    import json

    import cv2

    root = Path(__file__).resolve().parents[2] / "data" / "real" / "day02_traffic" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    names = ["ls640", "ls1024", "ls1536", "ls1024_dark", "tile512"]  # tile768 measured once on drive_001: 59.7s, too slow
    cms: Dict[str, Dict[str, np.ndarray]] = {}
    res: Dict[str, Any] = {"per_image": {}}
    for im in TUNE_IMGS:
        img = cv2.imread(str(root / f"{im}.jpg"))
        cms[im], rows = {}, {}
        for p in names:
            cm, inf = semantic_labels(img, {"preset": p})
            cms[im][p] = cm
            rows[p] = {"s": inf["s"], "in_hw": inf["in_hw"], "enh": inf["enhance"]["applied"],
                       "mean_gray": inf["enhance"]["mean_gray"], **boundary_stats(cm),
                       "frac": class_fractions(cm)}
        cm, inf = semantic_labels(img, {"ensemble": ["ls640", "ls1024", "ls1536"]})
        cms[im]["ens3"] = cm
        rows["ens3"] = {"s": inf["s"], **boundary_stats(cm), "frac": class_fractions(cm)}
        ref = cms[im]["ls1024"]
        for p, r in rows.items():
            r["agree_vs_ls1024"] = round(agreement_miou(cms[im][p], ref), 4)
            r["agree_vs_ens3"] = round(agreement_miou(cms[im][p], cms[im]["ens3"]), 4)
        res["per_image"][im] = rows
        (out_dir / f"{im}_cm_ls1024.png").write_bytes(cv2.imencode(".png", ref)[1].tobytes())
    if with_yolo:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from tools.t14_seg_local import instance_masks  # read-only import
        from ultralytics import YOLO
        import torch

        torch.set_num_threads(2)
        paths = __import__("cvat_tools.core", fromlist=["paths"]).paths
        yolo = YOLO(str(Path(paths.COPILOT) / "yolo26s-seg.pt"))
        res["residual"] = {}
        for im in TUNE_IMGS:
            img = cv2.imread(str(root / f"{im}.jpg"))
            inst, _ = instance_masks(yolo, img, 0.25, 1280)
            union = np.zeros(img.shape[:2], bool)
            for _, _, mk in inst:
                union |= mk
            r_im: Dict[str, Any] = {"n_inst": len(inst), "inst_px": int(union.sum())}
            for p in ("ls640", "ls1024", "ls1536", "ens3", "tile512"):
                cm = cms[im][p]
                rs = thing_residual(cm, union)
                tot_thing = int(np.isin(cm, THING_IDS).sum())
                ncomp = cv2.connectedComponents(rs.astype(np.uint8), connectivity=8)[0] - 1
                # stability: residual px that remain thing in every other policy
                others = [cms[im][q] for q in ("ls640", "ls1024", "ls1536") if q != p]
                stable = rs.copy()
                for o in others:
                    stable &= np.isin(o, THING_IDS)
                lab_h, kept, st = apply_residual(cm, rs, "hybrid")
                _, _, st_f = apply_residual(cm, rs, "flag")
                _, _, st_d = apply_residual(cm, rs, "drop")
                r_im[p] = {"thing_px": tot_thing, "resid_px": int(rs.sum()), "resid_frac_of_thing": round(float(rs.sum()) / max(1, tot_thing), 3),
                           "resid_components": int(ncomp), "resid_stable_px": int(stable.sum()),
                           "stable_frac": round(float(stable.sum()) / max(1, int(rs.sum())), 3),
                           "hybrid": st, "flag": st_f, "drop": st_d,
                           "hybrid_gap255": round(float((lab_h == UNLABELED).mean()), 5)}
            res["residual"][im] = r_im
    (out_dir / "bench.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def _sweep(out_dir: Path) -> dict:
    """Gap=0 rule: sweep keep-threshold (max softmax prob) for residual thing pixels. Proxy 'stable' = kept px that
    ls640 and ls1536 also call thing."""
    import json

    import cv2

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.t14_seg_local import instance_masks
    from ultralytics import YOLO
    import torch

    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[2] / "data" / "real" / "day02_traffic" / "images"
    paths = __import__("cvat_tools.core", fromlist=["paths"]).paths
    yolo = YOLO(str(Path(paths.COPILOT) / "yolo26s-seg.pt"))
    res: Dict[str, Any] = {}
    for im in TUNE_IMGS:
        img = cv2.imread(str(root / f"{im}.jpg"))
        cm, inf = semantic_labels(img, {"preset": "ls1024_dark", "with_conf": True})
        o1, _ = semantic_labels(img, {"preset": "ls640"})
        o2, _ = semantic_labels(img, {"preset": "ls1536"})
        inst, _ = instance_masks(yolo, img, 0.25, 1280)
        union = np.zeros(img.shape[:2], bool)
        for _, _, mk in inst:
            union |= mk
        rs = thing_residual(cm, union)
        agree = np.isin(o1, THING_IDS) & np.isin(o2, THING_IDS)
        r: Dict[str, Any] = {"resid_px": int(rs.sum()), "conf_mean": round(float(inf["conf"][rs].mean()) / 255, 3) if rs.any() else None,
                             "margin_mean": round(float(inf["margin"][rs].mean()) / 255, 3) if rs.any() else None, "sweep": {}}
        for thr in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95):
            lab, kept, st = apply_residual(cm, rs, "hybrid", conf=inf["conf"], min_conf=int(thr * 255))
            st["gap255"] = int((lab == UNLABELED).sum())
            st["kept_stable_frac"] = round(float((kept & agree).sum()) / max(1, int(kept.sum())), 3)
            st["dropped_stable_frac"] = round(float((rs & ~kept & agree).sum()) / max(1, int((rs & ~kept).sum())), 3)
            r["sweep"][str(thr)] = st
        for mode in ("drop", "flag"):
            lab, _, st = apply_residual(cm, rs, mode)
            st["gap255"] = int((lab == UNLABELED).sum())
            r[mode] = st
        res[im] = r
    (out_dir / "sweep.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def _selftest() -> None:
    cm = np.full((40, 40), 0, np.uint8)
    cm[:10] = 10
    cm[20:30, 10:30] = 13  # car 200 px
    cm[2:4, 2:4] = 11  # 4 px person speck
    inst = np.zeros((40, 40), bool)
    inst[20:30, 10:20] = True  # instance covers left half of car
    rs = thing_residual(cm, inst)
    assert rs.sum() == 100 + 4
    lab, kept, st = apply_residual(cm, rs, "hybrid", min_area=50, min_extent=3)
    assert kept.sum() == 100 and st["dropped_px"] == 4 and lab[3, 3] == 10 and (lab == UNLABELED).sum() == 0
    _, _, st = apply_residual(cm, rs, "drop")
    assert st["kept_px"] == 0
    assert agreement_miou(cm, cm) == 1.0
    print("selftest ok")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="semantic layer bench (5 fixed images, <=2 threads)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--yolo", action="store_true", help="also measure thing residual vs YOLO26s-seg@1280")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--out", default="out/opt/semantic")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    if a.bench:
        import json

        print(json.dumps(_bench(Path(a.out), a.yolo), indent=1)[:200000])
    if a.sweep:
        import json

        Path(a.out).mkdir(parents=True, exist_ok=True)
        print(json.dumps(_sweep(Path(a.out))))
