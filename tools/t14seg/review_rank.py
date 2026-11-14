"""Review ranking + auto-drop split for the T14 segmentation draft (PROXY scores, no gold).

S = sum_f w_f * f   over  U, D, G, C, F, N  (every feature in [0,1], higher = needs more attention).
The weights are a HYPOTHESIS (see DEFAULT_WEIGHTS); nothing here claims ranking accuracy, only
stability / sensitivity (ablation). Dropped polygons leave holes: the caller MUST re-assign their
pixels to a neighbour label (rule "no gaps"); `split_drop_review` reports the area it would remove.

Features (all NaN-safe; NaN means "signal not available", handled by `nan_policy`):
  U  uncertainty   = clip((1-conf)/(1-conf_floor),0,1)       conf = YOLO conf of the instance. NaN w/o conf.
  D  disagreement  = 1 - share of polygon pixels where an alternative label map (e.g. raw SegFormer, or a
                     second config) gives the same class. NaN w/o alt map.
  G  geometry      = mean of the available sub-features (equal weights, hypothesis):
                       small    = clip((ln 1000 - ln area)/(ln 1000 - ln 16),0,1)
                       edge     = share of vertices within 1px of the image border (things only, else 0)
                       jag      = clip((perimeter/perimeter_after_approxPolyDP(1.5) - 1)/0.5,0,1)
                       solidity = clip((0.9 - area/hull_area)/0.6,0,1)
  C  class confusion = 1 - own-class share among alt-map pixels that fall in the same confusion family
                     (FAMILY below); 0 if no alt pixel is in the family. NaN w/o alt map.
  F  semantic_only = 1 for dynamic thing (person..bicycle) from SegFormer only (shape_source=="semantic"),
                     0.5 for pole/light/sign semantic-only (YOLO cannot make those), else 0.
  N  image-level   = max(night, deficit); night = clip((90-mean_gray)/60); deficit = clip(1 - n_inst/median(n_inst))
                     (needs >=3 images). Constant per image -> only matters in the GLOBAL ranking.

Tie-break: (-S rounded to 1e-12, -area, id). Time budget: cost_sec = base + per_px * perimeter (ASSUMED
constants, unmeasured; tune with real review timings).

CLI (needs CVAT_TOOLS_SRC; runs SegFormer+YOLO on the given images only, 2 threads):
    python -m tools.t14seg.review_rank --images drive_001,drive_002 --out out/opt/review_rank
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

FEATURES: tuple[str, ...] = ("U", "D", "G", "C", "F", "N")
DEFAULT_WEIGHTS: dict[str, float] = {"U": 0.35, "D": 0.25, "G": 0.15, "C": 0.10, "F": 0.10, "N": 0.05}
DYNAMIC = frozenset(range(11, 19))   # Cityscapes ids: person..bicycle
STATIC_THING = frozenset({5, 6, 7})  # pole, traffic light, traffic sign
THING = DYNAMIC | STATIC_THING
FAMILY: dict[int, tuple[int, ...]] = {}
for _fam in ((11, 12), (13, 14, 15, 16), (17, 18), (5, 6, 7), (0, 1), (2, 3, 4), (8, 9), (10,)):
    for _c in _fam:
        FAMILY[_c] = _fam
ROOT = Path(__file__).resolve().parents[2]
REAL = ROOT / "data" / "real" / "day02_traffic"


# ----------------------------------------------------------------------------- feature functions
def _clip01(x: float) -> float:
    """Clip to [0,1]; NaN/inf-safe (NaN stays NaN, +-inf clipped)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return float("nan")
    return float(min(1.0, max(0.0, x)))


def feat_u(conf: float | None, conf_floor: float = 0.25) -> float:
    """Uncertainty from a detector confidence; conf==conf_floor -> 1, conf==1 -> 0, None/NaN -> NaN."""
    if conf is None or not np.isfinite(conf):
        return float("nan")
    return _clip01((1.0 - float(conf)) / max(1e-9, 1.0 - conf_floor))


def feat_f(cs: int, shape_source: str) -> float:
    """Semantic-only flag (graded by class kind). cs = Cityscapes id."""
    if shape_source != "semantic":
        return 0.0
    if cs in DYNAMIC:
        return 1.0
    if cs in STATIC_THING:
        return 0.5
    return 0.0


def feat_g(pts: Sequence[float], area: float, W: int, H: int, cs: int) -> dict[str, float]:
    """Geometry sub-features (each in [0,1] or NaN) and their mean 'G' (NaN if none available)."""
    import cv2

    nan = float("nan")
    out = {"small": nan, "edge": nan, "jag": nan, "solidity": nan, "G": nan}
    p = np.asarray(pts, np.float32).reshape(-1, 2)
    if len(p) < 3 or not np.isfinite(area) or area <= 0:
        return out
    out["small"] = _clip01((math.log(1000.0) - math.log(area)) / (math.log(1000.0) - math.log(16.0)))
    if cs in THING:
        near = (p[:, 0] <= 1) | (p[:, 0] >= W - 2) | (p[:, 1] <= 1) | (p[:, 1] >= H - 2)
        out["edge"] = float(near.mean())
    else:
        out["edge"] = 0.0
    c = p.reshape(-1, 1, 2)
    per = float(cv2.arcLength(c, True))
    sm = float(cv2.arcLength(cv2.approxPolyDP(c, 1.5, True), True))
    if sm > 0:
        out["jag"] = _clip01((per / sm - 1.0) / 0.5)
    hull_a = float(cv2.contourArea(cv2.convexHull(c)))
    if hull_a > 0:
        out["solidity"] = _clip01((0.9 - area / hull_a) / 0.6)
    vals = [v for k, v in out.items() if k != "G" and np.isfinite(v)]
    out["G"] = float(np.mean(vals)) if vals else nan
    return out


def feat_dc(pts: Sequence[float], cs: int, alt: np.ndarray) -> tuple[float, float]:
    """(D, C) of one polygon against an alternative HxW class map. Rasterised on its bbox crop."""
    import cv2

    nan = float("nan")
    p = np.round(np.asarray(pts, np.float32).reshape(-1, 2)).astype(np.int32)
    H, W = alt.shape[:2]
    x0, y0 = max(0, int(p[:, 0].min())), max(0, int(p[:, 1].min()))
    x1, y1 = min(W, int(p[:, 0].max()) + 1), min(H, int(p[:, 1].max()) + 1)
    if x1 <= x0 or y1 <= y0:
        return nan, nan
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(m, [p - np.array([x0, y0], np.int32)], 1)
    vals = alt[y0:y1, x0:x1][m.astype(bool)]
    if vals.size == 0:
        return nan, nan
    d = _clip01(1.0 - float((vals == cs).mean()))
    fam = FAMILY.get(cs, (cs,))
    inf = vals[np.isin(vals, fam)]
    c = 0.0 if inf.size == 0 else _clip01(1.0 - float((inf == cs).mean()))
    return d, c


def feat_n_images(n_inst: Mapping[str, int], mean_gray: Mapping[str, float] | None = None) -> dict[str, float]:
    """Image-level N per image id. See module docstring. NaN parts are skipped; all-NaN -> NaN."""
    ids = list(n_inst)
    med = float(np.median([n_inst[i] for i in ids])) if len(ids) >= 3 else float("nan")
    out: dict[str, float] = {}
    for i in ids:
        parts = []
        if mean_gray and i in mean_gray and np.isfinite(mean_gray[i]):
            parts.append(_clip01((90.0 - mean_gray[i]) / 60.0))
        if np.isfinite(med) and med > 0:
            parts.append(_clip01(1.0 - n_inst[i] / med))
        out[i] = max(parts) if parts else float("nan")
    return out


# ----------------------------------------------------------------------------- scoring
def score(df, weights: Mapping[str, float] | None = None, nan_policy: str = "neutral") -> np.ndarray:
    """S per row from feature columns. nan_policy: 'neutral' (NaN->0.5), 'zero' (NaN->0),
    'renorm' (drop NaN feature, renormalise remaining weights; a row with all NaN -> 0.5)."""
    w = dict(DEFAULT_WEIGHTS if weights is None else weights)
    X = np.column_stack([df[f].to_numpy(float) if f in df else np.full(len(df), np.nan) for f in FEATURES])
    X = np.where(np.isinf(X), np.clip(X, 0, 1), X)
    X = np.where(np.isfinite(X), np.clip(X, 0.0, 1.0), np.nan)
    W_ = np.array([max(0.0, float(w.get(f, 0.0))) for f in FEATURES])
    if nan_policy == "renorm":
        ok = np.isfinite(X)
        den = (ok * W_).sum(1)
        num = (np.where(ok, X, 0.0) * W_).sum(1)
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.5)
    fill = 0.5 if nan_policy == "neutral" else 0.0
    return np.where(np.isfinite(X), X, fill) @ W_


def order(df, S: np.ndarray | None = None):
    """Deterministic ordering indices: (-S rounded 1e-12, -area, id)."""
    S = df["S"].to_numpy(float) if S is None else S
    keys = (df["id"].to_numpy(), -df["area"].to_numpy(float), -np.round(S, 12))  # lexsort: last key primary
    return np.lexsort(keys)


def rank_polygons(coco: Mapping[str, Any], *, conf: Mapping[int, float] | None = None,
                  alt_labels: Mapping[str, np.ndarray] | None = None,
                  mean_gray: Mapping[str, float] | None = None,
                  image_ids: Sequence[str] | None = None,
                  weights: Mapping[str, float] | None = None, nan_policy: str = "neutral",
                  conf_floor: float = 0.25):
    """Rank polygons of a COCO draft (as written by t14_seg_local). Returns a pandas DataFrame sorted by
    review priority with columns id,image,cs,shape_source,area,perim,nverts,U,D,G,C,F,N,S,rank,rank_in_image.

    conf: {ann_id: yolo_conf}; alt_labels: {image_stem: HxW class map in Cityscapes ids};
    image_ids: restrict to these image stems (N is computed over the restricted set)."""
    import cv2
    import pandas as pd

    imgs = {im["id"]: im for im in coco["images"]}
    stem = {i: Path(im["file_name"]).stem for i, im in imgs.items()}
    keep = set(image_ids) if image_ids is not None else None
    anns = [a for a in coco["annotations"] if keep is None or stem[a["image_id"]] in keep]
    n_inst: dict[str, int] = {stem[i]: 0 for i in imgs if keep is None or stem[i] in keep}
    for a in anns:
        if a["attributes"].get("shape_source") == "instance" and a["category_id"] - 1 in DYNAMIC:
            n_inst[stem[a["image_id"]]] += 1
    nimg = feat_n_images(n_inst, mean_gray)
    rows = []
    for a in anns:
        im = imgs[a["image_id"]]
        s, cs = stem[a["image_id"]], a["category_id"] - 1
        pts = a["segmentation"][0]
        area = float(a.get("area", 0.0))
        g = feat_g(pts, area, im["width"], im["height"], cs)
        d = c = float("nan")
        if alt_labels is not None and s in alt_labels:
            d, c = feat_dc(pts, cs, alt_labels[s])
        cf = None if conf is None else conf.get(a["id"])
        per = float(cv2.arcLength(np.asarray(pts, np.float32).reshape(-1, 1, 2), True))
        src = a["attributes"].get("shape_source", "")
        rows.append({"id": a["id"], "image": s, "cs": cs, "shape_source": src, "area": area, "perim": per,
                     "nverts": len(pts) // 2, "U": feat_u(cf, conf_floor), "D": d, "G": g["G"], "C": c,
                     "F": feat_f(cs, src), "N": nimg[s]})
    df = pd.DataFrame(rows, columns=["id", "image", "cs", "shape_source", "area", "perim", "nverts", *FEATURES])
    df["S"] = score(df, weights, nan_policy) if len(df) else np.array([])
    df = df.iloc[order(df)].reset_index(drop=True) if len(df) else df
    df["rank"] = np.arange(1, len(df) + 1)
    df["rank_in_image"] = df.groupby("image").cumcount() + 1 if len(df) else []
    return df


def rank_images(df, *, top_k: int = 10, n_inst_min: int = 3, n_thresh: float = 0.5):
    """Image-level review. image_S = 0.5*N + 0.5*mean(top_k S) (hypothesis). image_review flag when the image
    has < n_inst_min dynamic instances OR N >= n_thresh (night / abnormally few instances): those images need
    a full-image pass instead of per-polygon triage. Sorted (-image_S, image)."""
    import pandas as pd

    rec = []
    for im, g in df.groupby("image"):
        n = float(g["N"].iloc[0]) if g["N"].notna().any() else float("nan")
        ninst = int(((g["shape_source"] == "instance") & g["cs"].isin(DYNAMIC)).sum())
        topm = float(g["S"].nlargest(top_k).mean())
        sem_dyn = int(((g["shape_source"] == "semantic") & g["cs"].isin(DYNAMIC)).sum())
        s_img = 0.5 * (0.5 if math.isnan(n) else n) + 0.5 * topm
        rec.append({"image": im, "n_polys": len(g), "n_dyn_instances": ninst, "n_dyn_semantic_only": sem_dyn,
                    "N": n, "topk_mean_S": topm, "image_S": s_img,
                    "image_review": bool(ninst < n_inst_min or (not math.isnan(n) and n >= n_thresh))})
    out = pd.DataFrame(rec)
    return out.sort_values(["image_S", "image"], ascending=[False, True]).reset_index(drop=True)


# ----------------------------------------------------------------------------- budget + drop
def select_budget(df, budget_sec: float, *, base_sec: float = 3.0, per_px: float = 0.02,
                  mode: str = "prefix"):
    """Top-K by time budget. cost = base_sec + per_px*perimeter (ASSUMED). mode 'prefix': take the rank order
    until the next item no longer fits (stop); 'density': order by S/cost, greedy fill. Returns subframe + cost_sec."""
    d = df.copy()
    d["cost_sec"] = base_sec + per_px * d["perim"].to_numpy(float)
    if mode == "density":
        idx = np.lexsort((d["id"].to_numpy(), -d["area"].to_numpy(float), -np.round(d["S"].to_numpy(float) / d["cost_sec"].to_numpy(), 12)))
        d = d.iloc[idx]
    take, tot = [], 0.0
    for i, c in zip(d.index, d["cost_sec"]):
        if tot + c > budget_sec:
            if mode == "prefix":
                break
            continue
        take.append(i)
        tot += c
    return d.loc[take]


def split_drop_review(df, tau_s: float, *, require_f: bool = False, max_area_frac: float = 0.05,
                      image_area: Mapping[str, float] | None = None):
    """Split into (auto_drop, review) by S >= tau_s (and F>0 if require_f). Per image, drops are taken in rank
    order and stop once the dropped polygon area would exceed max_area_frac of the image (hole risk; the
    pipeline must refill dropped pixels from neighbours - no gaps allowed). Protected: none of the stuff
    classes is exempt, hence the area cap. image_area default 640*640."""
    cand = (df["S"] >= tau_s) & ((df["F"] > 0) if require_f else True)
    drop_idx = []
    for im, g in df[cand].groupby("image", sort=False):
        tot = (image_area or {}).get(im, 640.0 * 640.0)
        used = 0.0
        for i, a in zip(g.index, g["area"]):  # g is already in rank order
            if used + a > max_area_frac * tot:
                continue
            used += a
            drop_idx.append(i)
    m = df.index.isin(drop_idx)
    return df[m].copy(), df[~m].copy()


def drop_sweep(df, taus: Sequence[float], *, require_f: bool = False, max_area_frac: float = 0.05):
    """Counts per threshold (PROXY): n dropped, share of polygons, dropped area share, dropped by shape_source."""
    import pandas as pd

    rows = []
    tot_area = float(df["area"].sum())
    for t in taus:
        dr, _ = split_drop_review(df, t, require_f=require_f, max_area_frac=max_area_frac)
        rows.append({"tau_s": t, "require_f": require_f, "n_drop": len(dr), "share_polys": len(dr) / max(1, len(df)),
                     "drop_area_share": float(dr["area"].sum()) / tot_area if tot_area else float("nan"),
                     "drop_semantic": int((dr["shape_source"] == "semantic").sum()),
                     "drop_instance": int((dr["shape_source"] == "instance").sum())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- ablation
def _kendall(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import kendalltau

    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(kendalltau(a, b).statistic)


def overlap_at_k(base_rank_ids: Sequence, other_rank_ids: Sequence, k: int) -> float:
    k = min(k, len(base_rank_ids))
    return float("nan") if k == 0 else len(set(base_rank_ids[:k]) & set(other_rank_ids[:k])) / k


def ablation(df, *, weights: Mapping[str, float] | None = None, nan_policy: str = "neutral",
             ks: Sequence[int] = (10, 30), per_image: bool = True):
    """Drop each feature (weight 0), compare with the full S: Kendall tau-b on S and overlap@K of top-K sets.
    per_image=True averages over images (N is constant inside an image so its ablation is trivially tau=1 there;
    use per_image=False for the global effect). Only features that are not entirely NaN are meaningful."""
    import pandas as pd

    w0 = dict(DEFAULT_WEIGHTS if weights is None else weights)
    groups = [g for _, g in df.groupby("image")] if per_image else [df]
    rows = []
    for f in FEATURES:
        wa = {**w0, f: 0.0}
        taus, ov = [], {k: [] for k in ks}
        for g in groups:
            g = g.copy()
            s0, s1 = score(g, w0, nan_policy), score(g, wa, nan_policy)
            taus.append(_kendall(s0, s1))
            g["_S0"], g["_S1"] = s0, s1
            i0 = g.iloc[np.lexsort((g["id"].to_numpy(), -g["area"].to_numpy(float), -np.round(s0, 12)))]["id"].tolist()
            i1 = g.iloc[np.lexsort((g["id"].to_numpy(), -g["area"].to_numpy(float), -np.round(s1, 12)))]["id"].tolist()
            for k in ks:
                ov[k].append(overlap_at_k(i0, i1, k))
        row = {"dropped": f, "frac_nan": float(df[f].isna().mean()) if f in df else 1.0,
               "kendall_tau": float(np.nanmean(taus)) if np.isfinite(taus).any() else float("nan")}
        for k in ks:
            row[f"overlap@{k}"] = float(np.nanmean(ov[k])) if np.isfinite(ov[k]).any() else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def weight_perturbation(df, *, n: int = 20, rel: float = 0.3, seed: int = 0, k: int = 30,
                        nan_policy: str = "neutral"):
    """Sensitivity: multiply each weight by U(1-rel,1+rel), n draws; mean per-image Kendall tau and overlap@k
    vs the default weights. Stability only, not accuracy."""
    rng = np.random.RandomState(seed)
    res_t, res_o = [], []
    for _ in range(n):
        w = {f: DEFAULT_WEIGHTS[f] * rng.uniform(1 - rel, 1 + rel) for f in FEATURES}
        for _, g in df.groupby("image"):
            s0, s1 = score(g, None, nan_policy), score(g, w, nan_policy)
            res_t.append(_kendall(s0, s1))
            o0 = g.iloc[np.lexsort((g["id"].to_numpy(), -g["area"].to_numpy(float), -np.round(s0, 12)))]["id"].tolist()
            o1 = g.iloc[np.lexsort((g["id"].to_numpy(), -g["area"].to_numpy(float), -np.round(s1, 12)))]["id"].tolist()
            res_o.append(overlap_at_k(o0, o1, k))
    return {"n_draws": n, "rel": rel, f"mean_kendall_tau": float(np.nanmean(res_t)),
            "min_kendall_tau": float(np.nanmin(res_t)), f"mean_overlap@{k}": float(np.nanmean(res_o)),
            f"min_overlap@{k}": float(np.nanmin(res_o))}


# ----------------------------------------------------------------------------- signals (CLI only, heavy)
def collect_signals(coco: Mapping[str, Any], stems: Sequence[str], out_dir: Path, seg_dir: Path,
                    threads: int = 2, yolo_name: str = "yolo26s-seg.pt", imgsz: int = 1280, yconf: float = 0.25):
    """Re-run SegFormer (alt map for D/C) and YOLO (conf per instance polygon, matched by mask IoU with
    *_inst.png) on the given images. Cached in out_dir/signals. Returns (conf {ann_id:conf}, alt {stem:cm}, gray)."""
    import cv2

    cache = out_dir / "signals"
    cache.mkdir(parents=True, exist_ok=True)
    stem_of = {im["id"]: Path(im["file_name"]).stem for im in coco["images"]}
    todo = [s for s in stems if not (cache / f"{s}.npz").exists()]
    gray = {}
    if todo:
        src = os.environ.get("CVAT_TOOLS_SRC")
        if not src or not Path(src).is_dir():
            sys.exit("Set CVAT_TOOLS_SRC to <learning>/cvat/src")
        sys.path.insert(0, src)
        import torch
        torch.set_num_threads(threads)
        from cvat_tools.core import paths
        from cvat_tools.segmentation.onnx_seg import OnnxSeg
        from ultralytics import YOLO
        seg = OnnxSeg("b2", threads=threads)
        yolo = YOLO(str(Path(paths.COPILOT) / yolo_name))
        sys.path.insert(0, str(ROOT))
        from tools.t14_seg_local import COCO_TO_CS
        for s in todo:
            img = cv2.imread(str(REAL / "images" / f"{s}.jpg"))
            H, W = img.shape[:2]
            cm, _ = seg.class_map(img)
            imap = cv2.imread(str(seg_dir / "labels" / f"{s}_inst.png"), cv2.IMREAD_UNCHANGED)
            r = yolo.predict(img, conf=yconf, imgsz=imgsz, device="cpu", verbose=False)[0]
            ks, cfs = [], []
            if r.masks is not None:
                for m, c, sc in zip(r.masks.data.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()):
                    if c not in COCO_TO_CS:
                        continue
                    mk = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                    ids, cnt = np.unique(imap[mk], return_counts=True)
                    best, bi = 0.0, 0
                    for k, n in zip(ids, cnt):
                        if k == 0:
                            continue
                        iou = n / float(mk.sum() + (imap == k).sum() - n)
                        if iou > best:
                            best, bi = iou, int(k)
                    if best > 0.5:
                        ks.append(bi)
                        cfs.append(float(sc))
            np.savez_compressed(cache / f"{s}.npz", cm=cm.astype(np.uint8), k=np.array(ks), conf=np.array(cfs),
                                gray=float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()))
    conf, alt = {}, {}
    for s in stems:
        z = np.load(cache / f"{s}.npz")
        alt[s] = z["cm"]
        gray[s] = float(z["gray"])
        imap = cv2.imread(str(seg_dir / "labels" / f"{s}_inst.png"), cv2.IMREAD_UNCHANGED)
        kconf = dict(zip(z["k"].tolist(), z["conf"].tolist()))
        for a in coco["annotations"]:
            if stem_of[a["image_id"]] != s or a["attributes"].get("shape_source") != "instance":
                continue
            p = np.round(np.asarray(a["segmentation"][0], np.float32).reshape(-1, 2)).astype(np.int32)
            m = np.zeros(imap.shape, np.uint8)
            cv2.fillPoly(m, [p], 1)
            v = imap[m.astype(bool)]
            v = v[v > 0]
            if v.size:
                k = int(np.bincount(v).argmax())
                if k in kconf:
                    conf[a["id"]] = kconf[k]
    return conf, alt, gray


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", default=str(ROOT / "out" / "seg_local"))
    ap.add_argument("--images", default="drive_001,drive_002,drive_010,drive_030")
    ap.add_argument("--out", default=str(ROOT / "out" / "opt" / "review_rank"))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--budget-min", type=float, default=10.0)
    a = ap.parse_args()
    seg, out = Path(a.seg), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stems = a.images.split(",")  # dev only: eval images (e.g. drive_057) must not be used to tune weights/thresholds
    coco = json.load(open(seg / "polys" / "coco_segmentation_draft.json", encoding="utf-8"))
    conf, alt, gray = collect_signals(coco, stems, out, seg, a.threads)
    df = rank_polygons(coco, conf=conf, alt_labels=alt, mean_gray=gray, image_ids=stems)
    df.to_csv(out / "rank_polygons.csv", index=False)
    img = rank_images(df)
    img.to_csv(out / "rank_images.csv", index=False)
    ab = ablation(df)
    ab.to_csv(out / "ablation_per_image.csv", index=False)
    abg = ablation(df, per_image=False)
    abg.to_csv(out / "ablation_global.csv", index=False)
    sw = drop_sweep(df, [0.3, 0.4, 0.5, 0.6, 0.7])
    swf = drop_sweep(df, [0.3, 0.4, 0.5, 0.6, 0.7], require_f=True)
    sw.to_csv(out / "drop_sweep.csv", index=False)
    swf.to_csv(out / "drop_sweep_require_f.csv", index=False)
    bud = select_budget(df, a.budget_min * 60)
    summary = {"n_polys": len(df), "frac_nan": {f: float(df[f].isna().mean()) for f in FEATURES},
               "S_quantiles": df["S"].quantile([0, .25, .5, .75, .9, 1]).round(4).tolist(),
               "budget_min": a.budget_min, "budget_items": len(bud),
               "perturbation": weight_perturbation(df)}
    json.dump(summary, open(out / "summary.json", "w", encoding="utf-8"), indent=1)
    import pandas as pd
    pd.set_option("display.width", 200)
    print(img.to_string(index=False), "\n", ab.to_string(index=False), "\n", abg.to_string(index=False), "\n",
          sw.to_string(index=False), "\n", swf.to_string(index=False), "\n", json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
