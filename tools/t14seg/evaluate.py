"""Evaluation harness for the T14 segmentation draft. Three modes, never mixed up:

  gold   : pred vs hand-made gold (data/real/day02_traffic/gold/). mIoU/class, instance F1@0.5/0.75,
           boundary F-score, open-pixel rate, bootstrap CI BY IMAGE (seed 42), diff vs baseline.
  proxy  : no gold. Agreement between 2 configs + gap%/polygons/instances/time. Reports are stamped
           'PROXY, khong phai do chinh xac'.
  iaa    : inter-annotator agreement gold A vs cross-check B (human-human ceiling).

Formats (pred / gold / baseline), either:
  * directory with `<stem>.png` single-channel label maps (class id 0..18, 255 = unlabeled/gap),
    optional `<stem>_inst.png` (uint16 instance ids, 0 = none);
  * COCO json with polygons (category name matched to NAMES, else category_id-1; later annotations paint on top).
Refuses (EvalError) on: image-name mismatch, size mismatch, gold flagged as tuned-on / not held-out.

    python -m tools.t14seg.evaluate gold  --pred P --gold G --split heldout [--baseline B] [--tuned-stems f.txt]
    python -m tools.t14seg.evaluate proxy --pred-a A --pred-b B [--run-info a.json]
    python -m tools.t14seg.evaluate iaa   --gold-a A --gold-b B
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

GAP = 255
NCLS = 19
NAMES = {0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence", 5: "pole", 6: "traffic light",
         7: "traffic sign", 8: "vegetation", 9: "terrain", 10: "sky", 11: "pedestrian", 12: "rider", 13: "car",
         14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle"}
PROXY_LABEL = "PROXY, khong phai do chinh xac"
SMALL_N = 30
BOOT_N = 1000
SEED = 42


class EvalError(RuntimeError):
    """Refusal to evaluate (mismatched names/sizes, contaminated gold)."""


@dataclass
class Sample:
    stem: str
    labels: np.ndarray            # (H,W) uint8, GAP = unlabeled
    inst: np.ndarray              # (H,W) int32, 0 = no instance
    n_polys: int = 0
    n_dropped: int = 0            # polygons dropped by the pipeline because blurry/uncertain (sidecar dropped.json)


# ---------------------------------------------------------------- loading
def _stem(name: str) -> str:
    return Path(name).stem


def render_coco(coco: Mapping, names: Mapping[int, str] = NAMES) -> dict[str, Sample]:
    """Render COCO polygons to label + instance maps (annotation order = paint order)."""
    import cv2
    inv = {v.lower(): k for k, v in names.items()}
    cat = {}
    for c in coco.get("categories", []):
        cat[c["id"]] = inv.get(str(c.get("name", "")).lower(), c["id"] - 1)
    out: dict[str, Sample] = {}
    by_img: dict[int, list] = {}
    for a in coco.get("annotations", []):
        by_img.setdefault(a["image_id"], []).append(a)
    for im in coco["images"]:
        h, w = int(im["height"]), int(im["width"])
        lab = np.full((h, w), GAP, np.uint8)
        inst = np.zeros((h, w), np.int32)
        n = 0
        for k, a in enumerate(by_img.get(im["id"], []), 1):
            cid = cat.get(a["category_id"], a["category_id"] - 1)
            for poly in a.get("segmentation", []):
                pts = np.round(np.asarray(poly, np.float64).reshape(-1, 2)).astype(np.int32)
                if len(pts) < 3:
                    continue
                cv2.fillPoly(lab, [pts], int(cid))
                cv2.fillPoly(inst, [pts], k)
                n += 1
        out[_stem(im["file_name"])] = Sample(_stem(im["file_name"]), lab, inst, n)
    return out


def _apply_dropped(out: dict[str, Sample], side: Path) -> dict[str, Sample]:
    """Optional sidecar dropped.json {stem: n_polygons_dropped_as_blurry} next to the pred set."""
    if side.is_file():
        d = json.loads(side.read_text(encoding="utf-8"))
        for k, v in d.items():
            if _stem(k) in out:
                out[_stem(k)].n_dropped = int(v)
    return out


def load_set(path: str | Path) -> dict[str, Sample]:
    import cv2
    p = Path(path)
    if p.is_file() and p.suffix == ".json":
        return _apply_dropped(render_coco(json.loads(p.read_text(encoding="utf-8"))), p.with_name("dropped.json"))
    if not p.is_dir():
        raise EvalError(f"not found: {p}")
    out = {}
    for f in sorted(p.glob("*.png")):
        if f.stem.endswith("_inst"):
            continue
        lab = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if lab is None or lab.ndim != 2:
            raise EvalError(f"{f.name}: label map must be a single-channel PNG")
        ip = f.with_name(f.stem + "_inst.png")
        inst = cv2.imread(str(ip), cv2.IMREAD_UNCHANGED) if ip.exists() else None
        inst = np.zeros(lab.shape, np.int32) if inst is None else inst.astype(np.int32)
        n_inst = int(len(np.unique(inst)) - (1 if (inst == 0).any() else 0))
        out[f.stem] = Sample(f.stem, lab.astype(np.uint8), inst, n_inst)
    if not out:
        raise EvalError(f"no label PNGs in {p}")
    return _apply_dropped(out, p / "dropped.json")


# ---------------------------------------------------------------- guards
def check_aligned(a: Mapping[str, Sample], b: Mapping[str, Sample], what: str = "pred vs gold") -> list[str]:
    """Same stems and same HxW, else refuse. Returns sorted stems."""
    sa, sb = set(a), set(b)
    if sa != sb:
        raise EvalError(f"{what}: image names differ. only in first: {sorted(sa - sb)[:5]}, only in second: {sorted(sb - sa)[:5]}")
    for s in sorted(sa):
        if a[s].labels.shape != b[s].labels.shape:
            raise EvalError(f"{what}: size mismatch on {s}: {a[s].labels.shape} vs {b[s].labels.shape}")
    return sorted(sa)


def check_split(stems: Sequence[str], split: str | None, tuned_stems: Sequence[str] = ()) -> None:
    """Gold must be explicitly flagged held-out and must not overlap images used to tune thresholds."""
    if split != "heldout":
        raise EvalError(f"gold split flag is {split!r}; only 'heldout' gold may be scored (tune/dev gold would inflate results)")
    bad = sorted(set(stems) & {_stem(t) for t in tuned_stems})
    if bad:
        raise EvalError(f"gold overlaps images used to tune thresholds: {bad[:10]}")


# ---------------------------------------------------------------- per-image stats
def _conf_stats(pred: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """(NCLS,2) intersection, union over pixels where gold != GAP. Pred gap counts as a miss."""
    v = gold != GAP
    p, g = pred[v].astype(np.int64), gold[v].astype(np.int64)
    pc = np.where(p < NCLS, p, NCLS)
    g = np.where(g < NCLS, g, 0)
    cm = np.bincount(g * (NCLS + 1) + pc, minlength=NCLS * (NCLS + 1)).reshape(NCLS, NCLS + 1)
    inter = np.diag(cm[:, :NCLS]).astype(np.float64)
    union = cm.sum(1) + cm.sum(0)[:NCLS] - inter
    return np.stack([inter, union], 1)


def _boundary(lab: np.ndarray) -> np.ndarray:
    b = np.zeros(lab.shape, bool)
    d = lab[:, 1:] != lab[:, :-1]
    b[:, 1:] |= d
    b[:, :-1] |= d
    d = lab[1:] != lab[:-1]
    b[1:] |= d
    b[:-1] |= d
    return b


def _boundary_counts(pred: np.ndarray, gold: np.ndarray, tol: int = 2) -> np.ndarray:
    """[pred_boundary_hit, pred_boundary_total, gold_boundary_hit, gold_boundary_total] within tol px."""
    import cv2
    bp, bg = _boundary(pred), _boundary(gold)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    dg = cv2.dilate(bg.astype(np.uint8), k).astype(bool)
    dp = cv2.dilate(bp.astype(np.uint8), k).astype(bool)
    return np.array([(bp & dg).sum(), bp.sum(), (bg & dp).sum(), bg.sum()], np.float64)


def _inst_counts(pred: Sample, gold: Sample, thr: float) -> np.ndarray:
    """[TP, FP, FN] with greedy same-class matching by IoU >= thr (instance pixels only; ids > 0)."""
    def objs(s: Sample):
        res = []
        for i in np.unique(s.inst):
            if i == 0:
                continue
            m = s.inst == i
            vals = s.labels[m]
            vals = vals[vals != GAP]
            if vals.size:
                res.append((int(np.bincount(vals).argmax()), m))
        return res
    po, go = objs(pred), objs(gold)
    pairs = []
    for gi, (gc, gm) in enumerate(go):
        for pi, (pc, pm) in enumerate(po):
            if gc != pc:
                continue
            u = np.logical_or(gm, pm).sum()
            iou = np.logical_and(gm, pm).sum() / u if u else 0.0
            if iou >= thr:
                pairs.append((iou, gi, pi))
    pairs.sort(reverse=True)
    ug, up, tp = set(), set(), 0
    for _, gi, pi in pairs:
        if gi in ug or pi in up:
            continue
        ug.add(gi)
        up.add(pi)
        tp += 1
    return np.array([tp, len(po) - tp, len(go) - tp], np.float64)


def per_image_stats(pred: Sample, gold: Sample) -> dict[str, np.ndarray]:
    v = gold.labels != GAP
    return {
        "cls": _conf_stats(pred.labels, gold.labels),
        "bnd": _boundary_counts(pred.labels, gold.labels),
        "i50": _inst_counts(pred, gold, 0.5),
        "i75": _inst_counts(pred, gold, 0.75),
        "gap": np.array([(pred.labels[v] == GAP).sum(), v.sum()], np.float64),
        "gapall": np.array([(pred.labels == GAP).sum(), pred.labels.size], np.float64),
    }


# ---------------------------------------------------------------- aggregation
def _agg(stats: Sequence[Mapping[str, np.ndarray]], idx: np.ndarray) -> dict[str, float]:
    cls = sum(stats[i]["cls"] for i in idx)
    present = cls[:, 1] > 0
    iou = np.where(present, cls[:, 0] / np.maximum(cls[:, 1], 1), np.nan)
    res: dict[str, float] = {"mIoU": float(np.nanmean(iou)) if present.any() else float("nan")}
    for t, name in (("i50", "0.5"), ("i75", "0.75")):
        tp, fp, fn = sum(stats[i][t] for i in idx)
        den = 2 * tp + fp + fn
        res[f"instF1@{name}"] = float(2 * tp / den) if den else float("nan")
    ph, pt, gh, gt = sum(stats[i]["bnd"] for i in idx)
    pr = ph / pt if pt else np.nan
    rc = gh / gt if gt else np.nan
    res["boundaryF"] = float(2 * pr * rc / (pr + rc)) if (pr + rc) > 0 else float("nan")
    gp, gv = sum(stats[i]["gap"] for i in idx)
    res["gap_rate"] = float(gp / gv) if gv else float("nan")
    ga, gn = sum(stats[i]["gapall"] for i in idx)
    res["gap_rate_all_pixels"] = float(ga / gn) if gn else float("nan")
    for t, name in (("i50", "0.5"), ("i75", "0.75")):
        tp, fp, fn = sum(stats[i][t] for i in idx)
        res[f"instRecall@{name}"] = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    return res


def _perclass(stats: Sequence[Mapping[str, np.ndarray]]) -> dict[str, float | None]:
    cls = sum(s["cls"] for s in stats)
    return {NAMES[c]: (float(cls[c, 0] / cls[c, 1]) if cls[c, 1] > 0 else None) for c in range(NCLS)}


def bootstrap(stats: Sequence[Mapping[str, np.ndarray]], base: Sequence[Mapping[str, np.ndarray]] | None = None,
              n_boot: int = BOOT_N, seed: int = SEED) -> dict:
    """Image-level bootstrap (resample images with replacement). base => paired diff pred - baseline."""
    n = len(stats)
    point = _agg(stats, np.arange(n))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(n_boot, n))
    out = {"n_images": n, "n_boot": n_boot, "seed": seed, "metrics": {},
           "warning": (f"n={n} < {SMALL_N}: CI rat rong, chi la chi bao xu huong" if n < SMALL_N else None)}
    bp = [_agg(stats, d) for d in draws]
    bb = [_agg(base, d) for d in draws] if base is not None else None
    pb = _agg(base, np.arange(n)) if base is not None else None
    for k, v in point.items():
        arr = np.array([b[k] for b in bp], float)
        e = {"value": v, "ci95": [float(np.nanpercentile(arr, 2.5)), float(np.nanpercentile(arr, 97.5))]}
        if bb is not None and pb is not None:
            d = arr - np.array([b[k] for b in bb], float)
            e["baseline"] = pb[k]
            e["diff"] = v - pb[k]
            e["diff_ci95"] = [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]
        out["metrics"][k] = e
    out["per_class_IoU"] = _perclass(stats)
    return out


# ---------------------------------------------------------------- modes
def gap_gate(pred: Mapping[str, Sample], stems: Sequence[str]) -> dict:
    """HARD GATE (user rule): zero open pixels anywhere in pred. FAIL if any pixel == GAP.
    Also reports the share of polygons dropped as blurry: dropping lowers recall, so read it WITH instRecall."""
    gap_px = {s: int((pred[s].labels == GAP).sum()) for s in stems}
    kept = sum(pred[s].n_polys for s in stems)
    dropped = sum(pred[s].n_dropped for s in stems)
    tot = kept + dropped
    return {"gap_gate": "PASS" if not any(gap_px.values()) else "FAIL",
            "gap_pixels_by_image": gap_px,
            "dropped_polygons": dropped, "kept_polygons": kept,
            "dropped_polygon_rate": (dropped / tot) if tot else 0.0,
            "dropped_note": "Bo polygon mo lam GIAM recall: luon doc dropped_polygon_rate cung instRecall, "
                            "khong duoc bao precision/F1 rieng le khi bo nhieu."}


def eval_gold(pred: Mapping[str, Sample], gold: Mapping[str, Sample], split: str | None,
              baseline: Mapping[str, Sample] | None = None, tuned_stems: Sequence[str] = ()) -> dict:
    stems = check_aligned(pred, gold)
    check_split(stems, split, tuned_stems)
    if baseline is not None:
        check_aligned(baseline, gold, "baseline vs gold")
    st = [per_image_stats(pred[s], gold[s]) for s in stems]
    bs = [per_image_stats(baseline[s], gold[s]) for s in stems] if baseline is not None else None
    rep = bootstrap(st, bs)
    rep.update(gap_gate(pred, stems))
    rep.update(mode="gold", images=stems, split=split,
               note="Gold = nhan tay; CI bootstrap theo anh; chi la so do tren tap held-out nay.")
    return rep


def eval_iaa(gold_a: Mapping[str, Sample], gold_b: Mapping[str, Sample]) -> dict:
    stems = check_aligned(gold_a, gold_b, "gold A vs gold B")
    st = [per_image_stats(gold_b[s], gold_a[s]) for s in stems]
    rep = bootstrap(st)
    rep.update(mode="iaa", images=stems,
               note="Dong thuan nguoi-nguoi (A lam chuan, B kiem cheo): tran tren thuc te cho model; "
                    "gap_rate = pixel B de ho ma A da gan.")
    return rep


def eval_proxy(a: Mapping[str, Sample], b: Mapping[str, Sample], seconds: Mapping[str, float] | None = None) -> dict:
    """No gold: stability between config A and B plus counts. NOT accuracy."""
    stems = check_aligned(a, b, "config A vs config B")
    ious, per = [], []
    for s in stems:
        both = (a[s].labels != GAP) & (b[s].labels != GAP)
        cs = _conf_stats(np.where(both, a[s].labels, GAP).astype(np.uint8),
                         np.where(both, b[s].labels, GAP).astype(np.uint8))
        pres = cs[:, 1] > 0
        iou = float(np.mean(cs[pres, 0] / cs[pres, 1])) if pres.any() else float("nan")
        ious.append(iou)
        row: dict = {}
        for tag, x in (("A", a[s]), ("B", b[s])):
            row[f"gap_pct_{tag}"] = float(100.0 * (x.labels == GAP).mean())
            row[f"polys_{tag}"] = x.n_polys
            row[f"instances_{tag}"] = int(len(np.unique(x.inst)) - (1 if (x.inst == 0).any() else 0))
        row.update(image=s, agreement_mIoU=iou, seconds=(seconds or {}).get(s))
        per.append(row)
    arr = np.array(ious, float)
    return {"mode": "proxy", "label": PROXY_LABEL, "images": stems, "n_images": len(stems),
            "agreement_mIoU_mean": float(np.nanmean(arr)), "per_image": per,
            "gap_gate_A": gap_gate(a, stems), "gap_gate_B": gap_gate(b, stems),
            "note": f"{PROXY_LABEL}: do on dinh giua 2 cau hinh/dem so, khong co nhan tay."}


def format_report(rep: dict) -> str:
    L: list[str] = []
    if rep["mode"] == "proxy":
        L += [f"# [{PROXY_LABEL}]",
              f"agreement mIoU (A vs B) = {rep['agreement_mIoU_mean']:.4f} tren {rep['n_images']} anh"]
        for t in ("A", "B"):
            gg = rep[f"gap_gate_{t}"]
            L.append(f"gap gate {t}: {gg['gap_gate']} (gap pixel = {sum(gg['gap_pixels_by_image'].values())}); "
                     f"polygon bo vi mo = {gg['dropped_polygons']} ({100 * gg['dropped_polygon_rate']:.1f}%)")
        for r in rep["per_image"]:
            L.append(f"- {r['image']}: agree={r['agreement_mIoU']:.4f} gapA={r['gap_pct_A']:.3f}% gapB={r['gap_pct_B']:.3f}% "
                     f"polys {r['polys_A']}/{r['polys_B']} inst {r['instances_A']}/{r['instances_B']} t={r['seconds']}")
        return "\n".join(L)
    L.append(f"# mode={rep['mode']} n={rep['n_images']} boot={rep['n_boot']} seed={rep['seed']}")
    if rep["warning"]:
        L.append("CANH BAO: " + rep["warning"])
    L.append(f"GAP GATE (gap == 0): {rep['gap_gate']}" if rep["mode"] == "gold" else "")
    if rep["mode"] == "gold":
        L.append(f"polygon bo vi mo: {rep['dropped_polygons']} / {rep['dropped_polygons'] + rep['kept_polygons']} "
                 f"({100 * rep['dropped_polygon_rate']:.1f}%) - bo nhieu lam giam recall, doc cung instRecall")
    for k, e in rep["metrics"].items():
        s = f"- {k}: {e['value']:.4f} CI95 [{e['ci95'][0]:.4f}, {e['ci95'][1]:.4f}]"
        if "diff" in e:
            s += (f" | vs baseline {e['baseline']:.4f}: diff {e['diff']:+.4f} "
                  f"CI95 [{e['diff_ci95'][0]:+.4f}, {e['diff_ci95'][1]:+.4f}]")
        L.append(s)
    return "\n".join(L)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    g = sub.add_parser("gold")
    g.add_argument("--pred", required=True)
    g.add_argument("--gold", required=True)
    g.add_argument("--baseline")
    g.add_argument("--split", required=True, help="must be 'heldout'")
    g.add_argument("--tuned-stems", help="text file, one image stem per line, used to tune thresholds")
    p = sub.add_parser("proxy")
    p.add_argument("--pred-a", required=True)
    p.add_argument("--pred-b", required=True)
    p.add_argument("--run-info", help="json {stem: seconds}")
    i = sub.add_parser("iaa")
    i.add_argument("--gold-a", required=True)
    i.add_argument("--gold-b", required=True)
    for s in (g, p, i):
        s.add_argument("--out", help="write json report here (under out/opt/...)")
    a = ap.parse_args(argv)
    try:
        if a.mode == "gold":
            tuned = Path(a.tuned_stems).read_text().split() if a.tuned_stems else []
            rep = eval_gold(load_set(a.pred), load_set(a.gold), a.split,
                            load_set(a.baseline) if a.baseline else None, tuned)
        elif a.mode == "proxy":
            sec = json.loads(Path(a.run_info).read_text()) if a.run_info else None
            rep = eval_proxy(load_set(a.pred_a), load_set(a.pred_b), sec)
        else:
            rep = eval_iaa(load_set(a.gold_a), load_set(a.gold_b))
    except EvalError as e:
        print(f"REFUSED: {e}")
        return 2
    print(format_report(rep))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
