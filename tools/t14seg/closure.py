"""Topology-preserving closure: polygons that tile the image with ZERO gap when painted at opacity 100%.

Rule (user): no gap at all. Anything dropped (faint object) or unclassified (label 255) is FILLED with the nearest
label (`fill_dropped`), never left open.

Method (TopoJSON-like, arc based):
  1. label map + instance map -> key map -> 4-connected components ("regions"); specks < min_px are merged into the
     neighbour sharing the longest border.
  2. crack-edge graph on pixel corners; every region ring is cut at "node" corners (3+ regions meet / pinch).
  3. each arc between two regions is simplified ONCE (Douglas-Peucker importance per vertex, endpoints fixed) and
     the SAME vertices are used by both neighbours -> no sliver can open between them.
  4. one global threshold (eps, tightened by rank if over `max_points`) keeps the vertices; degenerate rings get
     vertices promoted back.
  5. safety net: render (painter, big area first); any residual uncovered pixel makes nearby arcs un-simplified,
     and as the last resort is covered by unit-square polygons. => gap == 0 by construction + verified.

Coordinates are pixel-CORNER coordinates (pixel (x,y) = [x,x+1]x[y,y+1]), which is the COCO/CVAT convention.

API:
  fill_dropped(lab, dropped_mask, imap=None) -> (lab2, imap2)
  close_and_simplify(lab, imap, eps, max_points) -> [(cid, flat_pts, is_instance)]
  close_and_simplify_ex(...) -> (polys, info)
  render_exact(polys, H, W) / gap_audit(polys, lab) -> dict   (gap must be 0)
CLI benchmark: python -m tools.t14seg.closure --labels out/seg_local/labels --out out/opt/closure
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Optional, Sequence

GAP = 255
Poly = tuple  # (cid:int, flat_pts:list[float], is_instance:bool)


# ----------------------------------------------------------------------------- fill
def fill_dropped(lab, dropped_mask, imap=None):
    """Fill pixels in `dropped_mask` (faint/removed objects) with the nearest kept label (and instance id).
    Returns (lab2, imap2); imap2 is None if imap is None. Pure, inputs are not modified."""
    import cv2
    import numpy as np

    lab2 = np.array(lab, copy=True)
    imap2 = None if imap is None else np.array(imap, copy=True)
    drop = np.asarray(dropped_mask, bool)
    if not drop.any():
        return lab2, imap2
    if drop.all():
        raise ValueError("everything is dropped; nothing to fill from")
    src = drop.astype(np.uint8)  # non-zero = to fill, zero = valid source
    _, ids = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    valid = ~drop
    lut_lab = np.zeros(int(ids.max()) + 1, lab2.dtype)
    lut_lab[ids[valid]] = lab2[valid]
    lab2[drop] = lut_lab[ids[drop]]
    if imap2 is not None:
        lut_im = np.zeros(int(ids.max()) + 1, imap2.dtype)
        lut_im[ids[valid]] = imap2[valid]
        imap2[drop] = lut_im[ids[drop]]
    return lab2, imap2


# ----------------------------------------------------------------------------- regions
def _label_regions(key, lab, min_px: int):
    """4-connected components of `key`; merge specks (< min_px) into the neighbour with the longest border.
    -> comp (HxW int32, 0..R-1), cls (R,), is_inst (R,), n_merged"""
    import cv2
    import numpy as np

    H, W = key.shape
    comp = np.zeros((H, W), np.int32)
    ckey: list[int] = []
    ccls: list[int] = []
    boxes: list[tuple[int, int, int, int]] = []
    areas: list[int] = []
    n = 0
    for k in np.unique(key):
        m = (key == k).astype(np.uint8)
        cnt, cc, st, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
        if cnt <= 1:
            continue
        sel = cc > 0
        comp[sel] = cc[sel] - 1 + n
        cls = int(lab.flat[int(np.argmax(m))])
        for j in range(1, cnt):
            ckey.append(int(k))
            ccls.append(cls)
            boxes.append((int(st[j, 0]), int(st[j, 1]), int(st[j, 2]), int(st[j, 3])))
            areas.append(int(st[j, 4]))
        n += cnt - 1
    area = np.array(areas, np.int64)
    ccls_a = np.array(ccls)
    merged = 0
    kern = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    alias = np.arange(n)
    for t in np.argsort(area, kind="stable"):
        if area[t] >= min_px:
            break
        if alias[t] != t or area[t] <= 0:
            continue
        x, y, w, h = boxes[t]
        x0, y0, x1, y1 = max(0, x - 3), max(0, y - 3), min(W, x + w + 3), min(H, y + h + 3)
        roi = comp[y0:y1, x0:x1]
        mt = roi == t
        if not mt.any():
            continue
        ring = cv2.dilate(mt.astype(np.uint8), kern).astype(bool) & ~mt
        nb = roi[ring]
        if nb.size == 0:
            continue
        ids, cn = np.unique(nb, return_counts=True)
        ok = ccls_a[ids] != GAP if ccls_a[t] != GAP else np.ones(len(ids), bool)
        if not ok.any():
            continue  # classified speck surrounded only by unclassified: keep it, never leave it open
        u = int(ids[ok][np.argmax(cn[ok])])
        roi[mt] = u
        alias[t] = u
        area[u] += area[t]
        area[t] = 0
        merged += 1
    uniq, inv = np.unique(comp, return_inverse=True)
    comp = inv.reshape(comp.shape).astype(np.int32)
    ckey_a = np.array(ckey)[uniq]
    return comp, ccls_a[uniq], ckey_a >= 1000, merged


# ----------------------------------------------------------------------------- importance (Douglas-Peucker ranks)
def _dp_importance(P):
    """Per-vertex importance: distance at which Douglas-Peucker would drop the vertex. Ends = inf, collinear = 0."""
    import numpy as np

    n = len(P)
    imp = np.zeros(n)
    imp[0] = imp[-1] = np.inf
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        a, b = P[i], P[j]
        seg = P[i + 1:j]
        d = b - a
        L = math.hypot(d[0], d[1])
        if L < 1e-12:
            dist = np.hypot(seg[:, 0] - a[0], seg[:, 1] - a[1])
        else:
            dist = np.abs(d[0] * (seg[:, 1] - a[1]) - d[1] * (seg[:, 0] - a[0])) / L
        k = int(np.argmax(dist))
        if dist[k] <= 0:
            continue
        m = i + 1 + k
        imp[m] = dist[k]
        stack.append((i, m))
        stack.append((m, j))
    return imp


def _shoelace(pts) -> float:
    import numpy as np

    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# ----------------------------------------------------------------------------- core
def close_and_simplify_ex(lab, imap=None, eps: float = 1.0, max_points: Optional[int] = 1500, min_px: int = 3):
    """Gap-free polygons. -> (polys [(cid, flat_pts, is_instance)] sorted big->small, info dict)."""
    import cv2
    import numpy as np

    lab = np.asarray(lab)
    H, W = lab.shape
    t0 = time.time()
    info: dict[str, Any] = {"eps": float(eps), "max_points": max_points}

    # 0) no gap allowed: unclassified pixels are filled with the nearest label
    unc = lab == GAP
    info["unclassified_filled_px"] = int(unc.sum())
    if unc.any():
        lab, imap = fill_dropped(lab, unc, imap)
    key = lab.astype(np.int32)
    if imap is not None:
        im = np.asarray(imap)
        key = np.where(im > 0, 1000 + im.astype(np.int32), key)

    # 1) regions
    comp, cls, is_inst, merged = _label_regions(key, lab, min_px)
    R = int(comp.max()) + 1
    info["n_regions"] = R
    info["merged_specks"] = merged

    # 2) crack edges (region on the RIGHT, clockwise on screen)
    P = np.full((H + 2, W + 2), -1, np.int32)
    P[1:-1, 1:-1] = comp
    spec = [(P[0:H, 1:W + 1], (0, 0), (1, 0)), (P[1:H + 1, 2:W + 2], (1, 0), (1, 1)),
            (P[2:H + 2, 1:W + 1], (1, 1), (0, 1)), (P[1:H + 1, 0:W], (0, 1), (0, 0))]
    SX, SY, EX, EY, REG, NREG = [], [], [], [], [], []
    for nb, (sx_, sy_), (ex_, ey_) in spec:
        yy, xx = np.nonzero(comp != nb)
        SX.append(xx + sx_); SY.append(yy + sy_); EX.append(xx + ex_); EY.append(yy + ey_)
        REG.append(comp[yy, xx]); NREG.append(nb[yy, xx])
    sx, sy, ex, ey = (np.concatenate(a).astype(np.int64) for a in (SX, SY, EX, EY))
    reg = np.concatenate(REG).astype(np.int64)
    nreg = np.concatenate(NREG).astype(np.int64)
    N = len(sx)
    if N == 0:  # single region covering the image
        poly = (int(cls[0]), [0.0, 0.0, float(W), 0.0, float(W), float(H), 0.0, float(H)], bool(is_inst[0]))
        return [poly], {**info, "n_points": 4, "n_polys": 1, "sec": time.time() - t0}
    W1 = W + 1
    sc, ec = sy * W1 + sx, ey * W1 + ex
    C = (H + 1) * W1

    # twin (reverse) edge
    kd = sc * C + ec
    od = np.argsort(kd)
    kds = kd[od]
    q = ec * C + sc
    pos = np.minimum(np.searchsorted(kds, q), N - 1)
    twin = np.where(kds[pos] == q, od[pos], -1)

    # next edge of the same region
    kout = sc * R + reg
    oo = np.argsort(kout, kind="stable")
    ks = kout[oo]
    kin = ec * R + reg
    lo, hi = np.searchsorted(ks, kin, "left"), np.searchsorted(ks, kin, "right")
    cnt = hi - lo
    assert (cnt >= 1).all(), "broken crack graph"
    nxt = oo[lo]
    ddx, ddy = ex - sx, ey - sy
    for e in np.nonzero(cnt > 1)[0]:  # pinch: sharpest right turn keeps the two pixels as separate rings
        cands = oo[lo[e]:hi[e]]
        cr = ddx[e] * ddy[cands] - ddy[e] * ddx[cands]
        nxt[e] = cands[int(np.argmax(cr))]

    node = np.zeros(C, bool)
    node[ec[nreg != nreg[nxt]]] = True
    node[ec[cnt > 1]] = True
    end_node = node[ec].tolist()
    nxt_l = nxt.tolist()

    # 3) pieces (arcs)
    piece_edges: list[np.ndarray] = []
    visited = np.zeros(N, bool)
    for e0 in np.nonzero(node[sc])[0].tolist():
        es = [e0]
        e = e0
        while not end_node[e]:
            e = nxt_l[e]
            es.append(e)
        visited[es] = True
        piece_edges.append(np.array(es))
    for e0 in np.nonzero(~visited)[0].tolist():  # node-free rings (islands)
        if visited[e0]:
            continue
        es = [e0]
        e = nxt_l[e0]
        while e != e0:
            es.append(e)
            e = nxt_l[e]
        visited[es] = True
        a = np.array(es)
        a = np.roll(a, -int(np.argmin(sc[a])))  # deterministic anchor (twin ring uses the same corner)
        piece_edges.append(a)
    nP = len(piece_edges)
    piece_of_edge = np.full(N, -1, np.int64)
    for i, es in enumerate(piece_edges):
        piece_of_edge[es] = i

    pts_of: list[Any] = []
    for es in piece_edges:
        xs = np.append(sx[es], ex[es[-1]])
        ys = np.append(sy[es], ey[es[-1]])
        d = np.stack([ddx[es], ddy[es]], 1)
        keep = np.ones(len(xs), bool)
        keep[1:-1] = (d[1:] != d[:-1]).any(1)
        pts_of.append(np.stack([xs[keep], ys[keep]], 1).astype(np.float64))
    canon = np.arange(nP)
    flip = np.zeros(nP, bool)
    for i, es in enumerate(piece_edges):
        tw = twin[es[0]]
        if tw >= 0:
            tp = int(piece_of_edge[tw])
            if tp < i:
                canon[i], flip[i] = tp, True

    # rings
    first = {int(es[0]): i for i, es in enumerate(piece_edges)}
    ring_of = np.full(nP, -1, np.int64)
    rings: list[list[int]] = []
    for i in range(nP):
        if ring_of[i] >= 0:
            continue
        r = []
        j = i
        while ring_of[j] < 0:
            ring_of[j] = len(rings)
            r.append(j)
            j = first[nxt_l[int(piece_edges[j][-1])]]
        rings.append(r)

    def full_ring(r):
        return np.concatenate([pts_of[j][:-1] for j in r])

    ring_reg = [int(reg[piece_edges[r[0]][0]]) for r in rings]
    ring_area = [abs(_shoelace(full_ring(r))) for r in rings]
    best: dict[int, int] = {}
    for ri, rg in enumerate(ring_reg):
        if rg not in best or ring_area[ri] > ring_area[best[rg]]:
            best[rg] = ri
    emitted = [ri for rg, ri in best.items() if int(cls[rg]) != GAP]

    uses = np.zeros(nP, np.int64)
    for ri in emitted:
        for j in rings[ri]:
            uses[canon[j]] += 1
    used = np.nonzero(uses > 0)[0]
    imp_of = {int(c): _dp_importance(pts_of[c]) for c in used}

    # 4) global selection
    flat_c, flat_v, flat_imp, flat_w = [], [], [], []
    for c in used:
        m = len(pts_of[c])
        if m > 2:
            flat_c.append(np.full(m - 2, c)); flat_v.append(np.arange(1, m - 1))
            flat_imp.append(imp_of[int(c)][1:-1]); flat_w.append(np.full(m - 2, uses[c]))
    kept_of = {int(c): np.zeros(len(pts_of[c]), bool) for c in used}
    for c in used:
        kept_of[int(c)][[0, -1]] = True
    base = int(sum(len(rings[ri]) for ri in emitted))
    info["eps_effective"] = float(eps)
    if flat_c:
        fc, fv, fi, fw = (np.concatenate(a) for a in (flat_c, flat_v, flat_imp, flat_w))
        sel = (fi >= eps) & (fi > 0)
        if max_points is not None and base + int(fw[sel].sum()) > max_points:
            order = np.argsort(-fi, kind="stable")
            cw = np.cumsum(fw[order])
            kmax = int(np.searchsorted(base + cw, max_points, "right"))  # items [0,kmax) fit
            sel = np.zeros(len(fi), bool)
            sel[order[:kmax]] = True
            sel &= fi > 0
            info["eps_effective"] = float(fi[sel].min()) if sel.any() else float("inf")
        for c, v in zip(fc[sel], fv[sel]):
            kept_of[int(c)][v] = True

    def ring_pts(r):
        out = []
        for j in r:
            p = pts_of[canon[j]][kept_of[int(canon[j])]]
            if flip[j]:
                p = p[::-1]
            out.append(p[:-1])
        return np.concatenate(out)

    # degenerate rings: promote the most important dropped vertex
    promoted = 0
    dropped_rings = 0
    final: dict[int, Any] = {}
    for ri in emitted:
        r = rings[ri]
        for _ in range(64):
            pp = ring_pts(r)
            if len(pp) >= 3 and abs(_shoelace(pp)) >= 0.5:
                break
            cand = [(float(imp_of[int(canon[j])][v]), int(canon[j]), v) for j in r
                    for v in range(1, len(pts_of[canon[j]]) - 1)
                    if not kept_of[int(canon[j])][v] and imp_of[int(canon[j])][v] > 0]
            if not cand:
                break
            _, c, v = max(cand)
            kept_of[c][v] = True
            promoted += 1
        pp = ring_pts(r)
        if len(pp) < 3 or abs(_shoelace(pp)) < 1e-9:
            dropped_rings += 1
            continue
        final[ri] = pp
    info["promoted_vertices"] = promoted
    info["dropped_rings"] = dropped_rings

    def build():
        res = []
        for ri, pp in final.items():
            rg = ring_reg[ri]
            res.append((abs(_shoelace(pp)), (int(cls[rg]), [float(v) for v in pp.ravel()], bool(is_inst[rg]))))
        res.sort(key=lambda t: -t[0])
        return [p for _, p in res]

    polys = build()

    # 5) safety net: verify, un-simplify near residual gaps, last resort unit squares
    repair_rounds = 0
    for _ in range(4):
        gapm = render_exact(polys, H, W) == GAP
        if not gapm.any():
            break
        repair_rounds += 1
        near = cv2.dilate(gapm.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        for c in used:
            p = np.clip(pts_of[c].astype(int), 0, [W - 1, H - 1])
            kept_of[int(c)] |= near[p[:, 1], p[:, 0]]
        for ri in list(final):
            final[ri] = ring_pts(rings[ri])
        polys = build()
    gapm = render_exact(polys, H, W) == GAP
    info["fallback_unit_px"] = int(gapm.sum())
    for y, x in zip(*np.nonzero(gapm)):
        polys.append((int(lab[y, x]), [float(x), float(y), x + 1.0, float(y), x + 1.0, y + 1.0, float(x), y + 1.0], False))
    info["repair_rounds"] = repair_rounds
    info["n_polys"] = len(polys)
    info["n_points"] = int(sum(len(p[1]) // 2 for p in polys))
    info["sec"] = round(time.time() - t0, 3)
    return polys, info


def close_and_simplify(lab, imap=None, eps: float = 1.0, max_points: Optional[int] = 1500) -> list:
    """-> polygons [(cid, flat_pts, is_instance)], gap-free at opacity 100% (see close_and_simplify_ex)."""
    return close_and_simplify_ex(lab, imap, eps, max_points)[0]


# ----------------------------------------------------------------------------- render / audit
def render_exact(polys: Sequence[Poly], H: int, W: int, scale: int = 4):
    """Painter's algorithm (area desc) with polygon geometry sampled at pixel centres (4x supersampled).
    Corner-coordinate polygons. Uncovered pixels stay GAP(255)."""
    import cv2
    import numpy as np

    big = np.full((H * scale, W * scale), GAP, np.uint8)
    arr = [(cid, np.array(p, np.float64).reshape(-1, 2)) for cid, p, *_ in polys]
    for cid, a in sorted(arr, key=lambda t: -abs(_shoelace(t[1]))):
        cv2.fillPoly(big, [np.round(a * scale).astype(np.int32)], int(cid))
    o = scale // 2
    return np.ascontiguousarray(big[o::scale, o::scale])


def render_legacy(polys: Sequence[Poly], H: int, W: int):
    """Same as tools.t14_seg_local.render (rounded ints, inclusive fill) for the old centre-coordinate polygons."""
    import cv2
    import numpy as np

    canvas = np.full((H, W), GAP, np.uint8)
    for cid, pts, *_ in sorted(polys, key=lambda t: -cv2.contourArea(np.array(t[1], np.float32).reshape(-1, 2))):
        cv2.fillPoly(canvas, [np.round(np.array(pts, np.float32).reshape(-1, 2)).astype(np.int32)], int(cid))
    return canvas


def gap_audit(polys: Sequence[Poly], lab, renderer: str = "exact") -> dict:
    """HARD check: gap_px must be 0 (gap = pixel painted by no polygon). A label==255 pixel is NOT allowed to stay
    open either (unclassified_gap_px). diff = rendered != lab (so filled 255 pixels count as diff)."""
    import numpy as np

    lab = np.asarray(lab)
    H, W = lab.shape
    rend = render_exact(polys, H, W) if renderer == "exact" else render_legacy(polys, H, W)
    gap = rend == GAP
    unc = lab == GAP
    d = rend != lab
    return {
        "gap_px": int(gap.sum()),
        "gap_pct": 100.0 * float(gap.mean()),
        "class_gap_px": int((gap & ~unc).sum()),
        "unclassified_gap_px": int((gap & unc).sum()),
        "in_label_unclassified_px": int(unc.sum()),
        "diff_px": int(d.sum()),
        "diff_pct": 100.0 * float(d.mean()),
        "n_polys": len(polys),
        "n_points": int(sum(len(p[1]) // 2 for p in polys)),
        "ok": bool(gap.sum() == 0),
    }


# ----------------------------------------------------------------------------- benchmark (proxy metrics, no GT)
def _baseline(lab, imap, eps):
    from tools.t14_seg_local import polygons  # old: independent polygons + approxPolyDP

    return polygons(lab, imap, eps=eps)


_KEYS = ("n_points", "gap_pct", "gap_px", "diff_pct")


def benchmark(label_dir: Path, ids: Sequence[str], eps_list: Sequence[float], budget: int = 1500) -> list[dict]:
    import cv2

    rows: list[dict] = []
    for iid in ids:
        lab = cv2.imread(str(label_dir / f"{iid}.png"), cv2.IMREAD_UNCHANGED)
        imap = cv2.imread(str(label_dir / f"{iid}_inst.png"), cv2.IMREAD_UNCHANGED)
        for eps in eps_list:
            a = gap_audit(_baseline(lab, imap, eps), lab, "legacy")
            rows.append({"image": iid, "method": "independent+approxPolyDP", "eps": eps, **{k: a[k] for k in _KEYS}})
            polys, info = close_and_simplify_ex(lab, imap, eps, None)
            a = gap_audit(polys, lab)
            rows.append({"image": iid, "method": "arc-shared", "eps": eps, **{k: a[k] for k in _KEYS},
                         "sec": info["sec"], "fallback_px": info["fallback_unit_px"]})
        polys, info = close_and_simplify_ex(lab, imap, 0.8, budget)
        a = gap_audit(polys, lab)
        rows.append({"image": iid, "method": f"arc-shared budget{budget}", "eps": info["eps_effective"],
                     **{k: a[k] for k in _KEYS}, "sec": info["sec"], "fallback_px": info["fallback_unit_px"]})
        for eps in (0.8, 1.5, 2.5, 4, 6, 8, 12, 16, 24):  # baseline at the same budget
            old = _baseline(lab, imap, eps)
            if sum(len(p[1]) // 2 for p in old) <= budget:
                a = gap_audit(old, lab, "legacy")
                rows.append({"image": iid, "method": f"independent budget{budget}", "eps": eps, **{k: a[k] for k in _KEYS}})
                break
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="out/seg_local/labels")
    ap.add_argument("--ids", nargs="+", default=["drive_001", "drive_002", "drive_010", "drive_030", "drive_057"])
    ap.add_argument("--eps", nargs="+", type=float, default=[0.5, 0.8, 1.5, 2.5])
    ap.add_argument("--budget", type=int, default=1500)
    ap.add_argument("--out", default="out/opt/closure")
    a = ap.parse_args()
    import cv2

    cv2.setNumThreads(2)
    rows = benchmark(Path(a.labels), a.ids, a.eps, a.budget)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "closure_table.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    for r in rows:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
