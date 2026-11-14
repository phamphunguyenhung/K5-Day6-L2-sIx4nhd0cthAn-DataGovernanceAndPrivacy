"""Glue hooks for the v2 runner (kept tiny on purpose; the logic lives in semantic.py / closure.py).

`prepare` is the runner's "closure" slot: f(lab, imap, ctx) -> (lab, imap).
Rule (user): no gap anywhere; anything blurry is dropped and the pixels fall back to the background beneath.
Here "blurry" = pixels SegFormer calls a thing (person..bicycle) that no kept YOLO instance covers and that
fail the size/extent test of `semantic.apply_residual` (mode from postprocess.residual_mode).
"""
from __future__ import annotations

import numpy as np


def prepare(lab, imap, ctx):
    from tools.t14_seg_local import _import_cvat, compose
    from tools.t14seg.semantic import apply_residual, thing_residual

    cm, inst, cfg = ctx["cm"], ctx["inst"], ctx["cfg"]
    pp = cfg.get("postprocess", {})
    mode = pp.get("residual_mode", "hybrid")
    if mode == "off":
        return lab, imap
    union = np.zeros(cm.shape, bool)
    for _, _, m in inst:
        union |= m
    residual = thing_residual(cm, union)
    cm2, _keep, st = apply_residual(cm, residual, mode=mode, min_area=int(pp.get("residual_min_area", 150)))
    ctx.setdefault("stats", {})["residual"] = st
    _, full_stuff_map, _, _ = _import_cvat()
    return compose(cm2, full_stuff_map, inst)
