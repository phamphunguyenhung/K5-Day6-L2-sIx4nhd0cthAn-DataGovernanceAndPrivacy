"""Pixel-union mask and ground-truth coverage computation."""

from __future__ import annotations

import numpy as np


def compute_union_mask(
    img_h: int,
    img_w: int,
    boxes_xywh: list[tuple[int, int, int, int]],
) -> np.ndarray:
    """Build a 2D boolean union mask for all redacted ``(x, y, w, h)`` boxes."""
    h = max(0, int(img_h))
    w = max(0, int(img_w))
    mask = np.zeros((h, w), dtype=bool)
    if h == 0 or w == 0 or not boxes_xywh:
        return mask

    for bx, by, bw, bh in boxes_xywh:
        if bw <= 0 or bh <= 0:
            continue
        x1 = max(0, int(bx))
        y1 = max(0, int(by))
        x2 = min(w, int(bx) + int(bw))
        y2 = min(h, int(by) + int(bh))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True

    return mask


def mask_coverage(
    gt_box_xywh: tuple[int, int, int, int],
    union_mask: np.ndarray,
) -> float:
    """Compute the fraction of ``gt_box_xywh`` pixels covered by ``union_mask``."""
    gx, gy, gw, gh = gt_box_xywh
    if gw <= 0 or gh <= 0 or union_mask.size == 0:
        return 0.0

    img_h, img_w = union_mask.shape[:2]
    x1 = max(0, int(gx))
    y1 = max(0, int(gy))
    x2 = min(img_w, int(gx) + int(gw))
    y2 = min(img_h, int(gy) + int(gh))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    region = union_mask[y1:y2, x1:x2]
    cov = float(np.count_nonzero(region)) / float(region.size)
    return min(1.0, max(0.0, cov))
