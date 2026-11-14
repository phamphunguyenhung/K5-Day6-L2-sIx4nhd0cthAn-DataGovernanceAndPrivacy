"""Bounding-box padding and image redaction (mask / blur) utilities."""

from __future__ import annotations

import cv2
import numpy as np


def pad_bbox(
    x: int,
    y: int,
    w: int,
    h: int,
    img_w: int,
    img_h: int,
    pad_ratio: float = 0.12,
    small_box_pad_ratio: float = 0.22,
    small_box_area_thr: int = 900,
) -> tuple[int, int, int, int]:
    """Expand an xywh bounding box by a ratio and clamp it to image bounds.

    Small boxes (area < ``small_box_area_thr``) receive ``small_box_pad_ratio``.
    Returns ``(0, 0, 0, 0)`` if the input box or image dimensions are invalid or
    if the box lies completely outside the image.
    """
    if img_w <= 0 or img_h <= 0 or w <= 0 or h <= 0:
        return (0, 0, 0, 0)

    area = int(w) * int(h)
    ratio = small_box_pad_ratio if area < small_box_area_thr else pad_ratio
    pad_x = max(0, int(round(w * ratio)))
    pad_y = max(0, int(round(h * ratio)))

    x1 = max(0, int(x) - pad_x)
    y1 = max(0, int(y) - pad_y)
    x2 = min(int(img_w), int(x) + int(w) + pad_x)
    y2 = min(int(img_h), int(y) + int(h) + pad_y)

    if x2 <= x1 or y2 <= y1:
        return (0, 0, 0, 0)

    return (x1, y1, x2 - x1, y2 - y1)


def redact_image(
    raw_image_bgr: np.ndarray,
    regions: list[tuple[int, int, int, int]],
    mode: str = "mask",
) -> np.ndarray:
    """Return a redacted copy of ``raw_image_bgr`` without mutating the input.

    Always reads source pixels from ``raw_image_bgr`` so overlapping regions
    are never double-blurred.

    Args:
        raw_image_bgr: Source image array (H, W, 3) or (H, W).
        regions: List of ``(x, y, w, h)`` regions to redact.
        mode: ``"mask"`` for solid ``(16, 16, 16)`` fill or ``"blur"`` for
            ``(31, 31)`` Gaussian blur.

    Raises:
        ValueError: If ``mode`` is not ``"mask"`` or ``"blur"``.
    """
    if mode not in ("mask", "blur"):
        raise ValueError(f"Unknown redaction mode: {mode!r}. Expected 'mask' or 'blur'.")

    out = raw_image_bgr.copy()
    if out.size == 0 or not regions:
        return out

    img_h, img_w = out.shape[:2]
    is_color = out.ndim >= 3

    for x, y, w, h in regions:
        if w <= 0 or h <= 0:
            continue
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(img_w, int(x) + int(w))
        y2 = min(img_h, int(y) + int(h))
        if x2 <= x1 or y2 <= y1:
            continue

        if mode == "mask":
            out[y1:y2, x1:x2] = (16, 16, 16) if is_color else 16
        else:
            roi = raw_image_bgr[y1:y2, x1:x2]
            out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (31, 31), 0)

    return out
