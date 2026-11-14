"""Redaction masking and pixel-coverage utilities."""

from src.redaction.coverage import compute_union_mask, mask_coverage
from src.redaction.masking import pad_bbox, redact_image

__all__ = [
    "pad_bbox",
    "redact_image",
    "compute_union_mask",
    "mask_coverage",
]
