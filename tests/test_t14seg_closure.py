import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from tools.t14seg.closure import (  # noqa: E402
    GAP, close_and_simplify, close_and_simplify_ex, fill_dropped, gap_audit,
)


def _three_regions(H=60, W=80, seed=0):
    rng = np.random.RandomState(seed)
    ys, xs = np.mgrid[0:H, 0:W]
    wob = np.cumsum(rng.randint(-1, 2, W))  # jagged shared boundary
    lab = np.full((H, W), 1, np.uint8)
    lab[ys > 30 + wob[xs] // 3] = 2
    lab[(xs > 45 + wob[ys % W] // 4) & (ys < 30)] = 3
    return lab


@pytest.mark.parametrize("eps", [0.5, 1.5, 3.0])
def test_three_adjacent_regions_have_zero_gap(eps):
    lab = _three_regions()
    polys = close_and_simplify(lab, None, eps=eps, max_points=1500)
    a = gap_audit(polys, lab)
    assert a["gap_px"] == 0 and a["ok"]
    assert {p[0] for p in polys} == {1, 2, 3}
    assert a["diff_pct"] < 5.0


def test_gap_is_zero_even_with_dropped_and_unclassified_regions():
    lab = _three_regions()
    lab[10:20, 0:12] = GAP  # unclassified patch touching the border: must be filled, not left open
    dropped = np.zeros(lab.shape, bool)
    dropped[35:42, 20:30] = True  # faint object removed -> nearest label
    lab2, _ = fill_dropped(lab, dropped)
    assert not dropped[lab2 == GAP].any() and (lab2[dropped] != GAP).all()
    polys, info = close_and_simplify_ex(lab2, None, eps=1.0, max_points=1500)
    assert gap_audit(polys, lab2)["gap_px"] == 0
    # raw label with 255 inside: closure fills it itself, still zero gap and no 255 polygon
    polys = close_and_simplify(lab, None, eps=1.0, max_points=1500)
    a = gap_audit(polys, lab)
    assert a["gap_px"] == 0 and a["unclassified_gap_px"] == 0
    assert all(p[0] != GAP for p in polys)
