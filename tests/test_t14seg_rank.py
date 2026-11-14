"""Tests for tools/t14seg/review_rank.py (synthetic data only, no gold, no images)."""
import numpy as np
import pandas as pd
import pytest

from tools.t14seg import review_rank as rr


def _df(rows):
    d = pd.DataFrame(rows)
    for f in rr.FEATURES:
        if f not in d:
            d[f] = np.nan
    d["S"] = rr.score(d)
    return d


def test_monotonic_each_feature():
    rng = np.random.RandomState(0)
    for _ in range(200):
        base = {f: rng.rand() for f in rr.FEATURES}
        for f in rr.FEATURES:
            hi = {**base, f: min(1.0, base[f] + rng.rand())}
            for pol in ("neutral", "zero", "renorm"):
                s0 = rr.score(pd.DataFrame([base]), nan_policy=pol)[0]
                s1 = rr.score(pd.DataFrame([hi]), nan_policy=pol)[0]
                assert s1 >= s0 - 1e-12


def test_dominating_polygon_ranks_first():
    d = _df([{"id": 1, "area": 100.0, "U": .2, "D": .2, "G": .2, "C": .2, "F": 0, "N": .1},
             {"id": 2, "area": 100.0, "U": .3, "D": .2, "G": .2, "C": .2, "F": 0, "N": .1}])
    assert d.iloc[rr.order(d)]["id"].tolist() == [2, 1]


def test_nan_handling():
    row = {"U": np.nan, "D": np.nan, "G": .4, "C": np.nan, "F": 1.0, "N": np.nan}
    for pol in ("neutral", "zero", "renorm"):
        s = rr.score(pd.DataFrame([row]), nan_policy=pol)[0]
        assert np.isfinite(s) and 0.0 <= s <= 1.0
    assert rr.score(pd.DataFrame([{f: np.nan for f in rr.FEATURES}]), nan_policy="renorm")[0] == 0.5
    assert np.isnan(rr.feat_u(None)) and np.isnan(rr.feat_u(float("nan")))
    assert rr.feat_u(0.25) == 1.0 and rr.feat_u(1.0) == 0.0 and rr.feat_u(5.0) == 0.0
    assert np.isnan(rr.feat_g([0, 0, 1, 1], 0.0, 10, 10, 13)["G"])  # degenerate polygon
    # out-of-range inputs are clipped
    assert 0 <= rr.score(pd.DataFrame([{**row, "G": 7.0, "F": np.inf}]))[0] <= 1


def test_tie_break_area_then_id():
    d = _df([{"id": 5, "area": 10.0, "U": .5}, {"id": 3, "area": 10.0, "U": .5},
             {"id": 9, "area": 50.0, "U": .5}, {"id": 1, "area": 1.0, "U": .9}])
    assert d.iloc[rr.order(d)]["id"].tolist() == [1, 9, 3, 5]
    shuffled = d.sample(frac=1.0, random_state=3).reset_index(drop=True)
    assert shuffled.iloc[rr.order(shuffled)]["id"].tolist() == [1, 9, 3, 5]  # input order irrelevant


def _coco():
    sq = lambda x, y, s: [float(v) for v in (x, y, x + s, y, x + s, y + s, x, y + s)]
    ann = lambda i, c, p, src, ar: {"id": i, "image_id": 1, "category_id": c + 1, "segmentation": [p], "area": ar,
                                     "attributes": {"shape_source": src}}
    return {"images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
            "annotations": [ann(1, 13, sq(10, 10, 30), "instance", 900.0),
                            ann(2, 13, sq(50, 50, 8), "semantic", 64.0),
                            ann(3, 0, sq(0, 0, 40), "semantic", 1600.0)]}


def test_rank_polygons_and_alt_conf():
    alt = np.full((100, 100), 13, np.uint8)
    alt[50:59, 50:59] = 0  # SegFormer disagrees on polygon 2 only
    df = rr.rank_polygons(_coco(), conf={1: 0.9}, alt_labels={"a": alt}, image_ids=["a"])
    assert set(df["id"]) == {1, 2, 3} and df["rank"].tolist() == [1, 2, 3]
    r = df.set_index("id")
    assert r.loc[2, "D"] == pytest.approx(1.0) and r.loc[1, "D"] < 0.1
    assert r.loc[2, "F"] == 1.0 and r.loc[1, "F"] == 0.0
    assert r.loc[1, "U"] == pytest.approx(0.1 / 0.75)
    assert df.iloc[0]["id"] == 2
    assert (df[list(rr.FEATURES)].fillna(0).values >= 0).all() and (df[list(rr.FEATURES)].fillna(0).values <= 1).all()


def test_drop_review_split_and_area_cap():
    d = _df([{"id": i, "image": "a", "area": 100.0, "shape_source": "semantic", "U": 1, "D": 1, "G": 1, "C": 1,
              "F": 1, "N": 1} for i in range(1, 6)] +
            [{"id": 9, "image": "a", "area": 100.0, "shape_source": "instance", "U": 0, "D": 0, "G": 0, "C": 0,
              "F": 0, "N": 0}])
    d = d.iloc[rr.order(d)].reset_index(drop=True)
    drop, rev = rr.split_drop_review(d, 0.5, max_area_frac=1.0, image_area={"a": 1000.0})
    assert sorted(drop["id"]) == [1, 2, 3, 4, 5] and rev["id"].tolist() == [9]
    drop2, rev2 = rr.split_drop_review(d, 0.5, max_area_frac=0.2, image_area={"a": 1000.0})  # cap = 200 px
    assert len(drop2) == 2 and len(rev2) == 4 and set(drop2["id"]) | set(rev2["id"]) == {1, 2, 3, 4, 5, 9}
    assert len(rr.split_drop_review(d, 1.01)[0]) == 0
    # drop count is non-increasing in tau
    sw = rr.drop_sweep(d, [0.1, 0.5, 0.9], max_area_frac=1.0)
    assert sw["n_drop"].is_monotonic_decreasing


def test_budget_and_images():
    d = _df([{"id": i, "image": "a", "area": 100.0 + i, "perim": 100.0, "shape_source": "instance", "cs": 13,
              "U": i / 10, "N": 0.0} for i in range(1, 6)])
    d = d.iloc[rr.order(d)].reset_index(drop=True)
    b = rr.select_budget(d, budget_sec=2 * (3 + 2.0) + 0.1)
    assert b["id"].tolist() == [5, 4]
    imgs = rr.rank_images(d)
    assert not bool(imgs.loc[0, "image_review"])  # 5 dynamic instances, N=0
