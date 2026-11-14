"""Build the benchmark table from two finished runs. Reads files only (no model, no network), writes nothing unless --out.

    python -m tools.t14seg.bench_report --a out/seg_local --b out/seg_v2/run [--out docs/benchmark/table.md]

Each run dir needs gap_report.csv and run_info.json; cvat_import/export_report.json (sibling dir or inside run dir)
gives the gap of the file that is actually imported into CVAT. Missing fields are printed as "n/a", never guessed.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median


def _load(run: Path) -> dict:
    rows = list(csv.DictReader(open(run / "gap_report.csv", encoding="utf-8")))
    info = json.loads((run / "run_info.json").read_text(encoding="utf-8"))
    exp = None
    for c in (run / "cvat_import" / "export_report.json", run.parent / "cvat_import" / "export_report.json"):
        if c.is_file():
            exp = json.loads(c.read_text(encoding="utf-8"))
            break
    f = lambda k: [float(r[k]) for r in rows if r.get(k) not in (None, "")]  # noqa: E731
    sec = f("sec")
    return {
        "run_id": info.get("run_id"), "images": len(rows),
        "wall_sec": info.get("wall_sec"), "peak_ram_mb": info.get("peak_ram_mb"),
        "sec_img_mean": round(mean(sec), 2) if sec else info.get("mean_sec_per_image"),
        "sec_img_median": round(median(sec), 2) if sec else None,
        "instances": int(sum(f("n_instances"))), "polygons": int(sum(f("n_polys"))),
        "gap_render_mean": round(mean(f("gap_render_pct")), 4), "gap_render_max": max(f("gap_render_pct")),
        "diff_mean": round(mean(f("render_vs_label_diff_pct")), 3), "diff_max": max(f("render_vs_label_diff_pct")),
        "export_gap_mean": None if exp is None else exp.get("mean_gap_render_pct"),
        "export_gap_max": None if exp is None else exp.get("max_gap_render_pct"),
        "export_annotations": None if exp is None else exp.get("annotations"),
        "gap_zero_pass": None if exp is None else exp.get("gap_zero_pass"),
    }


def _fmt(v) -> str:
    return "n/a" if v is None else str(v)


def table(a: dict, b: dict, name_a: str = "cũ (v1)", name_b: str = "v2") -> str:
    rows = [
        ("run_id", "run_id"), ("số ảnh", "images"), ("tổng thời gian (s)", "wall_sec"),
        ("giây/ảnh (trung bình)", "sec_img_mean"), ("giây/ảnh (trung vị)", "sec_img_median"),
        ("RAM đỉnh (MB)", "peak_ram_mb"), ("số instance", "instances"), ("số polygon", "polygons"),
        ("gap polygon gốc, trung bình (%)", "gap_render_mean"), ("gap polygon gốc, tối đa (%)", "gap_render_max"),
        ("gap file import CVAT, trung bình (%)", "export_gap_mean"), ("gap file import CVAT, tối đa (%)", "export_gap_max"),
        ("gap == 0 (cổng validator)", "gap_zero_pass"), ("số annotation đã xuất", "export_annotations"),
        ("lệch polygon vẽ lại vs nhãn gốc, trung bình (%)", "diff_mean"), ("lệch tối đa (%)", "diff_max"),
    ]
    out = [f"| Chỉ số | {name_a} | {name_b} |", "|---|---|---|"]
    out += [f"| {lab} | {_fmt(a.get(k))} | {_fmt(b.get(k))} |" for lab, k in rows]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="run dir of the old pipeline")
    ap.add_argument("--b", required=True, help="run dir of v2")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t = table(_load(Path(a.a)), _load(Path(a.b)))
    print(t)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(t + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
