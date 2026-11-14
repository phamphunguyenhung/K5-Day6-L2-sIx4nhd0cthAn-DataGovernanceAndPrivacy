"""CVAT export v2 + offline validators + round-trip (LOCAL FILES ONLY, never talks to a CVAT server).

    export_cvat(run_dir, out_dir, eps=0.8, masks=False)  -> summary dict
    validate_coco(path, labels_json=None, task_images=None)       -> Report
    validate_cvat_xml(path, labels_json=None, task_images=None)   -> Report
    roundtrip(out_dir, run_dir)                                   -> dict (per-image label-map diff)

Validators are STRUCTURAL checks written from the public COCO / CVAT 1.1 formats. A pass here does NOT prove
CVAT accepts the file: that needs a real import, which this module cannot do.

CLI: python -m tools.t14seg.cvat_export --run out/seg_local --out out/opt/agent6/cvat --eps 0.8 [--masks]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence
from xml.etree import ElementTree as ET
from xml.sax.saxutils import quoteattr

COCO_MEMBER = "annotations/instances_default.json"
XML_MEMBER = "annotations.xml"
AREA_REL_TOL = 0.01   # area vs shoelace: coords are rounded to 0.1 px
AREA_ABS_TOL = 1.0
BBOX_TOL = 0.25


@dataclass
class Report:
    path: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def err(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "ok": self.ok, "errors": self.errors[:50], "n_errors": len(self.errors),
                "warnings": self.warnings[:50], "stats": self.stats}


# ---------------------------------------------------------------- helpers
def _colors(names: dict[int, str]) -> dict[int, str]:
    import numpy as np
    return {k: "#%02x%02x%02x" % tuple(int(v) for v in np.random.RandomState(100 + k).randint(60, 230, 3)) for k in names}


def _shoelace(pts: Sequence[float]) -> float:
    n = len(pts) // 2
    s = 0.0
    for i in range(n):
        x1, y1 = pts[2 * i], pts[2 * i + 1]
        x2, y2 = pts[2 * ((i + 1) % n)], pts[2 * ((i + 1) % n) + 1]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _read_member(path: Path, member: str) -> tuple[Optional[bytes], Optional[str]]:
    """-> (bytes, error). `path` may be the zip or the bare file."""
    if not path.exists():
        return None, f"file not found: {path}"
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            if member not in z.namelist():
                return None, f"zip has no '{member}' (members: {z.namelist()[:10]})"
            return z.read(member), None
    return path.read_bytes(), None


def _label_names_from_json(labels_json: Optional[Path]) -> Optional[dict[str, set[str]]]:
    if labels_json is None:
        return None
    spec = json.loads(Path(labels_json).read_text(encoding="utf-8"))
    return {l["name"]: {a["name"] for a in l.get("attributes", [])} for l in spec}


def _check_task_images(rep: Report, names: Iterable[str], task_images: Optional[Iterable[str] | Path]) -> None:
    if task_images is None:
        return
    if isinstance(task_images, (str, Path)):
        d = Path(task_images)
        have = {p.name for p in d.iterdir() if p.is_file()} if d.is_dir() else set()
        if not d.is_dir():
            rep.err(f"task_images dir not found: {d}")
            return
    else:
        have = set(task_images)
    ann = set(names)
    miss = sorted(ann - have)
    extra = sorted(have - ann)
    for n in miss[:20]:
        rep.err(f"image '{n}' referenced by annotations but not in the task images")
    if extra:
        rep.warn(f"{len(extra)} task image(s) without annotations entry (e.g. {extra[:3]})")
    rep.stats["task_images_matched"] = len(ann & have)


def _gap_pct(W: int, H: int, polys: Iterable[Sequence[float]]) -> float:
    """% of pixels covered by NO polygon when drawn at opacity 100% (painter's algorithm; order is irrelevant for
    coverage, so this is the same number whatever z_order is). Pixels are rounded like the exporter's renderer."""
    import cv2
    import numpy as np
    cv = np.zeros((H, W), np.uint8)
    for p in polys:
        cv2.fillPoly(cv, [np.round(np.array(p, np.float32).reshape(-1, 2)).astype(np.int32)], 1)
    return 100.0 * float((cv == 0).mean())


def _gap_check(rep: Report, per_image: dict[str, float], max_gap_pct: Optional[float]) -> None:
    if max_gap_pct is None:
        return
    bad = {k: v for k, v in per_image.items() if v > max_gap_pct}
    for k, v in sorted(bad.items(), key=lambda t: -t[1]):
        rep.err(f"GAP FAIL image '{k}': gap_render_pct={v:.4f} > {max_gap_pct}")
    rep.stats["gap_render_pct_max"] = round(max(per_image.values(), default=0.0), 4)
    rep.stats["gap_violations"] = sorted(bad)


# ---------------------------------------------------------------- COCO validator
def validate_coco(path: str | Path, labels_json: Optional[str | Path] = None,
                  task_images: Optional[Iterable[str] | str | Path] = None, use_pycocotools: bool = True,
                  max_gap_pct: Optional[float] = None) -> Report:
    path = Path(path)
    rep = Report(str(path))
    raw, e = _read_member(path, COCO_MEMBER)
    if e:
        rep.err(e)
        return rep
    try:
        d = json.loads(raw)
    except ValueError as ex:
        rep.err(f"invalid JSON: {ex}")
        return rep
    for k in ("images", "annotations", "categories"):
        if not isinstance(d.get(k), list):
            rep.err(f"missing/invalid top-level list '{k}'")
    if rep.errors:
        return rep

    imgs: dict[int, dict] = {}
    for im in d["images"]:
        i = im.get("id")
        if i in imgs:
            rep.err(f"duplicate image id {i}")
        if not isinstance(im.get("width"), int) or not isinstance(im.get("height"), int) or im["width"] <= 0 or im["height"] <= 0:
            rep.err(f"image {i}: bad width/height")
        if not im.get("file_name"):
            rep.err(f"image {i}: missing file_name")
        imgs[i] = im
    fnames = [im.get("file_name") for im in d["images"]]
    if len(set(fnames)) != len(fnames):
        rep.err("duplicate image file_name")

    cats: dict[int, str] = {}
    for c in d["categories"]:
        if c.get("id") in cats:
            rep.err(f"duplicate category id {c.get('id')}")
        cats[c.get("id")] = c.get("name")
    if len(set(cats.values())) != len(cats):
        rep.err("duplicate category name")
    spec = _label_names_from_json(Path(labels_json) if labels_json else None)
    if spec is not None:
        for cid, nm in cats.items():
            if nm not in spec:
                rep.err(f"category {cid} '{nm}' not declared in labels.json")
        for nm in spec:
            if nm not in cats.values():
                rep.warn(f"labels.json label '{nm}' has no category in COCO")

    seen: set[int] = set()
    npts = 0
    for a in d["annotations"]:
        aid = a.get("id")
        if aid in seen:
            rep.err(f"duplicate annotation id {aid}")
        seen.add(aid)
        if a.get("image_id") not in imgs:
            rep.err(f"ann {aid}: unknown image_id {a.get('image_id')}")
            continue
        if a.get("category_id") not in cats:
            rep.err(f"ann {aid}: unknown category_id {a.get('category_id')}")
        W, H = imgs[a["image_id"]].get("width", 0), imgs[a["image_id"]].get("height", 0)
        seg = a.get("segmentation")
        if not isinstance(seg, list) or not seg:
            rep.err(f"ann {aid}: segmentation must be a non-empty list of polygons")
            continue
        xs_all: list[float] = []
        ys_all: list[float] = []
        area_sum = 0.0
        for poly in seg:
            if not isinstance(poly, list) or len(poly) % 2 or len(poly) < 6:
                rep.err(f"ann {aid}: polygon needs an even number of coords and >= 3 points (got {len(poly) if isinstance(poly, list) else poly!r})")
                continue
            npts += len(poly) // 2
            xs, ys = poly[0::2], poly[1::2]
            if min(xs) < 0 or min(ys) < 0 or max(xs) > W or max(ys) > H:
                rep.err(f"ann {aid}: point outside image {W}x{H}")
            xs_all += xs
            ys_all += ys
            area_sum += _shoelace(poly)
        if not xs_all:
            continue
        bb = a.get("bbox")
        if not (isinstance(bb, list) and len(bb) == 4):
            rep.err(f"ann {aid}: bbox missing/invalid")
        else:
            exp = [min(xs_all), min(ys_all), max(xs_all) - min(xs_all), max(ys_all) - min(ys_all)]
            if any(abs(x - y) > BBOX_TOL for x, y in zip(bb, exp)):
                rep.err(f"ann {aid}: bbox {bb} != polygon extent {[round(v, 1) for v in exp]}")
        ar = a.get("area")
        if not isinstance(ar, (int, float)) or ar <= 0:
            rep.err(f"ann {aid}: area missing/<=0")
        elif abs(ar - area_sum) > max(AREA_ABS_TOL, AREA_REL_TOL * area_sum) and len(seg) == 1:
            rep.err(f"ann {aid}: area {ar} != polygon area {area_sum:.1f}")
        if a.get("iscrowd", 0) not in (0, 1):
            rep.err(f"ann {aid}: iscrowd must be 0/1")

    _check_task_images(rep, [n for n in fnames if n], task_images)
    if max_gap_pct is not None and not rep.errors:
        polys_by: dict[int, list] = {i: [] for i in imgs}
        for a in d["annotations"]:
            polys_by[a["image_id"]] += a["segmentation"]
        _gap_check(rep, {imgs[i]["file_name"]: _gap_pct(imgs[i]["width"], imgs[i]["height"], pl) for i, pl in polys_by.items()}, max_gap_pct)
    rep.stats.update({"images": len(imgs), "annotations": len(d["annotations"]), "categories": len(cats), "points": npts})

    if use_pycocotools and not rep.errors:
        try:
            from pycocotools.coco import COCO  # type: ignore
        except ImportError:
            rep.warn("pycocotools not installed: structural self-check only")
        else:
            with tempfile.TemporaryDirectory() as td:
                p = Path(td) / "i.json"
                p.write_bytes(raw)
                import contextlib
                import io
                with contextlib.redirect_stdout(io.StringIO()):
                    coco = COCO(str(p))
                    bad = 0
                    for aid, a in coco.anns.items():
                        m = coco.annToMask(a)
                        if m.shape != (imgs[a["image_id"]]["height"], imgs[a["image_id"]]["width"]):
                            rep.err(f"ann {aid}: pycocotools mask shape {m.shape} != image size")
                        elif int(m.sum()) == 0:
                            bad += 1
                if bad:
                    rep.warn(f"{bad} annotation(s) rasterize to an empty mask (sub-pixel polygons)")
            rep.stats["pycocotools_loaded"] = True
    return rep


# ---------------------------------------------------------------- CVAT XML validator
def validate_cvat_xml(path: str | Path, labels_json: Optional[str | Path] = None,
                      task_images: Optional[Iterable[str] | str | Path] = None,
                      max_gap_pct: Optional[float] = None) -> Report:
    path = Path(path)
    rep = Report(str(path))
    raw, e = _read_member(path, XML_MEMBER)
    if e:
        rep.err(e)
        return rep
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as ex:
        rep.err(f"XML parse error: {ex}")
        return rep
    if root.tag != "annotations":
        rep.err(f"root tag must be <annotations>, got <{root.tag}>")
        return rep
    if (root.findtext("version") or "").strip() != "1.1":
        rep.err("<version> must be 1.1")

    declared: dict[str, set[str]] = {}
    lab_nodes = root.findall("./meta/task/labels/label") or root.findall("./meta/job/labels/label")
    if not lab_nodes:
        rep.err("no <labels> declared under meta/task")
    for ln in lab_nodes:
        nm = (ln.findtext("name") or "").strip()
        if not nm:
            rep.err("<label> without <name>")
            continue
        if nm in declared:
            rep.err(f"label '{nm}' declared twice")
        attrs: set[str] = set()
        for an in ln.findall("./attributes/attribute"):
            an_name = (an.findtext("name") or "").strip()
            if not an_name:
                rep.err(f"label '{nm}': attribute without <name>")
            for req in ("mutable", "input_type"):
                if an.find(req) is None:
                    rep.err(f"label '{nm}' attribute '{an_name}': missing <{req}>")
            attrs.add(an_name)
        declared[nm] = attrs
    spec = _label_names_from_json(Path(labels_json) if labels_json else None)
    if spec is not None:
        for nm, at in declared.items():
            if nm not in spec:
                rep.err(f"label '{nm}' in XML not in labels.json")
            elif at != spec[nm]:
                rep.err(f"label '{nm}': XML attributes {sorted(at)} != labels.json {sorted(spec[nm])}")

    size = root.findtext("./meta/task/size")
    img_nodes = root.findall("image")
    if size is not None and size.strip().isdigit() and int(size) != len(img_nodes):
        rep.err(f"<size> {size} != number of <image> {len(img_nodes)}")

    ids: set[str] = set()
    names: list[str] = []
    npoly = 0
    gaps: dict[str, float] = {}
    for im in img_nodes:
        iid, nm = im.get("id"), im.get("name")
        if iid in ids:
            rep.err(f"duplicate image id {iid}")
        ids.add(iid)
        names.append(nm or "")
        try:
            W, H = int(im.get("width")), int(im.get("height"))
            if W <= 0 or H <= 0:
                raise ValueError
        except (TypeError, ValueError):
            rep.err(f"image '{nm}': bad width/height")
            continue
        okpolys: list[list[float]] = []
        for k, pg in enumerate(im.findall("polygon")):
            npoly += 1
            tag = f"image '{nm}' polygon #{k}"
            lb = pg.get("label")
            if lb not in declared:
                rep.err(f"{tag}: label '{lb}' not declared in <labels>")
            if pg.get("source") != "auto":
                rep.err(f"{tag}: source must be 'auto' (got {pg.get('source')!r})")
            if pg.get("occluded") not in ("0", "1"):
                rep.err(f"{tag}: occluded must be 0/1")
            try:
                int(pg.get("z_order"))
            except (TypeError, ValueError):
                rep.err(f"{tag}: z_order missing/not int")
            try:
                pts = [tuple(float(v) for v in p.split(",")) for p in (pg.get("points") or "").split(";")]
                if any(len(p) != 2 for p in pts):
                    raise ValueError
            except ValueError:
                rep.err(f"{tag}: unparsable points")
                continue
            okpolys.append([v for pt in pts for v in pt])
            if len(pts) < 3:
                rep.err(f"{tag}: needs >= 3 points")
            if any(x < 0 or y < 0 or x > W or y > H for x, y in pts):
                rep.err(f"{tag}: point outside image {W}x{H}")
            for at in pg.findall("attribute"):
                if lb in declared and at.get("name") not in declared[lb]:
                    rep.err(f"{tag}: attribute '{at.get('name')}' not declared for label '{lb}'")
        if max_gap_pct is not None:
            gaps[nm or ""] = _gap_pct(W, H, [p for p in okpolys if len(p) >= 6])
    if len(set(names)) != len(names):
        rep.err("duplicate image name")
    _check_task_images(rep, names, task_images)
    _gap_check(rep, gaps, max_gap_pct)
    rep.stats.update({"images": len(img_nodes), "polygons": npoly, "labels": len(declared)})
    return rep


# ---------------------------------------------------------------- export
def export_cvat(run_dir: str | Path, out_dir: str | Path, eps: float = 0.8, masks: bool = False,
                copy_images: bool = True, images_dir: Optional[str | Path] = None,
                labels_spec: Optional[Path] = None, max_points: Optional[int] = 1500) -> dict[str, Any]:
    """Read a t14_seg_local run (run_info.json, labels/*.png, labels/*_inst.png, polys/coco_segmentation_draft.json),
    write labels.json, coco_1.0.zip, cvat_images_1.1.zip, optionally cvat_masks_1.1.zip, export_report.json.
    Never overwrites run_dir. Validators are run on the outputs and their verdict is stored in the summary."""
    import cv2
    import numpy as np
    from tools.t14_seg_local import GAP, NAMES, REAL
    from tools.t14seg.closure import close_and_simplify, render_exact

    src, out = Path(run_dir), Path(out_dir)
    info = json.loads((src / "run_info.json").read_text(encoding="utf-8"))
    draft = json.loads((src / "polys" / "coco_segmentation_draft.json").read_text(encoding="utf-8"))
    rid = info["run_id"]
    img_src = Path(images_dir) if images_dir else REAL / "images"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    if copy_images:
        (out / "images").mkdir()
    colors = _colors(NAMES)

    per_image: list[dict[str, Any]] = []
    images, anns, xml_img = [], [], []
    used: set[int] = set()
    aid = 0
    for i, im in enumerate(draft["images"], 1):
        stem, W, H = Path(im["file_name"]).stem, im["width"], im["height"]
        if copy_images:
            shutil.copy2(img_src / im["file_name"], out / "images" / im["file_name"])
        lab = cv2.imread(str(src / "labels" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        imap = cv2.imread(str(src / "labels" / f"{stem}_inst.png"), cv2.IMREAD_UNCHANGED)
        # gap-free polygons (shared-boundary simplification, big -> small); unclassified pixels are filled
        polys = close_and_simplify(lab, imap, eps, max_points)
        polys = [(c, [min(max(v, 0.0), float(W if j % 2 == 0 else H)) for j, v in enumerate(p)], f) for c, p, f in polys]
        rend = render_exact(polys, H, W)
        images.append({"id": i, "file_name": im["file_name"], "width": W, "height": H})
        shapes = []
        for z, (cid, pts, is_inst) in enumerate(polys):
            used.add(cid)
            xs, ys = pts[0::2], pts[1::2]
            rp = [round(v, 1) for v in pts]
            aid += 1
            anns.append({"id": aid, "image_id": i, "category_id": cid + 1, "segmentation": [rp],
                         "area": round(_shoelace(rp), 1),
                         "bbox": [round(min(xs), 1), round(min(ys), 1), round(max(xs) - min(xs), 1), round(max(ys) - min(ys), 1)],
                         "iscrowd": 0, "attributes": {"ai_draft": True, "run_id": rid,
                                                     "shape_source": "instance" if is_inst else "semantic"}})
            p = ";".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
            shapes.append(f'    <polygon label={quoteattr(NAMES[cid])} source="auto" occluded="0" points="{p}" z_order="{z}">\n'
                          f'      <attribute name="ai_draft">true</attribute>\n      <attribute name="run_id">{rid}</attribute>\n'
                          f'    </polygon>')
        xml_img.append(f'  <image id="{i - 1}" name={quoteattr(im["file_name"])} width="{W}" height="{H}">\n'
                       + "\n".join(shapes) + "\n  </image>")
        per_image.append({"image": stem, "polygons": len(polys), "gap_render_pct": round(100 * float((rend == GAP).mean()), 4),
                          "render_vs_label_diff_pct": round(100 * float((rend != lab).mean()), 4)})
        if masks:
            _write_masks(out, stem, lab, imap, colors)

    ucats = sorted(used)
    labels = [{"name": NAMES[k], "color": colors[k], "type": "polygon",
               "attributes": [{"name": "ai_draft", "mutable": False, "input_type": "checkbox", "default_value": "true", "values": ["true"]},
                              {"name": "run_id", "mutable": False, "input_type": "text", "default_value": rid, "values": [rid]}]}
              for k in ucats]
    (out / "labels.json").write_text(json.dumps(labels, indent=1), encoding="utf-8")
    coco = {"info": {"description": "AI draft, unreviewed", "version": "1.0", "year": 2026},
            "licenses": [{"id": 1, "name": "CC-BY-4.0 (UA-DETRAC via Day02)", "url": ""}],
            "images": images, "annotations": anns,
            "categories": [{"id": k + 1, "name": NAMES[k], "supercategory": ""} for k in ucats]}
    xml_labels = "\n".join(
        f'      <label>\n        <name>{NAMES[k]}</name>\n        <color>{colors[k]}</color>\n        <type>polygon</type>\n'
        f'        <attributes>\n          <attribute><name>ai_draft</name><mutable>False</mutable><input_type>checkbox</input_type>'
        f'<default_value>true</default_value><values>true</values></attribute>\n'
        f'          <attribute><name>run_id</name><mutable>False</mutable><input_type>text</input_type>'
        f'<default_value>{rid}</default_value><values>{rid}</values></attribute>\n'
        f'        </attributes>\n      </label>' for k in ucats)
    xml = ('<?xml version="1.0" encoding="utf-8"?>\n<annotations>\n  <version>1.1</version>\n  <meta>\n    <task>\n'
           f'      <name>t14-seg-draft-{rid}</name>\n      <size>{len(images)}</size>\n      <mode>annotation</mode>\n'
           f'      <labels>\n{xml_labels}\n      </labels>\n    </task>\n  </meta>\n' + "\n".join(xml_img) + "\n</annotations>\n")
    with zipfile.ZipFile(out / "coco_1.0.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(COCO_MEMBER, json.dumps(coco))
    with zipfile.ZipFile(out / "cvat_images_1.1.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(XML_MEMBER, xml)
    zips = ["coco_1.0.zip", "cvat_images_1.1.zip"]
    if masks:
        _zip_masks(out, ucats, NAMES, colors, [Path(im["file_name"]).stem for im in draft["images"]])
        zips.append("cvat_masks_1.1.zip")

    task_imgs = out / "images" if copy_images else None
    vc = validate_coco(out / "coco_1.0.zip", out / "labels.json", task_imgs, max_gap_pct=0.0)
    vx = validate_cvat_xml(out / "cvat_images_1.1.zip", out / "labels.json", task_imgs, max_gap_pct=0.0)
    summ = {"run_id": rid, "eps_px": eps, "images": len(images), "annotations": len(anns),
            "labels": [NAMES[k] for k in ucats], "provenance": "ai_draft_unreviewed",
            "max_gap_render_pct": max(r["gap_render_pct"] for r in per_image),
            "mean_gap_render_pct": round(sum(r["gap_render_pct"] for r in per_image) / len(per_image), 4),
            "gap_zero_pass": bool(vc.ok and vx.ok and not vc.stats.get("gap_violations") and not vx.stats.get("gap_violations")),
            "gap_violations": vx.stats.get("gap_violations", []),
            "validate_coco": vc.to_dict(), "validate_cvat_xml": vx.to_dict(),
            "zip_sha256": {n: hashlib.sha256((out / n).read_bytes()).hexdigest() for n in zips},
            "cvat_upload_done": False, "cvat_import_verified": False}
    (out / "export_report.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    return summ


# ---------------------------------------------------------------- masks ("Segmentation mask 1.1")
def _hex_rgb(h: str) -> tuple[int, int, int]:
    return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)


def _write_masks(out: Path, stem: str, lab: Any, imap: Any, colors: dict[int, str]) -> None:
    """SegmentationClass = colour per class; SegmentationObject = colour per instance (0 = background/gap)."""
    import cv2
    import numpy as np
    from tools.t14_seg_local import GAP
    for sub in ("SegmentationClass", "SegmentationObject"):
        (out / "_masks" / sub).mkdir(parents=True, exist_ok=True)
    cls = np.zeros(lab.shape + (3,), np.uint8)
    for k, h in colors.items():
        cls[lab == k] = _hex_rgb(h)
    cls[lab == GAP] = 0
    cv2.imwrite(str(out / "_masks" / "SegmentationClass" / f"{stem}.png"), cv2.cvtColor(cls, cv2.COLOR_RGB2BGR))
    obj = np.zeros(lab.shape + (3,), np.uint8)
    for n, v in enumerate(np.unique(imap[imap > 0]), 1):   # distinct, deterministic instance colours
        obj[imap == v] = ((n * 67) % 200 + 40, (n * 131) % 200 + 40, (n * 29) % 200 + 40)
    cv2.imwrite(str(out / "_masks" / "SegmentationObject" / f"{stem}.png"), cv2.cvtColor(obj, cv2.COLOR_RGB2BGR))


def _zip_masks(out: Path, used: list[int], names: dict[int, str], colors: dict[int, str], stems: list[str]) -> None:
    lm = ["# label:color_rgb:parts:actions", "background:0,0,0::"] + [
        f"{names[k]}:{','.join(str(v) for v in _hex_rgb(colors[k]))}::" for k in used]
    with zipfile.ZipFile(out / "cvat_masks_1.1.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("labelmap.txt", "\n".join(lm) + "\n")
        z.writestr("ImageSets/Segmentation/default.txt", "\n".join(stems) + "\n")
        for sub in ("SegmentationClass", "SegmentationObject"):
            for s in stems:
                z.write(out / "_masks" / sub / f"{s}.png", f"{sub}/{s}.png")
    shutil.rmtree(out / "_masks")


def validate_masks_zip(path: str | Path, stems: Optional[Sequence[str]] = None) -> Report:
    """Structure check of the mask zip: labelmap.txt, background line, PNG per stem, only labelmap colours used."""
    import cv2
    import numpy as np
    rep = Report(str(path))
    with zipfile.ZipFile(path) as z:
        nl = z.namelist()
        if "labelmap.txt" not in nl:
            rep.err("labelmap.txt missing")
            return rep
        cmap: dict[tuple[int, int, int], str] = {}
        for ln in z.read("labelmap.txt").decode().splitlines():
            if not ln.strip() or ln.startswith("#"):
                continue
            nm, col = ln.split(":")[:2]
            cmap[tuple(int(v) for v in col.split(","))] = nm  # type: ignore[assignment]
        if (0, 0, 0) not in cmap:
            rep.err("labelmap.txt has no background (0,0,0) entry")
        pngs = sorted(n for n in nl if n.startswith("SegmentationClass/") and n.endswith(".png"))
        if stems is not None:
            for s in stems:
                if f"SegmentationClass/{s}.png" not in nl:
                    rep.err(f"missing SegmentationClass/{s}.png")
        for n in pngs:
            im = cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR)
            rgb = im[..., ::-1].reshape(-1, 3)
            uniq = {tuple(int(v) for v in c) for c in np.unique(rgb, axis=0)}
            bad = uniq - set(cmap)
            if bad:
                rep.err(f"{n}: colours not in labelmap: {sorted(bad)[:3]}")
        rep.stats["class_masks"] = len(pngs)
    return rep


# ---------------------------------------------------------------- round trip
def roundtrip(out_dir: str | Path, run_dir: str | Path) -> dict[str, Any]:
    """Re-read coco_1.0.zip and cvat_images_1.1.zip from out_dir, rasterise their polygons like an annotation tool
    (COCO: big area first; XML: ascending z_order), rebuild the label map and diff it against (a) the source
    labels/*.png and (b) each other. Mismatch % = share of pixels whose class differs. GAP pixels count as mismatch."""
    import cv2
    import numpy as np
    from tools.t14_seg_local import GAP, NAMES

    out, src = Path(out_dir), Path(run_dir)
    name2id = {v: k for k, v in NAMES.items()}
    coco = json.loads(zipfile.ZipFile(out / "coco_1.0.zip").read(COCO_MEMBER))
    cat = {c["id"]: name2id[c["name"]] for c in coco["categories"]}
    root = ET.fromstring(zipfile.ZipFile(out / "cvat_images_1.1.zip").read(XML_MEMBER))
    by_img: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)

    rows, issues = [], []
    xml_by_name = {im.get("name"): im for im in root.findall("image")}
    for im in coco["images"]:
        H, W, stem = im["height"], im["width"], Path(im["file_name"]).stem
        lab = cv2.imread(str(src / "labels" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        c_map = np.full((H, W), GAP, np.uint8)
        for a in sorted(by_img.get(im["id"], []), key=lambda a: -a["area"]):
            for poly in a["segmentation"]:
                cv2.fillPoly(c_map, [np.round(np.array(poly, np.float32).reshape(-1, 2)).astype(np.int32)], cat[a["category_id"]])
        x_map = np.full((H, W), GAP, np.uint8)
        xi = xml_by_name.get(im["file_name"])
        if xi is None:
            issues.append(f"{im['file_name']}: absent from XML")
            continue
        for pg in sorted(xi.findall("polygon"), key=lambda p: int(p.get("z_order"))):
            pts = np.array([[float(v) for v in p.split(",")] for p in pg.get("points").split(";")], np.float32)
            cv2.fillPoly(x_map, [np.round(pts).astype(np.int32)], name2id[pg.get("label")])
        rows.append({"image": stem,
                     "coco_vs_source_pct": round(100 * float((c_map != lab).mean()), 4),
                     "xml_vs_source_pct": round(100 * float((x_map != lab).mean()), 4),
                     "coco_vs_xml_pct": round(100 * float((c_map != x_map).mean()), 4),
                     "coco_gap_pct": round(100 * float((c_map == GAP).mean()), 4)})
    if len(xml_by_name) != len(coco["images"]):
        issues.append(f"image count COCO {len(coco['images'])} != XML {len(xml_by_name)}")
    return {"images": len(rows), "issues": issues,
            "max_coco_vs_source_pct": max((r["coco_vs_source_pct"] for r in rows), default=None),
            "max_xml_vs_source_pct": max((r["xml_vs_source_pct"] for r in rows), default=None),
            "max_coco_vs_xml_pct": max((r["coco_vs_xml_pct"] for r in rows), default=None),
            "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="out/seg_local")
    ap.add_argument("--out", default="out/opt/agent6/cvat")
    ap.add_argument("--eps", type=float, default=0.8)
    ap.add_argument("--masks", action="store_true")
    ap.add_argument("--no-images", action="store_true")
    a = ap.parse_args()
    s = export_cvat(a.run, a.out, a.eps, masks=a.masks, copy_images=not a.no_images)
    rt = roundtrip(a.out, a.run)
    (Path(a.out) / "roundtrip.json").write_text(json.dumps(rt, indent=1), encoding="utf-8")
    print(json.dumps({k: s[k] for k in ("images", "annotations", "max_gap_render_pct", "validate_coco", "validate_cvat_xml")}, indent=1))
    print(json.dumps({k: v for k, v in rt.items() if k != "rows"}, indent=1))
    if a.masks:
        print(validate_masks_zip(Path(a.out) / "cvat_masks_1.1.zip").to_dict())


if __name__ == "__main__":
    main()
