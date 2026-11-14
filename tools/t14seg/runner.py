"""Pipeline v2 runner (T14 segmentation draft): orchestration, cache, resume, workers, run_info, audit log.

    python -m tools.t14seg.runner --config configs/real.yaml --limit 5
    python -m tools.t14seg.runner --config out/opt/runner/real.yaml.proposed --images drive_001,drive_030 --workers 2
    python -m tools.t14seg.runner --bench            # baseline vs runner (cold / warm / resumed) and 1/2/3 workers
    python -m tools.t14seg.runner --selftest

API
    run_pipeline(cfg: dict | str | Path, images: list[str | Path] | None = None) -> dict   (run_info)

Steps (each replaceable by a hook "module:func" in cfg; default = legacy functions of tools/t14_seg_local):
    semantic(img_bgr, params)            -> (class_map uint8 HxW, info)      cached by sha256(image)+config hash
    instances(img_bgr, params)           -> (list[(cs_id, conf, bool HxW)], ood)   cached likewise
    compose  (legacy compose)            -> (lab, imap)
    closure / refine / layering (opt.)   f(lab, imap, ctx) -> (lab, imap)   ctx = {img, cm, inst, cfg}
    polygons + render (legacy)           -> polygons, gap metrics
    post hooks review_rank / cvat_export / evaluate:  f(run_dir: Path, results: list[dict], cfg: dict) -> dict | None
Hooks are never imported hard: a missing/failed hook is logged in the audit log and the legacy step runs instead.
Images never leave the machine, no network, never opens any ground-truth folder. Audit log holds no absolute paths.
"""
from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.util
import json
import os
import random
import sys
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
REAL = ROOT / "data" / "real" / "day02_traffic"
LEARNING = ROOT.parents[2]  # .../learning
DEFAULT_CVAT_SRC = LEARNING / "cvat" / "src"
CONFIG_VERSION = 1
FIXED5 = ["drive_001", "drive_002", "drive_010", "drive_030", "drive_057"]

DEFAULT_CFG: dict[str, Any] = {
    "config_version": CONFIG_VERSION,
    "seed": 42,
    "split": "dev",
    "limit": 0,
    "workers": 1,
    "threads": 2,                      # total thread budget; each worker gets max(1, threads // workers)
    "out": "out/opt/runner/run",
    "cache_dir": "out/opt/runner/cache",
    "cache": True,
    "resume": True,
    "semantic": {"model": "b2", "hook": None, "cfg_class": None, "params": {}},
    "instances": {"model": "yolo26s-seg", "imgsz": 1280, "conf": 0.25, "hook": None, "cfg_class": None,
                  "params": {}},
    "postprocess": {"eps": 0.8, "min_px": 1},
    "hooks": {"closure": None, "refine": None, "layering": None,
              "review_rank": None, "cvat_export": None, "evaluate": None},
}

_STATE: dict[str, Any] = {}  # per-process: models, legacy module (loaded once per worker)


# ------------------------------------------------------------------ config / hashing
def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(cfg: dict | str | Path | None) -> dict:
    if cfg is None:
        return json.loads(json.dumps(DEFAULT_CFG))
    if not isinstance(cfg, dict):
        import yaml
        cfg = yaml.safe_load(Path(cfg).read_text(encoding="utf-8")) or {}
        cfg = cfg.get("runner", cfg)  # allow nesting under `runner:` in a shared yaml
    return _deep_merge(DEFAULT_CFG, cfg)


def _h(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _file_sha_cached(p: Path, cache_dir: Path) -> str:
    """sha256 of a big weights file, memoised by (name,size,mtime_ns) so reruns do not rehash 100s of MB."""
    idx = cache_dir / "weights_sha.json"
    try:
        db = json.loads(idx.read_text(encoding="utf-8"))
    except Exception:
        db = {}
    st = p.stat()
    key = f"{p.name}:{st.st_size}:{st.st_mtime_ns}"
    if key not in db:
        db[key] = _sha256_file(p)
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = idx.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(db), encoding="utf-8")
        os.replace(tmp, idx)
    return db[key]


def _hook_sig(spec: str | None) -> str | None:
    """Identity of a hook = its spec + sha of the module source file, so editing a hook invalidates caches."""
    if not spec:
        return None
    mod = spec.split(":")[0]
    try:
        s = importlib.util.find_spec(mod)
        if s and s.origin and Path(s.origin).is_file():
            return f"{spec}@{_sha256_file(Path(s.origin))[:12]}"
    except Exception:
        pass
    return f"{spec}@?"


def resolve_hook(spec: str | None, log: list | None = None) -> Callable | None:
    if not spec:
        return None
    try:
        mod, fn = spec.split(":")
        return getattr(importlib.import_module(mod), fn)
    except Exception as e:  # missing module is expected while other lanes are unfinished
        if log is not None:
            log.append({"event": "hook_unavailable", "hook": spec, "error": type(e).__name__})
        return None


def _params_for(step_cfg: dict, hook: Callable | None) -> Any:
    """Hook gets its params as a dict, or as an instance of `cfg_class` ("module:Class") when given."""
    params = dict(step_cfg.get("params") or {})
    cc = step_cfg.get("cfg_class")
    if hook is not None and cc:
        cls = resolve_hook(cc)
        if cls is not None:
            return cls(**params)
    return params


# ------------------------------------------------------------------ cache I/O (atomic)
def _atomic_savez(path: Path, **arrays: Any) -> None:
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _load_sem(path: Path):
    import numpy as np
    try:
        with np.load(path) as z:
            return z["cm"], json.loads(str(z["info"]))
    except Exception:
        return None


def _save_sem(path: Path, cm, info: dict) -> None:
    _atomic_savez(path, cm=cm, info=json.dumps(info, default=str))


def _load_inst(path: Path):
    import numpy as np
    try:
        with np.load(path) as z:
            H, W = (int(v) for v in z["shape"])
            n = len(z["cs"])
            bits = z["bits"]
            inst = [(int(z["cs"][i]), float(z["conf"][i]),
                     np.unpackbits(bits[i], count=H * W).reshape(H, W).astype(bool)) for i in range(n)]
            return inst, int(z["ood"])
    except Exception:
        return None


def _save_inst(path: Path, inst: list, ood: int, shape: tuple[int, int]) -> None:
    import numpy as np
    bits = np.stack([np.packbits(m.reshape(-1)) for _, _, m in inst]) if inst else np.zeros((0, 1), np.uint8)
    _atomic_savez(path, cs=np.array([c for c, _, _ in inst], np.int16),
                  conf=np.array([s for _, s, _ in inst], np.float32), bits=bits,
                  shape=np.array(shape, np.int32), ood=np.array(ood))


# ------------------------------------------------------------------ models (loaded once per process)
def _legacy():
    src = os.environ.get("CVAT_TOOLS_SRC")
    if not src or not Path(src).is_dir():
        os.environ["CVAT_TOOLS_SRC"] = str(DEFAULT_CVAT_SRC)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if "legacy" not in _STATE:
        _STATE["legacy"] = importlib.import_module("tools.t14_seg_local")
    return _STATE["legacy"]


def _models(cfg: dict, threads: int) -> dict[str, Any]:
    if "models" not in _STATE:
        leg = _legacy()
        paths, full_stuff_map, z_order, OnnxSeg = leg._import_cvat()
        import numpy as np
        import torch
        torch.set_num_threads(threads)
        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        random.seed(cfg["seed"])
        _STATE["models"] = {"paths": paths, "full_stuff_map": full_stuff_map, "z_order": z_order,
                            "OnnxSeg": OnnxSeg, "seg": None, "yolo": None, "threads": threads}
    return _STATE["models"]


def _seg_model(cfg: dict, threads: int):
    m = _models(cfg, threads)
    if m["seg"] is None:
        m["seg"] = m["OnnxSeg"](cfg["semantic"]["model"], threads=threads)
    return m["seg"]


def _yolo_model(cfg: dict, threads: int):
    m = _models(cfg, threads)
    if m["yolo"] is None:
        from ultralytics import YOLO
        m["yolo"] = YOLO(str(Path(m["paths"].COPILOT) / f"{cfg['instances']['model']}.pt"))
    return m["yolo"]


# ------------------------------------------------------------------ one image
def _step_semantic(cfg, img, threads, log):
    hook = resolve_hook(cfg["semantic"].get("hook"), log)
    if hook is not None:
        return hook(img, _params_for(cfg["semantic"], hook))
    return _seg_model(cfg, threads).class_map(img)


def _step_instances(cfg, img, threads, log):
    hook = resolve_hook(cfg["instances"].get("hook"), log)
    if hook is not None:
        return hook(img, _params_for(cfg["instances"], hook))
    leg = _legacy()
    ic = cfg["instances"]
    return leg.instance_masks(_yolo_model(cfg, threads), img, ic["conf"], ic["imgsz"])[::1]


def cache_keys(cfg: dict, img_sha: str, wsha: dict[str, str]) -> dict[str, str]:
    sem = {"c": cfg["semantic"], "w": wsha["semantic"], "hook": _hook_sig(cfg["semantic"].get("hook"))}
    ins = {"c": cfg["instances"], "w": wsha["instances"], "hook": _hook_sig(cfg["instances"].get("hook"))}
    post = {"p": cfg["postprocess"], "h": {k: [v, _hook_sig(v)] for k, v in cfg["hooks"].items()
                                           if k in ("closure", "refine", "layering")}}
    sk, ik = f"{img_sha[:24]}_{_h(sem)}", f"{img_sha[:24]}_{_h(ins)}"
    return {"sem": sk, "inst": ik, "result": f"{sk}_{ik}_{_h(post)}"}


def process_image(task: dict) -> dict:
    """Runs in a worker (or the main process). Returns a small dict; heavy data stays in files."""
    import cv2
    import numpy as np
    cfg, iid, ipath = task["cfg"], task["iid"], Path(task["path"])
    threads, run_dir, cache_dir = task["threads"], Path(task["run_dir"]), Path(task["cache_dir"])
    log: list[dict] = []
    t_img = time.perf_counter()
    raw = ipath.read_bytes()
    img_sha = hashlib.sha256(raw).hexdigest()
    exp = task.get("expected_sha")
    if exp and exp != img_sha:
        return {"image_id": iid, "status": "error", "error": "sha256 mismatch vs manifest", "events": log}
    keys = cache_keys(cfg, img_sha, task["wsha"])
    rfile = run_dir / "polys" / f"{iid}.json"
    if cfg["resume"] and rfile.exists():
        try:
            prev = json.loads(rfile.read_text(encoding="utf-8"))
            if prev.get("result_key") == keys["result"]:
                return {"image_id": iid, "status": "resumed", "sha256": img_sha, "sec": round(time.perf_counter() - t_img, 3),
                        "steps": {}, "stats": prev["stats"], "events": [{"event": "resumed"}]}
        except Exception:
            pass
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    leg = _legacy()
    steps: dict[str, float] = {}
    hits: dict[str, bool] = {}
    use_cache = cfg["cache"]

    t = time.perf_counter()
    got = _load_sem(cache_dir / "sem" / f"{keys['sem']}.npz") if use_cache else None
    hits["semantic"] = got is not None
    if got is None:
        cm, sinfo = _step_semantic(cfg, img, threads, log)
        if use_cache:
            _save_sem(cache_dir / "sem" / f"{keys['sem']}.npz", cm, sinfo if isinstance(sinfo, dict) else {})
    else:
        cm = got[0]
    steps["semantic"] = round(time.perf_counter() - t, 3)

    t = time.perf_counter()
    got = _load_inst(cache_dir / "inst" / f"{keys['inst']}.npz") if use_cache else None
    hits["instances"] = got is not None
    if got is None:
        inst, ood = _step_instances(cfg, img, threads, log)
        if use_cache:
            _save_inst(cache_dir / "inst" / f"{keys['inst']}.npz", inst, ood, (H, W))
    else:
        inst, ood = got
    steps["instances"] = round(time.perf_counter() - t, 3)

    m = _models(cfg, threads)
    t = time.perf_counter()
    lab, imap = leg.compose(cm, m["full_stuff_map"], inst)
    steps["compose"] = round(time.perf_counter() - t, 3)
    ctx = {"img": img, "cm": cm, "inst": inst, "cfg": cfg}
    for name in ("closure", "refine", "layering"):
        fn = resolve_hook(cfg["hooks"].get(name), log)
        if fn is None:
            continue
        t = time.perf_counter()
        try:
            lab, imap = fn(lab, imap, ctx)
        except Exception as e:
            log.append({"event": "hook_failed", "hook": cfg["hooks"][name], "error": type(e).__name__})
        steps[name] = round(time.perf_counter() - t, 3)

    pp = cfg["postprocess"]
    t = time.perf_counter()
    if pp.get("mode") == "closure":  # gap-free: shared-boundary simplification, unclassified pixels filled
        from tools.t14seg.closure import close_and_simplify_ex, render_exact
        polys, pinfo = close_and_simplify_ex(lab, imap, pp["eps"], pp.get("max_points", 1500))
        steps["polygons"] = round(time.perf_counter() - t, 3)
        t = time.perf_counter()
        rend = render_exact(polys, H, W)
        steps["render"] = round(time.perf_counter() - t, 3)
        log.append({"event": "closure", "unclassified_filled_px": pinfo.get("unclassified_filled_px"),
                    "fallback_unit_px": pinfo.get("fallback_unit_px"), "n_points": pinfo.get("n_points")})
    else:
        polys = leg.polygons(lab, imap, pp["min_px"], pp["eps"])
        steps["polygons"] = round(time.perf_counter() - t, 3)
        t = time.perf_counter()
        rend = leg.render(polys, H, W, m["z_order"])
        steps["render"] = round(time.perf_counter() - t, 3)

    t = time.perf_counter()
    for d in ("labels", "overlay", "polys"):
        (run_dir / d).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(run_dir / "labels" / f"{iid}.png"), lab)
    cv2.imwrite(str(run_dir / "labels" / f"{iid}_inst.png"), imap)
    cv2.imwrite(str(run_dir / "overlay" / f"{iid}.jpg"), leg.palette()[rend])
    gap = leg.GAP
    stats = {"image_id": iid, "n_polys": len(polys), "n_instances": len(inst), "ood_dropped": ood,
             "gap_label_pct": round(100 * float((lab == gap).mean()), 3),
             "gap_render_pct": round(100 * float((rend == gap).mean()), 3),
             "render_vs_label_diff_pct": round(100 * float((rend != lab).mean()), 3)}
    doc = {"image_id": iid, "image_sha256": img_sha, "width": W, "height": H, "result_key": keys["result"],
           "stats": stats, "polys": [[int(c), p, bool(i)] for c, p, i in polys]}
    tmp = rfile.with_name(f"{iid}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(doc), encoding="utf-8")
    os.replace(tmp, rfile)  # written last = resume marker
    steps["write"] = round(time.perf_counter() - t, 3)
    return {"image_id": iid, "status": "done", "sha256": img_sha, "sec": round(time.perf_counter() - t_img, 3),
            "steps": steps, "cache_hit": hits, "stats": stats, "events": log}


def _init_worker(cfg: dict, threads: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    _models(cfg, threads)


# ------------------------------------------------------------------ RAM monitor
class RamPeak:
    """Samples summed RSS of this process + children (psutil)."""

    def __init__(self, every: float = 0.2):
        self.peak, self._stop, self._every = 0, threading.Event(), every
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        try:
            import psutil
            me = psutil.Process()
        except Exception:
            return
        while not self._stop.is_set():
            tot = 0
            for p in [me, *me.children(recursive=True)]:
                try:
                    tot += p.memory_info().rss
                except Exception:
                    pass
            self.peak = max(self.peak, tot)
            self._stop.wait(self._every)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=2)

    @property
    def peak_mb(self) -> float:
        return round(self.peak / 2**20, 1)


# ------------------------------------------------------------------ run_pipeline
def _rel(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except Exception:
        return Path(p).name  # never leak an absolute path


def _resolve_images(cfg: dict, images: list | None) -> list[tuple[str, Path, str | None]]:
    man = {}
    mf = REAL / "manifest.csv"
    if mf.exists():
        man = {r["file_id"]: r.get("sha256") for r in csv.DictReader(open(mf, encoding="utf-8"))}
    if images:
        out = []
        for x in images:
            p = Path(x)
            if not p.suffix:
                p = REAL / "images" / f"{x}.jpg"
            out.append((p.stem, p, man.get(p.stem)))
        return out
    rows = [r for r in csv.DictReader(open(REAL / "images.csv", encoding="utf-8")) if r["split"] == cfg["split"]]
    rows.sort(key=lambda r: r["image_id"])
    if cfg["limit"]:
        rows = rows[:cfg["limit"]]
    return [(r["image_id"], REAL / r["path"].replace("\\", "/"), man.get(r["image_id"])) for r in rows]


def _weights_sha(cfg: dict, cache_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    """-> (sha per step, weights file names). Resolved in the parent once; workers receive the strings."""
    leg = _legacy()
    src = os.environ["CVAT_TOOLS_SRC"]
    if src not in sys.path:
        sys.path.insert(0, src)
    from cvat_tools.core import paths
    from cvat_tools.segmentation.onnx_seg import ONNX_FILES
    onnx = Path(ONNX_FILES[cfg["semantic"]["model"]])
    pt = Path(paths.COPILOT) / f"{cfg['instances']['model']}.pt"
    del leg
    return ({"semantic": _file_sha_cached(onnx, cache_dir), "instances": _file_sha_cached(pt, cache_dir)},
            {"semantic": onnx.name, "instances": pt.name})


def run_pipeline(cfg: dict | str | Path | None, images: list[str | Path] | None = None) -> dict:
    cfg = load_config(cfg)
    t_all = time.perf_counter()
    run_id = uuid.uuid4().hex[:8]
    run_dir = (ROOT / cfg["out"]) if not Path(cfg["out"]).is_absolute() else Path(cfg["out"])
    cache_dir = (ROOT / cfg["cache_dir"]) if not Path(cfg["cache_dir"]).is_absolute() else Path(cfg["cache_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CVAT_TOOLS_SRC", str(DEFAULT_CVAT_SRC))
    todo = _resolve_images(cfg, images)
    workers = max(1, min(int(cfg["workers"]), len(todo)))
    threads = max(1, int(cfg["threads"]) // workers)
    wsha, wname = _weights_sha(cfg, cache_dir)
    tasks = [{"cfg": cfg, "iid": iid, "path": str(p), "expected_sha": sha, "threads": threads,
              "run_dir": str(run_dir), "cache_dir": str(cache_dir), "wsha": wsha} for iid, p, sha in todo]
    results: list[dict] = []
    with RamPeak() as ram:
        if workers == 1:
            for t in tasks:
                results.append(process_image(t))
                _progress(results[-1], len(results), len(tasks))
        else:
            with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(cfg, threads)) as ex:
                for r in ex.map(process_image, tasks):
                    results.append(r)
                    _progress(r, len(results), len(tasks))
    errors = [r for r in results if r["status"] == "error"]
    ok = [r for r in results if r["status"] != "error"]

    # ---- aggregate (COCO draft in the same shape as t14_seg_local) + gap report
    leg = _legacy()
    coco = {"images": [], "annotations": [], "categories": [{"id": k + 1, "name": v} for k, v in leg.NAMES.items()]}
    aid = 1
    for i, r in enumerate(ok, 1):
        doc = json.loads((run_dir / "polys" / f"{r['image_id']}.json").read_text(encoding="utf-8"))
        coco["images"].append({"id": i, "file_name": f"{r['image_id']}.jpg", "width": doc["width"],
                               "height": doc["height"], "sha256": doc["image_sha256"]})
        for cid, pts, is_inst in doc["polys"]:
            xs, ys = pts[0::2], pts[1::2]
            import numpy as np
            area = float(__import__("cv2").contourArea(np.array(pts, np.float32).reshape(-1, 2)))
            coco["annotations"].append({
                "id": aid, "image_id": i, "category_id": cid + 1, "segmentation": [pts], "area": area,
                "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)], "iscrowd": 0,
                "attributes": {"provenance": "ai_draft_unreviewed", "run_id": run_id,
                               "source_model": f"segformer-{cfg['semantic']['model']}+{cfg['instances']['model']}",
                               "image_sha256": doc["image_sha256"], "review_status": "unreviewed",
                               "shape_source": "instance" if is_inst else "semantic"}})
            aid += 1
    (run_dir / "polys").mkdir(exist_ok=True)
    (run_dir / "polys" / "coco_segmentation_draft.json").write_text(json.dumps(coco), encoding="utf-8")
    if ok:
        with open(run_dir / "gap_report.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(ok[0]["stats"].keys()))
            w.writeheader()
            w.writerows(r["stats"] for r in ok)

    # ---- post hooks
    post: dict[str, Any] = {}
    plog: list[dict] = []
    for name in ("review_rank", "cvat_export", "evaluate"):
        fn = resolve_hook(cfg["hooks"].get(name), plog)
        if fn is None:
            continue
        t = time.perf_counter()
        try:
            post[name] = {"result": fn(run_dir, ok, cfg), "sec": round(time.perf_counter() - t, 3)}
        except Exception as e:
            plog.append({"event": "hook_failed", "hook": cfg["hooks"][name], "error": type(e).__name__})

    # ---- audit log (JSONL, relative / hashed identifiers only) + run_info
    now = lambda: time.strftime("%Y-%m-%dT%H:%M:%S%z")  # noqa: E731
    with open(run_dir / "audit_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": now(), "run_id": run_id, "event": "run_start", "n_images": len(tasks),
                            "workers": workers, "cfg_hash": _h(cfg)}) + "\n")
        for r in results:
            f.write(json.dumps({"ts": now(), "run_id": run_id, "event": r["status"], "image_id": r["image_id"],
                                "image_sha256": r.get("sha256"), "sec": r.get("sec"), "steps": r.get("steps"),
                                "cache_hit": r.get("cache_hit"), "error": r.get("error"),
                                "extra": r.get("events") or None}) + "\n")
        for e in plog:
            f.write(json.dumps({"ts": now(), "run_id": run_id, **e}) + "\n")
        f.write(json.dumps({"ts": now(), "run_id": run_id, "event": "run_end", "errors": len(errors)}) + "\n")
    step_tot: dict[str, float] = {}
    for r in ok:
        for k, v in (r.get("steps") or {}).items():
            step_tot[k] = round(step_tot.get(k, 0) + v, 3)
    info = {"run_id": run_id, "seed": cfg["seed"], "config_version": cfg["config_version"], "config_hash": _h(cfg),
            "split": cfg["split"], "images": len(todo), "done": sum(r["status"] == "done" for r in results),
            "resumed": sum(r["status"] == "resumed" for r in results), "errors": len(errors),
            "workers": workers, "threads_per_worker": threads,
            "models": {"semantic": {"name": f"segformer-{cfg['semantic']['model']}-cityscapes", "file": wname["semantic"],
                                    "sha256": wsha["semantic"]},
                       "instances": {"name": cfg["instances"]["model"], "file": wname["instances"],
                                     "sha256": wsha["instances"], "imgsz": cfg["instances"]["imgsz"],
                                     "conf": cfg["instances"]["conf"]}},
            "hooks": {**{k: cfg[k].get("hook") for k in ("semantic", "instances")}, **cfg["hooks"]},
            "step_seconds_total": step_tot,
            "mean_sec_per_image": round(sum(r.get("sec", 0) for r in ok) / max(1, len(ok)), 3),
            "wall_sec": round(time.perf_counter() - t_all, 2), "peak_ram_mb": ram.peak_mb,
            "provenance": "ai_draft_unreviewed", "cache_dir": _rel(cache_dir), "out": _rel(run_dir),
            "post": {k: v["sec"] for k, v in post.items()}}
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info


def _progress(r: dict, n: int, total: int) -> None:
    print(f"[{n}/{total}] {r['image_id']} {r['status']} {r.get('sec', '')}s", flush=True)


# ------------------------------------------------------------------ benchmark (proxy for operations, not quality)
def _baseline(ids: list[str], threads: int) -> dict:
    """Legacy loop of tools/t14_seg_local (no cache, models loaded once) on the same images."""
    import cv2
    leg = _legacy()
    t0 = time.perf_counter()
    paths, fsm, z_order, OnnxSeg = leg._import_cvat()
    from ultralytics import YOLO
    import torch
    torch.set_num_threads(threads)
    seg = OnnxSeg("b2", threads=threads)
    yolo = YOLO(str(Path(paths.COPILOT) / "yolo26s-seg.pt"))
    load = time.perf_counter() - t0
    per = []
    with RamPeak() as ram:
        for iid in ids:
            img = cv2.imread(str(REAL / "images" / f"{iid}.jpg"))
            H, W = img.shape[:2]
            t = time.perf_counter()
            cm, _ = seg.class_map(img)
            inst, _ = leg.instance_masks(yolo, img, 0.25, 1280)
            lab, imap = leg.compose(cm, fsm, inst)
            polys = leg.polygons(lab, imap)
            leg.render(polys, H, W, z_order)
            per.append(round(time.perf_counter() - t, 2))
    return {"load_sec": round(load, 2), "per_image_sec": per, "mean_sec": round(sum(per) / len(per), 2),
            "peak_ram_mb": ram.peak_mb, "threads": threads}


def bench(out_json: Path | None = None) -> dict:
    import shutil
    base_out = ROOT / "out" / "opt" / "runner" / "bench"
    res: dict[str, Any] = {"images": FIXED5, "note": "operational timing only; quality metrics are proxies elsewhere"}
    res["baseline_t14_seg_local_loop_threads2"] = _baseline(FIXED5, 2)
    print("baseline", res["baseline_t14_seg_local_loop_threads2"], flush=True)

    def go(tag: str, workers: int, clear: bool, resume: bool, keep_cache: str) -> None:
        cache = base_out / f"cache_{keep_cache}"
        out = base_out / f"run_{tag}"
        if clear:
            shutil.rmtree(cache, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True) if not resume else None
        cfg = {"workers": workers, "threads": 2, "out": str(out), "cache_dir": str(cache), "resume": resume}
        t = time.perf_counter()
        info = run_pipeline(cfg, FIXED5)
        res[tag] = {"workers": workers, "wall_sec": round(time.perf_counter() - t, 2),
                    "sec_per_image_wall": round((time.perf_counter() - t) / len(FIXED5), 2),
                    "mean_sec_per_image_inproc": info["mean_sec_per_image"], "peak_ram_mb": info["peak_ram_mb"],
                    "done": info["done"], "resumed": info["resumed"], "step_seconds_total": info["step_seconds_total"]}
        print(tag, res[tag], flush=True)

    go("w1_cold", 1, True, False, "w1")
    go("w1_warm_cache_no_resume", 1, False, False, "w1")
    # resume must reuse the SAME out dir as a finished run (the previous step), so point at it explicitly
    cfg = {"workers": 1, "threads": 2, "out": str(base_out / "run_w1_warm_cache_no_resume"),
           "cache_dir": str(base_out / "cache_w1"), "resume": True}
    t = time.perf_counter()
    info = run_pipeline(cfg, FIXED5)
    res["w1_resumed"] = {"wall_sec": round(time.perf_counter() - t, 2), "done": info["done"],
                         "resumed": info["resumed"], "peak_ram_mb": info["peak_ram_mb"]}
    print("w1_resumed", res["w1_resumed"], flush=True)
    go("w2_cold", 2, True, False, "w2")
    go("w3_cold", 3, True, False, "w3")
    p = out_json or (base_out / "bench.json")
    p.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


# ------------------------------------------------------------------ self test (no models needed)
def _selftest() -> None:
    import tempfile
    import numpy as np
    c = load_config(None)
    assert c["seed"] == 42 and c["workers"] == 1
    c2 = load_config({"instances": {"imgsz": 640}})
    assert c2["instances"]["imgsz"] == 640 and c2["instances"]["conf"] == 0.25
    assert _h(c) != _h(c2) and _h(c) == _h(load_config(None))
    k1 = cache_keys(c, "a" * 64, {"semantic": "x", "instances": "y"})
    k2 = cache_keys(c2, "a" * 64, {"semantic": "x", "instances": "y"})
    assert k1["sem"] == k2["sem"] and k1["inst"] != k2["inst"] and k1["result"] != k2["result"]
    assert resolve_hook("no_such_module_xyz:f") is None
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        rng = np.random.RandomState(0)
        masks = [rng.rand(7, 9) > 0.5, rng.rand(7, 9) > 0.7]
        _save_inst(d / "i.npz", [(13, 0.9, masks[0]), (11, 0.5, masks[1])], 3, (7, 9))
        inst, ood = _load_inst(d / "i.npz")
        assert ood == 3 and inst[0][0] == 13 and (inst[0][2] == masks[0]).all() and (inst[1][2] == masks[1]).all()
        _save_inst(d / "e.npz", [], 0, (7, 9))
        assert _load_inst(d / "e.npz") == ([], 0)
        cm = rng.randint(0, 19, (5, 5)).astype(np.uint8)
        _save_sem(d / "s.npz", cm, {"s": 1.0})
        assert (_load_sem(d / "s.npz")[0] == cm).all()
        assert _load_sem(d / "missing.npz") is None
    assert "\\" not in _rel(Path("C:/some/where/else/x.png")) and ":" not in _rel(Path("C:/some/where/else/x.png"))
    print("selftest ok")


def main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m tools.t14seg.runner")
    ap.add_argument("--config", help="yaml; keys override DEFAULT_CFG")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--images", help="comma-separated image ids (e.g. drive_001,drive_030) or 'fixed5'")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--out")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    if a.bench:
        bench()
        return
    cfg = load_config(a.config)
    for k in ("limit", "workers", "out"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    if a.no_cache:
        cfg["cache"] = False
    if a.no_resume:
        cfg["resume"] = False
    imgs = FIXED5 if a.images == "fixed5" else (a.images.split(",") if a.images else None)
    info = run_pipeline(cfg, imgs)
    print(json.dumps({k: info[k] for k in ("run_id", "images", "done", "resumed", "errors", "workers",
                                          "mean_sec_per_image", "wall_sec", "peak_ram_mb", "out")}, indent=1))
    if info["errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
