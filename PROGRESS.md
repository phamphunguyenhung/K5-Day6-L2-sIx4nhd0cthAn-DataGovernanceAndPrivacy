# PROGRESS — T14

## 2026-11-14
- **Khôi phục codebase:** `main` reset về `a569f4d` (T14); bản T07 giữ ở branch `backup/t07-main-before-restore`. Force-push lên `fork/main` theo yêu cầu.
- **`.gitignore`:** thêm `data/real/`, `.env.local`, `models_optional/`, `*.pt`, `*.onnx`, `*.engine`. Kiểm bằng `git check-ignore -v` (ảnh thật, manifest, .env.local, weights đều bị ignore; `src/main.py` không bị).
- **Số ảnh thật (filesystem):** 60 jpg trong `data/real/day02_traffic/images/`; `images.csv` = 50 dev + 10 eval; license CC-BY-4.0; 0 file tracked.
- **Lane A regression (trước khi code Lane B):**
  - `python -m pytest -q` → 21 passed.
  - `python -m src.main --config configs/lab.yaml --track t14 --output out/poc_pre --run-name poc_pre --offline --seed 42` → exit 0, runtime 0.91s, đủ `predictions.csv`, `review_queue.csv`, `audit_log.jsonl`, `decision.json`, `coco.json`, `yolo/`.
  - Chưa chạy evaluate/F1: cần overlay GT của giảng viên; nếu thiếu → unavailable.
- **CVAT:** không có thao tác ghi nào.
