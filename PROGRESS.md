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

## 2026-11-14 (tối) — Lane B: segmentation draft local + export CVAT
- **Lệnh:** `CVAT_TOOLS_SRC=<learning>/cvat/src python -m tools.t14_seg_local --out out/seg_local` rồi `python -m tools.t14_export_cvat`.
- **Kết quả (run_id 49418fa7):** 50 ảnh dev (SHA256 khớp manifest), 358s (~7.1s/ảnh, CPU), 1319 instance YOLO26s-seg @1280, 6653 polygon (1419 instance, còn lại semantic), gap sau khi vẽ polygon: trung bình 0.0415% (nhãn gốc) / 0.157% (sau đơn giản hóa eps=0.8px), tối đa 0.318%.
- **Output (local, gitignore):** `out/seg_local/cvat_import/{coco_1.0.zip, cvat_images_1.1.zip, labels.json, images/, export_report.*}`. Mọi shape `ai_draft_unreviewed`, `source="auto"`.
- **Giới hạn:** (1) chưa import thử vào CVAT; (2) khít mép ≠ nhãn đúng: vùng thing chỉ từ SegFormer (không có instance) tạo mảng thô, xe dày bị gộp mảng, ảnh đêm bỏ sót người/xe nhỏ; (3) chưa có gold để đo IoU/F1, nên chưa có số chất lượng; (4) YOLO @640 chỉ thấy 14/32 xe ở drive_030 nên đã đổi sang 1280.
- **CVAT:** không có thao tác ghi (cvat_upload_done=false).
