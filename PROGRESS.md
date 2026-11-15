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

## 2026-11-14 (khuya) — Lane B v2: tích hợp 10 module tối ưu, gap = 0
- **Lệnh:** `CVAT_TOOLS_SRC=<learning>/cvat/src python -m tools.t14seg.runner --config configs/t14_seg_v2.yaml` rồi `export_cvat('out/seg_v2/run','out/seg_v2/cvat_import', eps=0.8, masks=True)`.
- **Kết quả (run_id cee60c67, 50 ảnh dev):** 452s (46 chạy mới + 4 resume), 9.0s/ảnh, RAM đỉnh 5.7 GB, 0 lỗi, 0 sự kiện hook_unavailable/hook_failed trong audit log. 1090 instance YOLO26s-seg@1280 sau lọc mờ, 5766 polygon.
- **Gap sau khi vẽ polygon (opacity 100%): tối đa 0.0% trên cả 50 ảnh**, `gap_zero_pass=true`, 0 vi phạm (trước: 0.157% trung bình, 0.318% tối đa). Độ lệch polygon-vẽ-lại so với nhãn gốc: trung bình 1.12%, tối đa 4.75% (chỉ số proxy).
- **Luật hiện hành:** không hở nào; vùng mờ bị bỏ và lấp bằng nền bên dưới. Ngưỡng mờ (keep_conf 0.35, keep_area 100, keep_solid 0.8) là GIẢ THUYẾT, chưa tune vì chưa có gold.
- **Chưa đo / chặn:** (1) chưa có gold nên chưa có mIoU/F1; (2) chưa import thử vào CVAT (`cvat_import_verified=false`); (3) SAM refine tắt (chậm ~7s/instance, chưa chứng minh tốt hơn YOLO); (4) semantic policy của agent 2 chưa kiểm chứng, đang dùng SegFormer cũ; (5) layering/review_rank chưa nối vào pipeline; (6) song song nhiều worker chưa kết luận.
- **Sự cố đã xử lý:** 2 tiến trình bench của agent treo (7.4 GB RAM) đã dừng; lần chạy 50 ảnh đầu bị ngắt ở ảnh thứ 2 (nghi thiếu RAM), chạy lại thành công; `drive_057` thuộc split eval, đã loại khỏi mọi quyết định tune.
- **CVAT:** không có thao tác ghi (cvat_upload_done=false).

## 2026-11-15 — Issue #12 & Task E-01 (Bùi Văn Thịnh — 2A202604040)
- **Unit tests `decide()` (`tests/test_t14.py`):** bổ sung 5 unit tests kiểm tra nhánh `REWORK / EXPORT_QA_FAILED` (`n_invalid > 0`, `yolo_ok=False`, `coco_ok=False`) và thứ tự ưu tiên của `REWORK` trước `HOLD` / `GO`.
- **Chạy lại Baseline & PoC (Lane A, offline, seed 42):**
  - `.venv\Scripts\python.exe -m src.main --config configs/lab.yaml --track t14 --pipeline src.baseline_t14 --output out/baseline --run-name baseline --offline --seed 42` → exit 0 (`HOLD / BASELINE_ONLY`, `n_exported=227`, `n_invalid=3`).
  - `.venv\Scripts\python.exe -m src.main --config configs/lab.yaml --track t14 --output out/poc --run-name poc --offline --seed 42` → exit 0 (`HOLD / NOT_REVIEWED`, `n_exported=227`, `n_invalid=0`, `yolo_max_err_px=0.00024`, `coco_max_err_px=0.0`).
- **Decision Log & Báo cáo (`Issue #12` & `E-01`):**
  - Cập nhật `owner: "Bui Van Thinh (Lead)"` và `group_id: G01` trong `configs/lab.yaml` và `submission/README.md`.
  - Xuất `reports/decision_log.csv` chuẩn 8 cột (`ts,scope,status,canonical,reason_code,reason,owner,evidence`).
  - Hoàn thiện `reports/group_report.md` (mục 00–11), `reports/individual/2A202604040_BuiVanThinh.md`, `reports/team_contributions.csv`, và bộ tài liệu `docs/report.md`, `docs/demo.md`, `docs/contributions.md` kèm 4 ảnh minh họa lỗi trong `reports/figures/`.
- **Kiểm định:** `.venv\Scripts\python.exe -m pytest -q` → 87 passed, 5 skipped (riêng `tests/test_t14.py`: 26/26 passed); `.venv\Scripts\python.exe -m tools.preflight` → `0 FAIL, 1 WARN`.
