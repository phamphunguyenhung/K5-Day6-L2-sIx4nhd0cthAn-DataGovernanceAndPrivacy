# Kịch bản Demo & Dàn ý Trình bày (8–10 Phút) — Track T14

- **Nhóm:** `sIx4nhd0cthAn` (`G01`)
- **Đề tài:** T14 · Auto-Annotation + Human QC Exporter
- **Người trình bày (Presenter):** Bùi Văn Thịnh (`2A202604040`)
- **Thời lượng mục tiêu:** 8–10 phút (kèm Q&A)

---

## 1. Cấu trúc Slide & Phân bổ Thời gian (8–10 Phút)

| Slide | Thời lượng | Nội dung chính | Bằng chứng / File hiển thị |
|---|---|---|---|
| **Slide 1: Bài toán & Rủi ro thực tế** | 1.0 phút | Vì sao không thể dùng thẳng nhãn tự động (auto-annotation) cho xe tự hành? Nêu 3 rủi ro: lệch class map COCO vs Taxonomy, bỏ sót vật thể nhỏ `< 150 px²`, và lỗi tràn biên khi xuất YOLO/COCO. | `reports/group_report.md` (Mục 01), `configs/tracks/t14.yaml` |
| **Slide 2: Kiến trúc Pipeline 5 Bước & Cổng Quyết định** | 1.5 phút | Luồng `Load (SHA-256)` → `Process (Adapter + Clip + Suspicion Score)` → `Rank (Top B=30)` → `Human Review + AuditLog` → `Export (CSV/YOLO/COCO + Round-trip) & Decide (REWORK/HOLD/GO)`. | Sơ đồ Mermaid trong `docs/report.md` & `src/pipeline.py` |
| **Slide 3: Phân tích 2 Failure Cases Bắt buộc** | 2.0 phút | Trình chiếu trực tiếp 2 nhóm ảnh lỗi:<br>1. **Case 1 (`aa_00.jpg`):** Map ngây thơ làm sai 234/234 box (`car` thành `motorbike`, mất 29 `stop sign`).<br>2. **Case 2 (`aa_08.jpg`, `aa_02.jpg`):** Model tự tin ở xe lớn (`conf > 0.85`) nhưng người đi bộ nhỏ ở xa (`99.8 px²`) chỉ đạt `conf = 0.297`. | `reports/figures/case1_label_map_aa_00.png`<br>`reports/figures/case2_small_objects_aa_08.png` |
| **Slide 4: Cơ chế Xếp hạng Nghi vấn (Suspicion Queue)** | 1.5 phút | Giải thích công thức $S = 0.5\cdot U + 0.3\cdot\text{small} + 0.2\cdot\text{clipped} + 0.3\cdot\text{confusion}$. Chứng minh với ngân sách $B=30$ (chỉ 13.2% tổng box), hàng đợi đã bắt trúng **18/19 box nhỏ** (gấp ~7 lần chọn ngẫu nhiên). | `out/poc/review_queue.csv`, `out/poc/contact_sheet.html` |
| **Slide 5: Live Demo / Tái lập 1 Lệnh (Offline & Deterministic)** | 2.0 phút | Chạy trực tiếp trên terminal (offline, seed 42): kiểm tra `preflight`, chạy `baseline` vs `poc`, kiểm tra `qa_report.json` (`n_invalid: 3 -> 0`, round-trip `0.00024 px`), và chạy `pytest -q`. | Terminal + `reports/decision_log.csv` |
| **Slide 6: Mở rộng Lane B (Segmentation `gap=0`) & Đạo đức/License** | 1.0–1.5 phút | Tóm tắt mở rộng trên 50 ảnh dev thực tế (đóng kín khe hở `gap = 0.0%`), lưu ý giấy phép **AGPL-3.0** của YOLO26 (gợi ý thay bằng RF-DETR Apache-2.0 khi thương mại hóa) và quy định bảo vệ dữ liệu cá nhân. | `docs/benchmark/README.md`, `reports/group_report.md` (Mục 09) |

---

## 2. Kịch bản Lệnh Chạy Demo Trực tiếp (Copy-Paste Ready)

Chạy trên **Windows PowerShell** tại thư mục gốc của repository (đảm bảo hoạt động khi tắt Wi-Fi / `--offline`):

```powershell
# Bước 1: Kiểm tra môi trường, SHA-256 manifest, offline guard và quét rò rỉ token
.venv\Scripts\python.exe -m tools.preflight

# Bước 2: Chạy Baseline (không clip box, không review queue) -> out/baseline/
.venv\Scripts\python.exe -m src.main --config configs/lab.yaml --track t14 --pipeline src.baseline_t14 --output out/baseline --run-name baseline --offline --seed 42

# Bước 3: Chạy PoC (adapter guard + clip box + suspicion queue B=30 + YOLO/COCO round-trip) -> out/poc/
.venv\Scripts\python.exe -m src.main --config configs/lab.yaml --track t14 --output out/poc --run-name poc --offline --seed 42

# Bước 4: In bằng chứng định lượng 2 Failure Modes (không cần đọc gt/)
.venv\Scripts\python.exe -m tools.t14_failure_evidence

# Bước 5: Chạy toàn bộ bộ kiểm thử tự động (Unit tests + End-to-End + Safety checks)
.venv\Scripts\python.exe -m pytest -q
```

---

## 3. Điểm nhấn cần nói rõ khi Demo với Giảng viên

1. **Tính trung thực khoa học (Data Governance Integrity):**
   - Nhấn mạnh dự đoán trong Lane A là `simulated-detector-fixture`, không bao giờ báo cáo nhầm là độ chính xác của mô hình thật.
   - Khi chạy `python -m src.evaluate ...`, chương trình trả về `exit code 2` (`ground truth file not found`) vì bộ `gt/` chỉ nằm ở máy giảng viên. Nhóm giữ nguyên kết quả `N/A (chờ GT giảng viên)` tại Bảng 1, tuyệt đối không tự bịa số F1.
2. **Vì sao Quyết định hiện tại trong `reports/decision_log.csv` là `HOLD / NOT_REVIEWED`?**
   - Pipeline tự động kiểm tra trạng thái thực tế của tệp `out/poc/review_queue_reviewed.csv`. Khi chạy lệnh PoC độc lập từ đầu (chưa nạp file đã chấm tay vào `out/poc/`), cổng `decide()` trả về `HOLD / NOT_REVIEWED` để chặn việc phát hành nhãn máy chưa qua con người kiểm duyệt.
   - Trong `tests/test_t14.py`, nhóm đã kiểm thử đầy đủ cả 4 nhánh của cổng quyết định:
     - `REWORK / EXPORT_QA_FAILED` (khi có box lỗi hoặc lệch round-trip YOLO/COCO — luôn có độ ưu tiên cao nhất).
     - `HOLD / NOT_REVIEWED` (khi `n_reviewed == 0`).
     - `HOLD / REVIEW_INCOMPLETE` (khi tỷ lệ duyệt `< 80%`).
     - `GO / REVIEWED_AND_VALID` (khi tỷ lệ duyệt `>= 80%` và `n_invalid == 0`, `yolo_ok == coco_ok == True`).

