# Báo cáo cá nhân · Trần Nguyễn Phi Long (2A202604058)

**Task:** Issue #7 · Chấm tay 30 box trên trang review · git author `Keishaa`

## 1. Việc đã làm

1. **Lần 1 (không dùng):** duyệt 30 box trên `https://t14-review.pages.dev/`. Trang này dựng từ 30 box score nghi ngờ
   **thấp nhất** (`tools/t14_lowest_sheet.py`). Kết quả approve 8 · edit 22 · reject 0. Khi áp vào pipeline thì exit 2
   (`reviewed file has items outside the review queue`): `src/pipeline.py::review` chỉ nhận box thuộc
   `out/poc/review_queue.csv` (top 30 score **cao nhất**), và 2 bộ không trùng box nào. Bộ thấp nhất cũng lệch bước 03
   của đề (ưu tiên box bất định), nên bỏ.
2. **Dựng lại trang review từ đúng queue của pipeline:** dùng `tools/t14_lowest_sheet.py` (mặc định đọc
   `out/poc/review_queue.csv`) → `out/site_top/index.html`, cùng 30 `item_id` và thứ tự với `out/poc/review_queue.csv` (đã đối chiếu bằng script).
3. **Lần 2 (bản nộp):** duyệt một mình 30 box trên trang mới. Tiêu chí: `docs/HUONG_DAN_GAN_NHAN_CVAT.md`. Có zoom
   cho box nhỏ. Edit thì vẽ lại khung bằng `tools/t14_box_picker.py`: trang offline, kéo chuột vẽ khung trên ảnh phóng 3×,
   trang đổi về pixel ảnh gốc và in `class,x,y,w,h`. Tool chỉ đo khung người vẽ, không gợi ý box.
4. Chỉnh file export trước khi nộp (ghi rõ để minh bạch):
   - cột `reviewer`: trên trang gõ `Keishaa` (tên GitHub), đổi thành `long` theo tiêu chí issue #7;
   - `aa_27#0`: đã vẽ khung (`override_value`) nhưng quên chọn quyết định trong dropdown. Chốt là `edit`, `reviewed_at` để trống;
   - 17 dòng edit chưa có note: điền "Vẽ lại khung cho ôm sát vật thể, giữ class" (đúng việc đã làm: 17/17 edit không đổi class).
5. Áp file review vào pipeline, kiểm decision, audit log và correction diff.

## 2. Lệnh đã chạy

```powershell
.venv\Scripts\python -m src.main --config configs/lab.yaml --track t14 --output out/poc --run-name poc --offline --seed 42
.venv\Scripts\python -m tools.t14_lowest_sheet --out out/site_top
.venv\Scripts\python -m tools.t14_box_picker --queue web/public/queue.json --out out/site_top/box_picker.html
# (web/public/queue.json cùng thứ tự với out/poc/review_queue.csv)

# review xong → reports/review_queue.csv, copy sang input của bước review:
copy reports\review_queue.csv out\poc\review_queue_reviewed.csv
.venv\Scripts\python -m src.main --config configs/lab.yaml --track t14 --output out/poc --run-name poc --offline --seed 42
# [main] status=0
# decision.json: GO / REVIEWED_AND_VALID

.venv\Scripts\python -m pytest -q        # all passed
.venv\Scripts\python -m tools.preflight  # preflight: 0 FAIL, 1 WARN (models_optional/ không có)
```

## 3. Bằng chứng

Nộp: `reports/review_queue.csv`. Sinh lại được: `out/poc/correction_diff.csv`, `out/poc/audit_log.jsonl`
(30 dòng `actor = long`), `out/poc/decision.json`.

| Mục | Giá trị |
|---|---|
| Queue | 30 box top suspicion score (B = 30), 0 trùng `item_id`, cùng thứ tự với `out/poc/review_queue.csv` |
| Quyết định | **approve 10 · edit 17 · reject 3** · escalate 0 · skip 0 |
| Theo split | dev: approve 5 · edit 9 · reject 2 · eval: approve 5 · edit 8 · reject 1 |
| Queue chọn trúng | 20/30 box có lỗi (edit + reject). 10 box approve = model đúng nhưng queue nghi oan |
| Edit | 17/17 giữ class, chỉ vẽ lại khung. 10/17 là box nhỏ (< 150 px²). IoU khung mới vs box model: min 0.53 · median 0.72 · max 0.86 |
| Reject | `aa_38#1` pedestrian (box 2.0 × 58.1 px, gần như một đường kẻ) · `aa_50#3` pedestrian · `aa_32#4` sign (48.0 × 16.5 px, dẹt bất thường). Note: không có đối tượng trong box |
| Sau review | `GO / REVIEWED_AND_VALID`: 224 box export (227 − 3 reject), invalid-label rate 0.0, round-trip YOLO/COCO ok, review_fraction 1.0 |
| Thời gian duyệt | 17 phút 45 giây / 30 box ≈ 35 giây/box (tự bấm giờ; `reviewed_at` trong CSV chỉ là giờ export) |

**3 box khó nhất:**

| item_id | Quyết định | Vì sao khó |
|---|---|---|
| `aa_09#4` | edit (pedestrian) | Người đi bộ nhỏ ở xa, mép khung khó xác định; khung vẽ lại chỉ trùng 0.53 IoU với box model |
| `aa_22#3` | edit (pedestrian) | Như trên: pedestrian nhỏ, IoU 0.53. Lệch vài pixel có thể đổi kết quả match ở ngưỡng 0.5 |
| `aa_38#1` | reject | Box model rộng 2.0 px, cao 58.1 px, gần như một đường kẻ; phải zoom để chắc chắn không có người bên trong |

## 4. Hạn chế

- Chỉ review 30/234 box; box sai ngoài queue không ai thấy, cần QA ngẫu nhiên ngoài queue.
- Chỉ 1 reviewer nên không đo được độ đồng thuận. Box nhỏ (10/17 edit) khó nhìn ở ảnh 320×180, khung vẽ lại có sai số vài pixel.
- 7/17 edit có IoU < 0.7 so với box model; nếu GT gần khung vẽ lại thì đây là chỗ review tăng TP nhiều nhất. Chưa có `gt/` nên chưa có F1 chính thức.
- Object bị sót không thêm được qua trang review (cần `review_additions.csv`). Lần review này không ghi object sót nào.
- Prediction là mô phỏng (`simulated-detector-fixture`), không phải inference YOLO26 thật.
