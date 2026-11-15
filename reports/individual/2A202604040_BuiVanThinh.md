# Báo cáo cá nhân — Lab 06 · Field-to-Release PoC
**Học viên:** Bùi Văn Thịnh  
**MSSV:** 2A202604040  
**Nhóm:** `sIx4nhd0cthAn`  
**Track:** T14 · Auto-Annotation + Human QC Exporter  
**Nhiệm vụ:** Issue #12 & Task E-01 (kèm Task H) · Unit Test `decide()` REWORK, Failure Evidence, Decision Log, Hoàn thiện Báo cáo nhóm & Kịch bản Demo  

---

## 1. Việc đã làm

- **Đảm nhận Issue #12, Task E-01 (`TASKS.md`) & Task H**: Chịu trách nhiệm bổ sung unit test cho nhánh `REWORK / EXPORT_QA_FAILED` của `src/pipeline.py::decide`, lập nhật ký quyết định `reports/decision_log.csv`, phân tích và trích xuất ảnh bằng chứng cho 2 failure cases trong `reports/figures/`, hoàn thiện toàn bộ `reports/group_report.md` (Mục 00–11) và bộ tài liệu báo cáo/demo (`docs/report.md`, `docs/demo.md`, `docs/contributions.md`).
- **Phát triển Unit Test cho nhánh `REWORK / EXPORT_QA_FAILED` trong `tests/test_t14.py`**:
  - Trực tiếp kiểm thử hàm `decide()` trong `src/pipeline.py` với 4 kịch bản lỗi export QA: `n_invalid > 0`, `yolo_ok == False`, `coco_ok == False`, và kết hợp cả hai.
  - Kiểm tra độ ưu tiên tối cao (precedence) của `REWORK`: Ngay cả khi chưa review (`n_reviewed == 0`) hoặc đã review 100% (`n_reviewed == queue_size`), nếu có lỗi QA format thì pipeline bắt buộc trả về `REWORK`, ngăn chặn triệt để dữ liệu lỗi lọt vào release.
  - Toàn bộ test suite `tests/test_t14.py` đạt 26/26 passed (100%), và toàn bộ repo đạt 87 passed, 5 skipped (0 failed).
- **Phân tích Failure Case 1 (Lỗi Map ID sai / Lệch nhãn COCO sang Taxonomy)**:
  - Khảo sát sự không khớp schema giữa mã lớp COCO trong `model_card_fixture.yaml` (`0: person, 2: car, 3: motorcycle, 11: stop sign`) và `taxonomy.yaml` (`0: car, 1: pedestrian, 2: motorbike, 3: sign`).
  - Phân tích chi tiết trường hợp box `aa_00#0`, `aa_00#1`, `aa_00#2` trong ảnh `images/aa_00.jpg`, chứng minh nếu dùng ánh xạ ngây thơ (naive mapping) thì 100% (234/234) box sẽ bị gán sai nhãn dù file xuất ra vẫn hợp lệ về format kỹ thuật.
  - Phân tích hiện tượng nhầm lẫn giữa `pedestrian` và `motorbike` ở cự ly xa (ví dụ `aa_05#3`).
- **Phân tích Failure Case 2 (Mô hình tự tin ở vật thể lớn nhưng bỏ sót vật thể nhỏ < 150 px²)**:
  - Khảo sát ảnh `images/aa_08.jpg` và `images/aa_02.jpg`. Chứng minh mô hình phát hiện rất tự tin các ô tô lớn ở gần (conf > 0.75 - 0.90), nhưng người đi bộ nhỏ ở xa (`aa_08#3`, diện tích 99.8 px²) chỉ đạt conf 0.297, suýt soát bị loại bỏ; các vật thể nhỏ khác (như xe máy `aa_02#4`, conf 0.118) bị loại thẳng tay bởi ngưỡng cứng `conf_threshold = 0.25`.
- **Trích xuất bằng chứng trực quan (`reports/figures/`)**: Tạo 4 ảnh minh chứng trực quan lưu tại `reports/figures/` (`case1_label_map_aa_00.png`, `case1_confusion_aa_05.png`, `case2_small_objects_aa_08.png`, `case2_small_sign_aa_02.png`), thể hiện bounding box, độ tin cậy và các vật thể nhỏ bị ảnh hưởng.
- **Xây dựng Decision Log (`GO`), Báo cáo nhóm & Tài liệu Demo (`Task E-01`)**:
  - Cập nhật `owner: "Bui Van Thinh (Lead)"` và `group_id: G01` trong `configs/lab.yaml` và `submission/README.md`.
  - Áp dụng tệp kết quả review 30/30 box của Trần Nguyễn Phi Long (`reports/review_queue.csv` → `out/poc/review_queue_reviewed.csv`: approve 10, edit 17, reject 3) vào pipeline PoC, xác nhận quyết định chuyển từ `HOLD / NOT_REVIEWED` sang **`GO / REVIEWED_AND_VALID`** (224 box export, `n_invalid = 0`, `review_fraction = 1.0`).
  - Khởi tạo `reports/decision_log.csv` chuẩn 8 cột theo RFC dự án, ghi nhận đầy đủ lịch sử quyết định từ Baseline (`HOLD / BASELINE_ONLY`), PoC trước review (`HOLD / NOT_REVIEWED`) đến PoC sau review (`GO / REVIEWED_AND_VALID`) với owner `Bui Van Thinh (Lead)`.
  - Hoàn thiện toàn bộ các Mục 00–11 trong `reports/group_report.md`, loại bỏ 100% các ô `⟦TODO⟧`, đồng bộ bảng đóng góp nhóm và checklist bàn giao.
  - Biên soạn đầy đủ 3 tài liệu của Task `E-01`: `docs/report.md` (Báo cáo tổng hợp + sơ đồ kiến trúc Mermaid), `docs/demo.md` (Kịch bản demo + dàn ý slide 8–10 phút), `docs/contributions.md` (Bảng tổng hợp đóng góp Git).
  - Cập nhật dòng đóng góp trong `reports/team_contributions.csv`, `TASKS.md` (`E-01` → `DONE`) và `PROGRESS.md`.

---

## 2. Lệnh đã chạy

### a. Chạy công cụ kiểm định bằng chứng lỗi của nhóm
```powershell
.venv\Scripts\python.exe -m tools.t14_failure_evidence
```
*Output ghi nhận:*
- Baseline xuất 227 box, có 3 box bị lỗi tọa độ (out of bounds / negative origin: `aa_37`, `aa_40`, `aa_56`).
- Ánh xạ ngây thơ `class_id -> taxonomy` làm sai nhãn 234/234 box (trong đó 29 box `stop sign` bị bỏ rơi do taxonomy chỉ có id 0-3).
- Có 18/227 box là vật thể nhỏ (< 150 px²), trung vị conf của vật thể nhỏ chỉ là 0.38; số vật thể nhỏ có conf >= 0.8 là 0 box.

### b. Chạy kiểm chứng Baseline và PoC Pipeline
```powershell
# Chạy Baseline Pipeline
.venv\Scripts\python.exe -m src.main --config configs/lab.yaml --track t14 --pipeline src.baseline_t14 --output out/baseline --run-name baseline --offline --seed 42

# Chạy PoC Pipeline
.venv\Scripts\python.exe -m src.main --config configs/lab.yaml --track t14 --output out/poc --run-name poc --offline --seed 42
```
*Output ghi nhận:* Cả hai lệnh đều kết thúc với mã `status=0` (thành công 100%), runtime Baseline 0.0299s, PoC 0.7344s, sinh đầy đủ `decision.json`, `qa_report.json`, `run_info.json`.

### c. Chạy kiểm tra chất lượng unit test và kiểm định preflight
```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m tools.preflight
```
*Output ghi nhận:*
- `pytest -q`: `26 passed` cho riêng `tests/test_t14.py` (bao gồm 5 unit tests mới cho `decide()`); `87 passed, 5 skipped` cho toàn bộ test suite trên repo.
- `tools.preflight`: `0 FAIL, 1 WARN` (`models_optional/` không bắt buộc), mã thoát 0.

---

## 3. Bằng chứng

### Bằng chứng 1 · Case 1: Lỗi Map ID sai / Lệch nhãn COCO sang Taxonomy
* **Vị trí file ảnh gốc:** `fixtures/t14_auto_annotation/images/aa_00.jpg`
* **Ảnh minh chứng đã lưu:** `reports/figures/case1_label_map_aa_00.png`
* **Item ID & Metadata:**
  * Box `aa_00#0`: Tọa độ `x=88.91, y=102.9, w=39.3, h=25.53`, model `class_id = 2` (`car` trong COCO). Nếu map ngây thơ theo chỉ số taxonomy (`0: car, 1: pedestrian, 2: motorbike`), box này sẽ bị gán thành `motorbike`.
  * Box `aa_00#1`: Tọa độ `x=53.28, y=88.86, w=10.99, h=30.67`, model `class_id = 0` (`person` trong COCO). Nếu map ngây thơ, box này bị biến thành `car`.
  * Box `aa_00#2`: Tọa độ `x=278.22, y=37.94, w=18.12, h=16.93`, model `class_id = 11` (`stop sign` trong COCO). Bị mất hoàn toàn do taxonomy không có index 11.
* **Root cause:** Không đồng nhất schema giữa nguồn huấn luyện (COCO 80 classes) và bài toán nghiệp vụ (Taxonomy 4 classes). Baseline không có validator sẽ âm thầm xuất dữ liệu sai hoàn toàn khiến độ chính xác F1 rớt về 0.
* **Khắc phục trong PoC:** Module `src/t14_tools.py` áp dụng từ điển bí danh bắt buộc `DEFAULT_ALIAS` và hàm `validate_predictions()`, bắt lỗi rào chắn nếu có class không hợp lệ. Hàng đợi review lọc được các box có độ bất định cao để con người sửa nhãn (`edit`).

### Bằng chứng 2 · Case 2: Tự tin ở vật thể lớn nhưng bỏ sót vật thể nhỏ (< 150 px²)
* **Vị trí file ảnh gốc:** `fixtures/t14_auto_annotation/images/aa_08.jpg`
* **Ảnh minh chứng đã lưu:** `reports/figures/case2_small_objects_aa_08.png`
* **Item ID & Metadata:**
  * Các box lớn được phát hiện với conf cao: `aa_08#0` (car, conf=0.754, area=1224.7 px²), `aa_08#1` (car, conf=0.896, area=1448.2 px²), `aa_08#2` (car, conf=0.832, area=2502.9 px²), `aa_08#4` (motorbike, conf=0.783, area=716.4 px²).
  * Vật thể nhỏ bị yếu độ tin cậy: `aa_08#3` (người đi bộ nhỏ ở xa, `x=132.4, y=68.5, w=6.3, h=15.9`, diện tích chỉ `99.8 px²`). Độ tin cậy của model bị sụt giảm nghiêm trọng xuống `conf = 0.297`, mấp mé ngưỡng cắt `0.25`.
  * Trường hợp ảnh `aa_02.jpg` (`reports/figures/case2_small_sign_aa_02.png`): Box xe máy ở xa `aa_02#4` có `conf = 0.118` bị baseline loại bỏ hoàn toàn, gây ra lỗi bỏ sót nghiêm trọng (False Negative).
* **Root cause:** Các mạng nơ-ron tích chập (CNN) bị mất thông tin phân giải không gian qua các tầng pooling và downsampling, khiến các đặc trưng của vật thể nhỏ ở cự ly xa bị mờ nhạt. Việc áp dụng cùng một ngưỡng lọc tĩnh (`conf_threshold = 0.25`) gây bất lợi lớn cho vật thể nhỏ.
* **Khắc phục trong PoC:** PoC chấm điểm ưu tiên review cộng thêm trọng số `w_small = 0.3` cho vật thể có diện tích < 150 px², đưa `aa_08#3` (rank 14, score 0.643) và `aa_02#3` (rank 1, score 0.783) lên đầu hàng đợi `review_queue.csv` để con người cứu lại (`approve`). Đồng thời cho phép nhập `review_additions.csv` để vẽ bổ sung các vật thể bị sót 100%.

---

## 4. Hạn chế / Chưa xong

- **Tính chất dữ liệu giả lập (Synthetic Fixture):** Dữ liệu `fixtures/t14_auto_annotation/` là mô phỏng, phân phối hình học và độ sắc nét chưa phản ánh toàn diện các điều kiện thời tiết phức tạp ngoài đời thực (mưa lớn, sương mù dày, chói sáng ban đêm).
- **Phụ thuộc vào ngân sách Human Review:** Ngân sách review bị giới hạn $B=30$ box. Nếu một lô dữ liệu có tới hàng trăm vật thể nhỏ ở xa, hàng đợi review top-30 sẽ không thể bao phủ hết, vẫn tiềm ẩn nguy cơ lọt lưới các vật thể nhỏ chưa được gắn nhãn.
- **Rủi ro đạo đức và an toàn sinh mạng:** Trong hệ thống xe tự hành, việc bỏ sót người đi bộ nhỏ ở xa nguy hiểm hơn rất nhiều so với việc nhận diện nhầm một biển báo. Pipeline cần được nâng cấp áp dụng chi phí mất mát bất đối xứng (asymmetric loss / risk-weighted evaluation) trước khi triển khai thực tế.
- **Phần chưa xong (chờ repo giảng viên):** Các chỉ số F1 chính thức và kiểm định bootstrap CI 95% chưa thể chạy được do repo không phân phối tập nhãn chuẩn `gt/` (giảng viên sẽ overlay khi chấm thi).
