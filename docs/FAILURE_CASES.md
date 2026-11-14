# Failure Cases & License Compliance Audit — Day06 T07

> **Owner:** Hoàng Quốc Chung (Redaction, Failure Analysis & License)  
> **Scope:** `src/redaction/`, `configs/tracks/t07.yaml`, `fixtures/t07_redaction/`, `data/manifests/`  
> **Contract:** `PII Miss Rate` (primary, `min_coverage = 0.90`) with guardrail `over_redaction_rate <= 0.30` (`max_over_redaction: 0.3`).

---

## 1. Failure Modes Analysis

Mỗi failure mode dưới đây được phân tích theo 4 trục: **Triệu chứng & Slice**, **Root Cause**, **Detection Method** (cách phát hiện trong pipeline/audit), và **Mitigation** (biện pháp giảm thiểu đã cài hoặc khuyến nghị).

| # | Failure Mode | Slice / Condition | Guardrail / Metric Impact | Primary Mitigation |
|---|---|---|---|---|
| 1 | Small faces in dark / low-light scenes | `small`, `dark` (`face`) | Tăng `PII Miss Rate` trên slice `small` & `dark` | CLAHE / gamma pre-check, `face_conf = 0.30`, `small_box_pad_ratio = 0.22` |
| 2 | Blurred / occluded license plates | `blur`, `occluded` (`plate`) | Tăng `PII Miss Rate` trên slice `blur` & `occluded` | Multi-scale / aspect-ratio plate fallback + padding + review queue |
| 3 | Over-redaction masking task context | `normal` / `small` (excessive bbox) | Vi phạm guardrail `max_over_redaction <= 0.30` | Tách `pad_ratio = 0.12` vs `small_box_pad_ratio = 0.22`, union mask |
| 4 | Zero-detection images still containing PII | All slices (false negative) | Miss rate = 1.0 trên ảnh không có detection | Đưa ảnh `num_detections == 0` vào top-15 `review_queue.csv` (`B = 15`) |
| 5 | Hard negatives: pedestrians seen from behind | Non-PII background (`head_back`) | Tăng `over_redaction_rate` nếu bị che nhầm | Lọc theo đặc trưng khuôn mặt (YuNet landmarks/score), human reject |

---

### 1.1. Small Faces in Dark / Low-Light Scenes (`small`, `dark` slices)

- **Triệu chứng:** Khuôn mặt ở xa camera (diện tích `< 900 px²`, tương ứng `small_box_area_thr: 900` trong `configs/tracks/t07.yaml`) hoặc nằm trong vùng thiếu sáng (`dark`) không được YuNet phát hiện hoặc có confidence thấp hơn ngưỡng mặc định, dẫn đến `coverage < 0.90` và bị tính là miss.
- **Root Cause:**
  1. `face_detection_yunet_2023mar.onnx` hoạt động kém dần khi độ phân giải mặt quá nhỏ (< 20×20 px) do đặc trưng mắt/mũi/miệng bị mất chi tiết sau downsampling.
  2. Độ tương phản thấp trong vùng tối làm giảm đáp ứng của backbone CNN, khiến confidence score tụt xuống dưới `0.30` hoặc không vượt qua NMS.
  3. Nếu bbox dự đoán hơi lệch tâm trên vùng mặt nhỏ, chỉ cần lệch 2–3 px là tỷ lệ phủ pixel (`coverage`) rớt xuống dưới `0.90`.
- **Detection Method:**
  - Theo dõi `PII Miss Rate` tách riêng theo slice `condition == "small"` và `condition == "dark"` trong `src.evaluate --track t07`.
  - Trong pipeline inference (không đọc `gt/`), gắn cờ các detection có `area < small_box_area_thr` (`900 px²`) hoặc `confidence ∈ [0.30, 0.55]` vào hàng chờ human review (`review_queue.csv`).
- **Mitigation:**
  1. Hạ ngưỡng phát hiện `face_conf: 0.30` để tăng recall trên vùng tối/nhỏ, đẩy các ca ranh giới sang human review thay vì bỏ sót im lặng.
  2. Áp dụng padding thích ứng: `small_box_pad_ratio: 0.22` cho box có diện tích `< 900 px²` (thay vì `pad_ratio: 0.12` cho box thường) để đảm bảo đủ `min_coverage >= 0.90`.
  3. Dùng `redact_mode: mask` (solid fill) làm mặc định để không phụ thuộc kernel size của Gaussian blur trên vùng nhỏ.

---

### 1.2. Blurred / Occluded License Plates (`blur`, `occluded` slices)

- **Triệu chứng:** Biển số xe bị motion blur (`blur`, ví dụ xe chạy nhanh hoặc camera CCTV rung) hoặc bị che khuất một phần bởi phương tiện khác, thanh chắn, bùn đất (`occluded`) bị `license_plate_detection_lpd_yunet_2023mar.onnx` bỏ sót hoặc chỉ bắt được một nửa biển số (`coverage < 0.90`).
- **Root Cause:**
  1. Motion blur làm nhòe biên cạnh chữ số và viền khung biển số, phá vỡ đặc trưng tần số cao mà LPD-YuNet dựa vào.
  2. Che khuất một phần làm thay đổi tỷ lệ khung hình (aspect ratio) chuẩn của biển số Việt Nam (biển dài ~4.2:1 hoặc biển vuông ~1.4:1), khiến anchor/head của detector giảm score mạnh.
  3. Trên ảnh giao thông thực tế (`640×640`), biển số ở xa chỉ rộng ~15–30 px; khi kết hợp thêm blur/occlusion, detector đơn lẻ rất dễ miss hoàn toàn.
- **Detection Method:**
  - Kiểm tra `PII Miss Rate` theo slice `condition == "blur"` và `condition == "occluded"` trong báo cáo đánh giá.
  - Theo dõi phân bố confidence của `plate` detections và số lượng ảnh có `plate_count == 0` dù có ngữ cảnh giao thông.
- **Mitigation:**
  1. Giữ `plate_conf: 0.30` kết hợp `pad_ratio: 0.12` (và `0.22` nếu `area < 900`) để mở rộng nhẹ vùng che phủ phần biển số bị mờ mép.
  2. Đưa các box biển số confidence thấp hoặc kích thước bất thường vào `review_queue.csv` (`budget: 15`) để reviewer vẽ bổ sung hoặc nới rộng bbox trên `contact_sheet.html`.
  3. (Hướng mở rộng) Kết hợp multi-scale inference hoặc color/edge heuristic fallback cho vùng nghi ngờ biển số khi LPD-YuNet không trả về kết quả.

---

### 1.3. Over-Redaction Masking Task-Relevant Context (`max_over_redaction <= 0.30`)

- **Triệu chứng:** Vùng che phủ lan quá rộng ra ngoài ranh giới khuôn mặt/biển số thực tế — hoặc chiến lược tầm thường như che/blur toàn bộ ảnh ("blur whole image") — làm mất thông tin ngữ cảnh phục vụ bài toán hạ nguồn (biển báo, làn đường, loại phương tiện, tín hiệu giao thông).
- **Root Cause:**
  1. Tăng `pad_ratio` quá mức trên toàn bộ detection (đặc biệt trên box lớn) làm diện tích pixel che ngoài GT tăng theo bình phương tỷ lệ nới rộng.
  2. Phát hiện nhầm (false positive) các vùng nền có kết cấu giống mặt/biển số (biển quảng cáo, đèn xe, cửa sổ) và che kín luôn các vùng này.
  3. Cộng dồn diện tích các box chồng lấn (double-counting) khi tính pixel coverage/over-redaction thay vì tính hợp (union) mặt nạ nhị phân.
- **Detection Method:**
  - Guardrail bắt buộc trong `src/metrics/tracks.py::eval_t07`: `over_redaction_rate = (masked_pixels - gt_pixels) / masked_pixels <= 0.30` (`max_over_redaction: 0.3`). Nếu vượt `0.30`, primary metric (`PII Miss Rate`) bị đánh dấu **FAIL guardrail** và không hợp lệ.
  - Tính diện tích che thực tế bằng union mask trên ảnh gốc trong `src/redaction/coverage.py` để tránh đếm đôi vùng giao nhau.
- **Mitigation:**
  1. Giới hạn `pad_ratio: 0.12` cho box kích thước bình thường và chỉ nâng lên `small_box_pad_ratio: 0.22` khi `area < 900 px²`.
  2. Render mask trực tiếp từ ảnh gốc local một lần duy nhất (không chồng blur nhiều lần).
  3. Cho phép reviewer bấm **Reject** các false positive trong `contact_sheet.html` trước khi re-export `redactions.csv` cuối cùng.

---

### 1.4. Zero-Detection Images That Still Contain PII (False Negatives)

- **Triệu chứng:** Pipeline trả về 0 detection (`num_detections == 0`) cho một ảnh, nhưng ảnh thực tế vẫn chứa khuôn mặt hoặc biển số (ví dụ: mặt quay nghiêng > 60°, biển số quá nhỏ/tối, hoặc nhiều điều kiện bất lợi kết hợp). Nếu không có cơ chế cảnh báo, ảnh này lọt qua hệ thống với 0 vùng được che.
- **Root Cause:**
  1. Ngưỡng `face_conf` / `plate_conf` (`0.30`) lọc bỏ toàn bộ candidate yếu trong ảnh khó.
  2. Hệ thống hàng chờ ngây thơ chỉ xếp hạng các detection có sẵn theo `min(confidence)` sẽ **bỏ qua hoàn toàn** ảnh có 0 detection vì không có box nào để tính score!
- **Detection Method:**
  - Kiểm tra tập ảnh đầu ra có `count(redactions) == 0` trong `redactions.csv`.
  - Đánh dấu cờ ưu tiên `zero_detection` khi xây dựng `review_queue.csv`.
- **Mitigation:**
  1. `src/review/queue.py` bắt buộc gán mức ưu tiên cao cho các ảnh **zero-detection** (cùng nhóm với `low-conf` và `small box`) trong giới hạn `budget = 15` ảnh/lượt duyệt.
  2. Trên `contact_sheet.html`, reviewer có thể dùng chức năng **Add box** trực tiếp lên ảnh zero-detection để bổ sung vùng PII bị sót, sau đó re-export `redactions.csv` và ghi nhận vào `audit_log.jsonl` / `audit_log.csv`.

---

### 1.5. Hard Negatives: Pedestrians Seen from Behind (Not PII per T07 GT)

- **Triệu chứng:** Người đi bộ hoặc người đi xe máy quay lưng lại phía camera (chỉ thấy sau đầu/tóc/mũ bảo hiểm, không lộ khuôn mặt) bị detector phụ (ví dụ Haar cascade tổng quát hoặc heuristic hình tròn/oval) nhận nhầm là khuôn mặt và bị che.
- **Root Cause:**
  1. Theo quy chuẩn Ground Truth của T07 (`docs/tracks/T07.md`), đầu người nhìn từ phía sau (`head_back`) **không phải PII** vì không chứa đặc điểm nhận dạng khuôn mặt.
  2. Các bộ dò dựa trên hình dạng đầu hoặc Haar fallback dễ nhầm vùng đầu tròn tối màu với khuôn mặt, làm tăng số false positive và đẩy `over_redaction_rate` tiến gần hoặc vượt ngưỡng `0.30`.
- **Detection Method:**
  - Quan sát phân bố `over_redaction_rate` và kiểm tra trực quan các ảnh bị che trên `contact_sheet.html` đối với nhóm đối tượng người đi bộ quay lưng.
- **Mitigation:**
  1. Ưu tiên `face_detection_yunet_2023mar.onnx` (có dự đoán 5 facial landmarks: 2 mắt, mũi, 2 khóe miệng) làm detector chính, giúp phân biệt mặt thật với sau gáy hiệu quả hơn Haar cascade.
  2. Đưa các candidate confidence thấp (`0.30–0.50`) vào `review_queue.csv` để human reviewer loại bỏ (`reject`) nếu là người quay lưng.

---

## 2. License & Governance Audit

### 2.1. Model & Dataset License Inventory

| Asset | Type | Source / Author | License | Offline / Local Policy | Attribution & Compliance Notes |
|---|---|---|---|---|---|
| `face_detection_yunet_2023mar.onnx` | Pretrained Model | OpenCV Zoo (`opencv_zoo`) | **MIT** | Pre-bundled in `models/`, CPU offline 100%, no runtime download | Cho phép sử dụng thương mại/nghiên cứu, sửa đổi và phân phối lại; giữ nguyên thông báo bản quyền & license MIT khi phân phối. |
| `license_plate_detection_lpd_yunet_2023mar.onnx` | Pretrained Model | OpenCV Zoo (`opencv_zoo`) | **Apache-2.0** | Pre-bundled in `models/`, CPU offline 100%, no runtime download | Cho phép sử dụng thương mại/nghiên cứu; bắt buộc giữ nguyên thông báo bản quyền, file `LICENSE` Apache-2.0 và ghi chú nếu có chỉnh sửa model. |
| **Vietnam License Plate** | External Dataset | Roboflow Universe | **CC BY 4.0** | Local only (`data/real/`, gitignored), manifest SHA-256 committed | Bắt buộc **Attribution** (ghi rõ nguồn Roboflow & tác giả dataset) trong `docs/DATA_AUDIT.md` và báo cáo; không commit ảnh gốc chứa biển số thật lên repo public. |
| **Face Detection Dataset** | External Dataset | Roboflow Universe | **CC BY 4.0** | Local only (`data/real/`, gitignored), manifest SHA-256 committed | Bắt buộc **Attribution** (ghi rõ nguồn Roboflow & tác giả dataset); không commit ảnh gốc chứa khuôn mặt thật lên repo public. |
| `fixtures/t07_redaction/` | Synthetic Fixture | Day06 Course Fixture | **CC0** | Committed in repo (`60` synthetic images `640×360`) | Không chứa PII thật; `gt/gt_regions.csv` tách biệt cho evaluator (`gt_guard` chặn pipeline đọc `gt/`). |

### 2.2. Governance & Privacy Compliance (EDPB Guidelines 02/2026)

1. **Redacted / Pseudonymised $\neq$ Fully Anonymised:**
   - Theo **EDPB Guidelines 02/2026** về ẩn danh hóa (*anonymisation*), việc làm mờ (`blur`) hoặc che khối (`mask`) khuôn mặt và biển số xe **chưa đủ** để khẳng định tập dữ liệu đầu ra không còn là dữ liệu cá nhân.
   - Dữ liệu ảnh giao thông vẫn có thể tái nhận dạng cá nhân gián tiếp thông qua các yếu tố ngữ cảnh còn lại: **dấu thời gian (timestamp), tọa độ/địa điểm quay (location), dáng người, trang phục, màu sắc/nhãn hiệu/đặc điểm riêng của phương tiện, hoặc chuỗi di chuyển liên tiếp**.
   - **Quy ước thuật ngữ bắt buộc:** Mọi tài liệu, log và báo cáo của nhóm phải gọi ảnh đầu ra là **`redacted`** hoặc **`pseudonymised`**, tuyệt đối **không** khẳng định là **`fully anonymised`**.
2. **Bảo vệ dữ liệu gốc (Raw PII Handling):**
   - Toàn bộ ảnh thật (`data/real/`) phải lưu local và nằm trong `.gitignore`; chỉ commit manifest SHA-256 (`data/manifests/`) và ảnh đã che (`redacted`) phục vụ demo.
   - Pipeline chạy **100% offline trên CPU** (`python -m src.main --offline --seed 42`), nghiêm cấm gửi ảnh raw PII lên API đám mây bên thứ ba.
   - Mọi thao tác chỉnh sửa vùng che của con người đều được truy vết qua `audit_log.jsonl` và `audit_log.csv` để đảm bảo tính giải trình (accountability).
