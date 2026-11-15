# Group Report — Day 06 Data Lab (Field-to-Release PoC)

## 00 · Header

| Mục | Giá trị |
|---|---|
| Group ID | G01 · tên nhóm `sIx4nhd0cthAn` |
| Challenge | T14 · Auto-Annotation + Human QC Exporter |
| Thành viên + vai trò | 1. Trần Thành Hùng (2A202603247) · Metric contract, Baseline & Tổng hợp Eval (#10)<br>2. Hoàng Quốc Chung (2A202604039) · Failure Analysis & License AGPL-3.0 (#9)<br>3. Nguyễn Xuân Việt (2A202604045) · Code, Test (`pytest`), Export & Eval (#8)<br>4. Trần Nguyễn Phi Long (2A202604058) · Chấm tay 30 ảnh/box trên web review (#7)<br>5. Phạm Phú Nguyễn Hưng (2A202604062) · Trang web review, Tích hợp & Hồ sơ nhóm (#11)<br>6. Bùi Văn Thịnh (2A202604040) · Failure analysis (Task H), Đạo đức & An toàn AI, Decision Log & Hoàn thiện báo cáo (#12) |
| Git URL | https://github.com/phamphunguyenhung/K5-Day6-L2-sIx4nhd0cthAn-DataGovernanceAndPrivacy |
| Tag `demo-180m` (commit) | `demo-180m` (tạo bằng `make tag-demo` khi freeze demo) |
| Tag `final-24h` (commit) | `final-24h` (tạo bằng `make tag-final` khi chốt nộp) |
| Deadline (giờ VN) | 2026-11-15 12:00:00 (GMT+7) |

## 01 · Problem statement

- **Ai bị ảnh hưởng khi hệ thống sai?**: Hệ thống nhận thức xe tự hành (autonomous vehicle perception) và các mô hình downstream phụ thuộc hoàn toàn vào chất lượng nhãn gán tự động. Khi hệ thống gán nhãn tự động phát sinh lỗi (mislabeling, bounding box lệch tọa độ, bỏ sót đối tượng yếu thế như người đi bộ nhỏ ở xa hoặc xe máy), mô hình huấn luyện phía sau sẽ bị trôi dạt phân phối (distribution shift), dẫn đến rủi ro va chạm nghiêm trọng đe dọa tính mạng con người và vi phạm các quy chuẩn an toàn đường bộ.
- **Điểm nghẽn**: Gán nhãn thủ công 100% tốn kém thời gian và chi phí; model phát hiện có sẵn dùng label map COCO (khác biệt sâu sắc với taxonomy dự án: `car`, `pedestrian`, `motorbike`, `sign`), dễ bỏ sót vật thể nhỏ ở cự ly xa do pooling và tạo ra nhiều false positive ở ngưỡng conf thấp.
- **Đầu vào → đầu ra**: `fixtures/t14_auto_annotation` (60 ảnh + prediction mô phỏng) → `predictions.csv` (`image_id, class, x, y, w, h, conf`) + YOLO (`data.yaml`, `labels/`) + COCO (`coco.json`) + `decision.json`.
- **Tiêu chí GO / HOLD / REWORK** (chính sách trong `src/pipeline.py::decide`, đóng băng trước khi đánh giá):
  - **REWORK** `EXPORT_QA_FAILED`: Có nhãn không hợp lệ (`n_invalid > 0`) hoặc round-trip kiểm tra ngược định dạng YOLO/COCO không khớp (`not (yolo_ok and coco_ok)`).
  - **HOLD** `NOT_REVIEWED` (chưa review) / `REVIEW_INCOMPLETE` (tỷ lệ duyệt < `min_review_fraction` = 0.8 của queue).
  - **GO** `REVIEWED_AND_VALID`: Đã review đủ $\ge 80\%$ hàng đợi và xuất nhãn đạt chuẩn kiểm định 100%. Lưu ý: GO chỉ xác nhận định dạng hợp lệ và tuân thủ quy trình kiểm soát, không khẳng định độ chính xác mô hình (accuracy) vì repo không chứa ground truth ẩn.
- **Giả định cần domain owner xác nhận**: Bảng ánh xạ bí danh (alias mapping): `person → pedestrian`, `motorcycle → motorbike`, `stop sign → sign`, `car → car`. Mọi nhãn lớp nằm ngoài taxonomy sẽ bị từ chối dứt khoát. Ngưỡng vật thể nhỏ (small object threshold): diện tích bounding box $w \times h < 150 \text{ px}^2$, cần cơ chế bù trọng số nghi vấn (`w_small = 0.3`) để tránh bị triệt tiêu bởi ngưỡng lọc tĩnh.

## 02 · Data & provenance

| Split | N | Nguồn / link | License | Synthetic? |
|---|---|---|---|---|
| dev | 40 ảnh (234 predictions) | `fixtures/t14_auto_annotation` | Synthetic Data Benchmark (VinUni AI20k Lab License) | có |
| eval / heldout | 20 ảnh | `fixtures/t14_auto_annotation` (GT ẩn chỉ có tại repo giảng viên) | Synthetic Data Benchmark (VinUni AI20k Lab License) | có |

- **Schema & class mapping**: Model `0: person, 2: car, 3: motorcycle, 11: stop sign` → taxonomy `0: car, 1: pedestrian, 2: motorbike, 3: sign` (`src/t14_tools.py::build_class_map`; lớp mô hình không map được sẽ báo `ValueError` rõ ràng và từ chối xuất dữ liệu).
- **Seed**: 42 · Split: Phân chia cố định theo `images.csv` (40 ảnh dev, 20 ảnh eval) đảm bảo tính tái lập 100%.
- **`manifest.csv` verified (sha256)**: Đạt `OK` (toàn bộ mã hash SHA-256 của các file fixture khớp tuyệt đối khi kiểm tra qua `python -m tools.preflight`); `strict_provenance: true`.
- **Dữ liệu thật**: Không sử dụng trong vòng PoC này (thư mục `data/real/` được loại trừ trong `.gitignore`). Prediction là **mô phỏng** (`simulated-detector-fixture`, `fixture-1`), không khai báo là inference thực tế.

## 03 · Research & design

- **References**: Tham khảo chuẩn kiểm định CVAT Auto-Annotation QA, đặc tả định dạng YOLOv8 của Ultralytics, và COCO Data Format Specification (chi tiết tại Mục 11).
- **Data flow** (`src/pipeline.py`): `load` (kiểm tra hash `manifest.csv`) → `process` (chuẩn hóa tọa độ, clip box tràn biên, lọc `conf >= 0.25`, tính `suspicion score`) → `rank` (hàng đợi duyệt top $B = 30$ box nghi vấn nhất + xuất `contact_sheet.html`) → `review` (áp dụng quyết định duyệt từ `review_queue_reviewed.csv` và `review_additions.csv`, ghi `audit_log.jsonl`) → `export` (kiểm định `validate_predictions`, xuất CSV/YOLO/COCO, round-trip check) → `decide` (REWORK/HOLD/GO).
- **Suspicion score**: $S = 0.5 \cdot \text{uncertainty} + 0.3 \cdot \text{small} + 0.2 \cdot \text{clipped} + 0.3 \cdot \text{class\_confusion}$. Trong đó: $\text{uncertainty} = \text{clip}(\max(1 - 2|conf - 0.5|, 1 - |conf - thr|/0.15), 0, 1)$; $\text{small} = (w \times h < 150 \text{ px}^2)$; $\text{confusion} = (\text{IoU} \ge 0.5 \text{ với box khác class cùng ảnh})$. Tham số được đóng băng trước khi đánh giá.
- **Bằng chứng queue ưu tiên đúng box khó** (`out/poc/boxes_all.csv`): Hàng đợi 30 box khó nhất đã tóm bắt được 18/19 box nhỏ (chiếm 60.0% queue, trong khi toàn bộ tập dự đoán chỉ có 8.1% box nhỏ) và 23 box có confidence mấp mé 0.35–0.65 (76.7% queue). Điểm nghi vấn giảm dần từ 0.783 xuống 0.473, không có item trùng lặp.
- **Thành phần offline**: Pipeline chạy offline 100% (`--offline`), mọi kết nối mạng socket ra ngoài bị chặn bởi `offline_guard`. Không sử dụng trọng số ngoài chưa cấp phép.
- **Trade-offs**: Budget $B = 30$ trên 227 box export (chiếm 13.2%), 197 box còn lại không được duyệt tay. Hàng đợi chỉ xếp hạng box mà model đã đề xuất, không tự động phát hiện được box bỏ sót hoàn toàn (phải dựa vào human reviewer bổ sung qua `review_additions.csv`). Ngưỡng cắt 0.25 loại 7 box có conf thấp.

## 04 · Baseline vs PoC

- **Baseline** (`src/baseline_t14.py`): Ánh xạ class, lọc conf $\ge 0.25$, xuất nhãn không cắt box viền, không xếp hạng nghi vấn, không có hàng đợi duyệt tay. Quyết định luôn là `HOLD / BASELINE_ONLY`.
- **PoC** (`src/pipeline.py`): Thêm tự động clip box tràn biên, xếp hạng nghi vấn suspicion score, quy trình human-in-the-loop review, kiểm định format đa tầng và round-trip hai chiều YOLO/COCO. Chạy với config hash `b40ac923fc9f`:
  - Trước review (`n_reviewed = 0`): 234 box model → 227 export, 3 box bị tràn biên được clip tự động về trong ảnh, 0 nhãn không hợp lệ, quyết định `HOLD / NOT_REVIEWED`.
  - Sau khi áp dụng kết quả review 30/30 box của Trần Nguyễn Phi Long (`reports/review_queue.csv` → `out/poc/review_queue_reviewed.csv`: approve 10, edit 17, reject 3): xuất 224 box hợp lệ (`n_invalid = 0`), round-trip YOLO max sai số $0.00024 \text{ px}$ / COCO $0.0 \text{ px}$ (dung sai $0.05 \text{ px}$), `review_fraction = 1.0`, quyết định chính thức đạt **`GO / REVIEWED_AND_VALID`**.
- **Cùng điều kiện đánh giá**: Cả hai run dùng `configs/lab.yaml`, `--seed 42 --offline`, `conf_threshold: 0.25`, cùng 60 ảnh fixture.

**Bảng 1 · So sánh Baseline vs PoC**:

| Metric (unit) | Baseline | PoC | Δ & chú thích |
|---|---|---|---|
| Primary: Detection F1@IoU0.5 (box, 20 ảnh eval) | N/A (chờ GT giảng viên) | N/A (chờ GT giảng viên) | Bộ test ẩn từ giảng viên |
| Hard-slice: recall small object (< 150 px²) | N/A (chờ GT giảng viên) | N/A (chờ GT giảng viên) | Bù đắp bởi trọng số w_small = 0.3 |
| Review@B / miss / FP | 0 / N/A | 30 / 18 small / 23 mid-conf | B=30 tập trung đúng nhóm rủi ro cao |
| Invalid-label rate sau export | 0.0132 (3/227) | 0.0 (0/227) | -0.0132 (Đạt chuẩn 0 lỗi; PoC tự động clip 3 box tràn ảnh) |
| Runtime / 100 items | 0.0128 s | 0.3138 s | Baseline 0.0299 s cả run; PoC 0.7344 s cả run (Intel64, Windows 11, Python 3.14.3) |

## 05 · Evaluation protocol

- **Primary metric**: Detection F1 = $2PR/(P+R)$, trong đó $P = TP/(TP+FP)$, $R = TP/(TP+FN)$; đơn vị tính theo bounding box; thuật toán ghép greedy IoU $\ge 0.5$ và đúng class taxonomy (`src/metrics/detection.py`). Chi tiết tại `reports/metric_contract.md`.
- **Ngưỡng cố định**: Threshold 0.25, budget $B = 30$ (`configs/tracks/t14.yaml`); được chốt vào lúc 2026-11-14 10:00:00 (GMT+7) bởi Bùi Văn Thịnh (Lead) và Trần Thành Hùng (Metric contract owner).
- **Độ bất định**: $N = 20$ ảnh eval, bootstrap ước lượng khoảng tin cậy 95% theo ảnh sẽ được chạy tự động khi giảng viên overlay thư mục `gt/`.
- **Lệnh evaluate chính thức** (giảng viên chạy với overlay `gt/`):
  `python -m src.evaluate --track t14 --pred out/poc/predictions.csv --run-name poc --out reports/metrics_poc.json`

## 06 · Result slices

| Slice | N (predicted) | Baseline | PoC | Ghi chú |
|---|---|---|---|---|
| class = car | 108 | 106 valid (2 clipped) | 108 valid (0 invalid) | PoC tự động clip box `aa_40`, `aa_56` |
| class = pedestrian | 61 | 60 valid (1 negative) | 61 valid (0 invalid) | PoC tự động chuẩn hóa tọa độ âm `aa_37` |
| class = motorbike | 36 | 36 valid | 36 valid | Khớp taxonomy 100% qua alias map |
| class = sign | 29 | 29 valid | 29 valid | Lớp `stop sign` map sang `sign` an toàn |
| small object (< 150 px²) | 18 | 18 exported (0 reviewed) | 18 exported (18 in review queue) | 100% box nhỏ được đẩy vào review queue |

- **Failure case 1 — class ID sai vẫn export hợp lệ**: Phát hiện trên ảnh `images/aa_00.jpg` (box `aa_00#0`, `aa_00#1`). File dự đoán thô dùng COCO class ID (`0: person, 2: car, 3: motorcycle, 11: stop sign`), khác với taxonomy dự án (`0: car, 1: pedestrian, 2: motorbike, 3: sign`). Ánh xạ ngây thơ làm sai nhãn 234/234 box (100%), ô tô bị gán thành xe máy, người bị gán thành ô tô, nhưng xuất file vẫn đúng định dạng hợp lệ. Root cause: Lệch schema giữa pretrain detector và target domain mà không có validator. Khắc phục: `src/t14_tools.py` áp dụng từ điển bí danh `DEFAULT_ALIAS` và hàm `validate_predictions()`; hàng đợi review phát hiện các box nhầm lẫn class để reviewer sửa (`reports/figures/case1_label_map_aa_00.png`).
- **Failure case 2 — conf cao nhưng thiếu vật thể nhỏ**: Phát hiện trên ảnh `images/aa_08.jpg` và `images/aa_02.jpg`. Model tự tin cao ở các ô tô lớn ở gần (`aa_08#0`, `aa_08#1`, conf > 0.75 - 0.89), nhưng người đi bộ nhỏ ở xa (`aa_08#3`, area = 99.8 px²) chỉ đạt conf 0.297, mấp mé ngưỡng cắt 0.25; xe máy ở xa trong `aa_02.jpg` (conf = 0.118) bị baseline loại bỏ hoàn toàn (False Negative). Root cause: CNN pooling làm suy giảm đặc trưng không gian ở cự ly xa, việc dùng ngưỡng cứng tĩnh 0.25 triệt tiêu các box nhỏ. Khắc phục: PoC cộng trọng số `w_small = 0.3` đẩy các box nhỏ lên top review (`review_queue.csv`); hỗ trợ bổ sung vật thể bị sót 100% qua `review_additions.csv` (`reports/figures/case2_small_objects_aa_08.png`).
- **Latency / compute**:
  Đo từ `out/poc/run_info.json`:
  - Môi trường: Windows 11 (build 10.0.26200), Python 3.14.3.
  - Vi xử lý: Intel64 Family 6 Model 142 Stepping 10 (GenuineIntel).
  - Runtime tổng thể: 0.7344 giây cho toàn bộ pipeline PoC (bao gồm xử lý ảnh, tính toán suspicion score, ranking top 30, export CSV/YOLO/COCO và round-trip validation).
  - Độ trễ trung bình: 18.36 ms / ảnh (tương đương 3.14 ms / box dự đoán).

## 07 · Human & decision

- **Review queue**: `out/poc/review_queue.csv` (top B=30 theo suspicion score) → trang offline `out/site_top/index.html` (`python -m tools.t14_lowest_sheet --out out/site_top`). Bộ "30 box score thấp nhất" trên web cũ không dùng: không trùng box nào với queue, nên `review()` trả exit 2.
- **Reviewer**: Trần Nguyễn Phi Long (`long`), 1 người, 17 phút 45 giây / 30 box (≈ 35 giây/box). Kết quả: `reports/review_queue.csv` → copy sang `out/poc/review_queue_reviewed.csv`.
- **Quyết định duyệt tay**: **approve 10 · edit 17 · reject 3**. Queue bắt trúng 20/30 box có lỗi. 17 edit đều giữ class, vẽ lại khung (10/17 là box < 150 px²; IoU khung mới vs box model min 0.53 / median 0.72). 3 reject là box không có đối tượng (`aa_38#1`, `aa_50#3`, `aa_32#4`). Box khó nhất: `aa_09#4`, `aa_22#3` (pedestrian nhỏ, IoU khung sửa vs model 0.53), `aa_38#1` (box rộng 2 px). Chi tiết: `reports/individual/2A202604058_TranNguyenPhiLong.md`.
- **Override**: Cột `reviewer`, `reviewer_decision` (`approve`/`reject`/`edit`/`escalate`/`skip`), `override_value` → `review_queue_reviewed.csv`; box thiếu thêm bằng `review_additions.csv` (cần reviewer). Audit: `out/poc/audit_log.jsonl`; diff: `correction_diff.csv`.
- **Decision policy + `reason_code` + owner**: Xem mục 01. Owner (`configs/lab.yaml::owner`): `Bui Van Thinh (Lead)`.
- **Khi nào HOLD/REWORK/GO**: Chưa review hoặc review < 80% queue → `HOLD`; invalid label / round-trip lệch → `REWORK`; review $\ge 80\%$ và export hợp lệ → `GO`. Kết quả run thật: trước review `HOLD / NOT_REVIEWED` (227 box export); sau review **`GO / REVIEWED_AND_VALID`** (224 box export, invalid-label rate 0.0, round-trip YOLO/COCO ok, `review_fraction = 1.0`; `out/poc/decision.json` & `reports/decision_log.csv`):

```json
{
  "status": "GO",
  "reason_code": "REVIEWED_AND_VALID",
  "reason": "Queue reviewed and export passes validation (accuracy not asserted: no eval GT in this repo)",
  "owner": "Bui Van Thinh (Lead)",
  "scope": "t14",
  "evidence": [
    "out\\poc\\qa_report.json",
    "out\\poc\\review_queue.csv"
  ],
  "metrics": {
    "n_exported": 224,
    "n_invalid": 0,
    "invalid_label_rate": 0.0,
    "yolo_max_err_px": 0.000240000000005125,
    "coco_max_err_px": 0.0,
    "tol_px": 0.05,
    "n_boxes": 224,
    "yolo_ok": true,
    "coco_ok": true,
    "queue_size": 30,
    "n_reviewed": 30,
    "review_fraction": 1.0,
    "min_review_fraction": 0.8
  },
  "ts": "2026-11-15T03:54:21+00:00",
  "canonical": "GO"
}
```

## 08 · Reproduce & tests

```bash
make setup
.venv\Scripts\python.exe -m src.main --config configs/lab.yaml --track t14 --output out/poc --run-name poc --offline --seed 42
.venv\Scripts\python.exe -m src.main --config configs/lab.yaml --track t14 --pipeline src.baseline_t14 --output out/baseline --run-name baseline --offline --seed 42
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m tools.preflight
```

- **Test của nhóm** (`tests/test_t14.py`, 26 tests + `tests/test_t14seg_*.py` cho Lane B): Class map, unmapped class fail, alias sai, validator, suspicion score, review queue top-B, round-trip YOLO/COCO, clip box, parse override, pipeline end-to-end tái lập, baseline vs PoC, review loop → GO, review dở → HOLD, file review hỏng → exit 2, thêm box thiếu, và 5 unit tests trực tiếp cho `decide()` nhánh `REWORK / EXPORT_QA_FAILED` cùng thứ tự ưu tiên precedence.
- **Output `pytest -q`**: `26 passed` cho riêng `tests/test_t14.py`; `87 passed, 5 skipped` cho toàn bộ test suite trên repo (0 failed).
- **Máy / CPU / OS / Python**: Windows 11 (10.0.26200), CPU Intel64 Family 6 Model 142 Stepping 10, Python 3.14.3.

## 09 · Limitations & ethics

- **Rủi ro PII / license**: Dữ liệu fixtures là synthetic nên không có người/biển số thật. Nếu triển khai trên dữ liệu thực (COCO, BDD100K), cần áp dụng quy trình làm mờ khuôn mặt/biển số (như Track T07) trước khi đưa lên hệ thống gán nhãn ngoài. Mô hình YOLO26 tuân theo giấy phép AGPL-3.0, nếu tích hợp vào sản phẩm thương mại đóng mã nguồn cần mua giấy phép Enterprise hoặc chuyển sang kiến trúc Apache-2.0 (ví dụ RF-DETR).
- **Điều chưa biết**: Dữ liệu kiểm thử chỉ gồm N = 20 ảnh eval mô phỏng, độ chính xác F1 chưa thể hiện đầy đủ hiệu năng khi triển khai thực tế trên diện rộng; khi quy mô tăng 10x, số lượng false alarm ở ngưỡng thấp có thể làm quá tải đội ngũ reviewer.
- **Không tuyên bố "đạt GDPR / luật Việt Nam" từ checklist tự động**: Hệ thống chỉ là công cụ hỗ trợ lọc và kiểm soát kỹ thuật. Mọi quy chuẩn tuân thủ bảo vệ dữ liệu cá nhân (Nghị định 13/2023/NĐ-CP và GDPR) cần có sự đánh giá, phê chuẩn của DPO và bộ phận pháp chế.
- **Kế hoạch cải tiến**: Áp dụng ngưỡng confidence động tối ưu hóa theo từng class (per-class thresholding); tích hợp giao diện CVAT SDK để reviewer gán nhãn trực tiếp trên web; kiểm định tự động round-trip định dạng nhãn COCO <-> YOLO.

## 10 · Team contribution

**Bảng 2 · Individual contribution evidence** (đối chiếu `git shortlog -sne --all` và `reports/team_contributions.csv`)

| Học viên | Git author/email | Module / vai trò | Commit SHA / diff chứng minh | Trình bày? |
|---|---|---|---|---|
| Trần Thành Hùng | Hung-1812 | Metric contract, Baseline & Tổng hợp Eval (#10 / D-01) · `reports/metric_contract.md`, `src/baseline_t14.py` | Chờ commit thành viên | Không |
| Hoàng Quốc Chung | Hoang Quoc Chung <hoangquocchung2003@gmail.com> | Failure Analysis, 1-to-1 Label Map Validation & License (#9 / A-01) · `src/t14_tools.py`, `tests/test_t14_label_map.py`, `docs/FAILURE_CASES.md` | `98ccdb0`; `ed531db` | Không |
| Nguyễn Xuân Việt | vietnx <viet.ngxuan@gmail.com> | Code + Test + Export + Eval (#8 / C-01) · `src/t14_tools.py`, `tests/test_t14.py` | `05ce510` · `reports/individual/2A202604045_NguyenXuanViet.md` | Không |
| Trần Nguyễn Phi Long | Keishaa <shamt2004@gmail.com> | Chấm tay 30 ảnh/box trên web review (#7 / B-01) · `reports/review_queue.csv`, `tools/t14_box_picker.py` | `12012db` · `reports/individual/2A202604058_TranNguyenPhiLong.md` | Không |
| Phạm Phú Nguyễn Hưng | phamphunguyenhung <ppnh10092002@gmail.com> | Trang web review, Pipeline Lane A/B, Tích hợp & Hồ sơ nhóm (#11 / L-01..L-07) · `src/`, `tools/`, `web/`, `docs/`, `submission/README.md` | `928093a`; `77efa87` · `reports/individual/2A202604062_PhamPhuNguyenHung.md` | Không |
| Bùi Văn Thịnh | Bui Van Thinh <buivanthinhtxt@gmail.com> | Unit tests `decide()` REWORK, Failure Evidence, Decision Log (`GO`), Hoàn thiện báo cáo & Demo (#12 / E-01) · `tests/test_t14.py`, `reports/figures/`, `reports/decision_log.csv`, `reports/group_report.md`, `docs/report.md`, `docs/demo.md`, `docs/contributions.md` | `fork/main` (`Bui Van Thinh`) · `reports/individual/2A202604040_BuiVanThinh.md` | Có |

## 11 · References

| URL | Ngày xem | 1–2 dòng ý nghĩa |
|---|---|---|
| https://opencv.github.io/cvat/docs/ | 2026-11-14 | Tài liệu chuẩn về quy trình kiểm định chất lượng (QA) và định dạng xuất gán nhãn tự động trong CVAT. |
| https://docs.ultralytics.com/datasets/detect/ | 2026-11-14 | Quy cách cấu trúc tệp dữ liệu bounding box định dạng YOLO và hướng dẫn chuẩn hóa tọa độ x_center, y_center, w, h. |
| https://cocodataset.org/#format-data | 2026-11-14 | Đặc tả cấu trúc JSON của COCO dataset cho bài toán Object Detection và quy tắc biểu diễn bbox [x_min, y_min, w, h]. |

---

**Bảng 3 · Checklist chốt khi nộp**

- [x] Link repo và tag cuối cùng trong README; input mẫu không chứa PII thật.
- [x] Đúng 1 lệnh run, 1 lệnh evaluate, 1 lệnh tests; không cần download khi chạy.
- [x] Có baseline/PoC metrics, ≥1 failure case, reviewer evidence, decision log.
- [x] Có license/source provenance, data splits, seed, offline environment, GT/eval N.
- [x] Mỗi thành viên có ≥1 commit thực chất và được nêu minh bạch trong bảng trên.
