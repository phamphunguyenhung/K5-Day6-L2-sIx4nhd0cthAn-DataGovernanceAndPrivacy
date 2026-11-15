# Báo cáo cá nhân · Trần Thành Hùng (2A202603247)

**Task:** Issue #10 · Metric Contract, Baseline, benchmark theo split · GitHub `Hung-1812`

## 1. Việc đã làm
- Chạy thử pipeline baseline và PoC theo cấu hình T14;  cả hai lệnh kết thúc với exit code 0 và sinh `predictions.csv` mỗi file 227 dòng trên toàn bộ 60 ảnh (40 dev + 20 eval).
- Chạy `pytest -q` và `tools.preflight`; báo cáo tools ghi nhận 1 xfail có chủ đích và preflight 0 FAIL, 1 WARN do thiếu `models_optional/`.
- Thử `tools.t14seg.bench_report`; kết quả được ghi nhận chỉ chạy được với `PYTHONIOENCODING=utf-8` trên Windows. Báo cáo cũng nêu lỗi `UnicodeEncodeError` khi console dùng cp1252.
- Chạy thử đánh giá: `src.evaluate` dừng với exit 2 do không có `gt/`; `src.compare` không chạy được do thiếu file metrics. Bản ghi eval giả trên dev cho F1=1.000 chỉ xác nhận dây nối, không phải kết quả độ chính xác.

## 2. Lệnh đã chạy
```powershell
python -m pytest -q
python -m tools.preflight
python -m src.main --config configs/lab.yaml --track t14 --pipeline src.baseline_t14 --output out/baseline --run-name baseline --offline --seed 42
python -m src.main --config configs/lab.yaml --track t14 --output out/poc --run-name poc --offline --seed 42
$env:PYTHONIOENCODING = "utf-8"
python -m tools.t14seg.bench_report --a out/seg_local --b out/seg_v2/run
python -m src.evaluate --track t14 --pred out/poc/predictions.csv
```

## 3. Bằng chứng
```text

Ngày chạy: 2026-11-15
Máy: Windows 11, Python 3.14.6

tools.preflight: OK, 0 FAIL, 1 WARN
WARN: thiếu models_optional/ (chỉ cần khi dùng model pretrained)

src.main baseline / poc: OK, exit 0
mỗi run: predictions.csv 227 dòng; pipeline chạy trên cả 60 ảnh (không lọc split)

pytest -q: OK, 1 xfail có chủ đích
test_exported_draft_has_no_gap (lỗi gap của run cũ 49418fa7)

tools.t14seg.bench_report --a out/seg_local --b out/seg_v2/run:
OK chỉ khi PYTHONIOENCODING=utf-8; lỗi được ghi nhận trên console Windows cp1252:
UnicodeEncodeError

tools.t14_eval_dev với gold giả: chạy được, F1=1.000
(gold giả là chính dự đoán dev, chỉ chứng minh dây nối đúng)

src.evaluate --track t14 --pred out/poc/predictions.csv: exit 2
Lý do ghi trong TOOLS_REPORT.md: không có gt/
src.compare: không chạy được vì thiếu metrics_*.json do chưa có gt/
```

## 4. Hạn chế
- Pipeline ghi nhận chạy trên cả 60 ảnh, nhưng `TOOLS_REPORT.md` không có bảng metric baseline-vs-PoC được lọc riêng theo dev/eval. Do đó số baseline/PoC theo split: **unavailable**; không dùng F1 giả làm số chính thức.
- `reports/metric_contract.md`, `configs/tracks/t14.yaml`, `out/baseline/predictions.csv` và `fixtures/t14_auto_annotation/model_card_fixture.yaml` không có trong gói hiện tại. Chưa thể xác nhận trực tiếp contract (Detection F1@IoU0.5, conf_threshold 0.25, budget 30, seed 42), schema CSV, hay ghi nguồn/version/license cụ thể từ model card. Theo nhiệm vụ, prediction là fixture mô phỏng, không phải inference thật.
- Không có `gt/` hoặc gold tay: chưa có F1 chính thức; `src.compare` và các số chính thức ở mục 04–05 chờ dữ liệu/đầu ra của thành viên #8.
- `bench_report` có lỗi encoding trên Windows cp1252; các kết quả `t14seg.evaluate gold`, `iaa`, import thật vào CVAT và chạy đủ 50 ảnh của runner chưa được kiểm chứng theo `TOOLS_REPORT.md`.
