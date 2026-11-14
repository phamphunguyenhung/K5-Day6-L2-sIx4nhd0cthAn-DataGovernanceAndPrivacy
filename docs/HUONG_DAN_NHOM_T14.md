# HƯỚNG DẪN NHÓM — T14 Auto-Annotation + Human QC

> Đây là file hướng dẫn DUY NHẤT của nhóm. Tech Lead: **Phạm Phú Nguyễn Hưng**.
> Mọi thông tin về task nằm ở `TASKS.md`, quyết định kỹ thuật ở `DECISIONS.md`, tiến độ ở `PROGRESS.md`.

## 1. Luật chung (đọc 1 phút)

1. **Chỉ sửa file thuộc phần của mình** (bảng mục 3). Sửa chéo thì hỏi Hưng trước.
2. **Không ghi lên CVAT server**, không tạo task, không upload ảnh, không gửi token vào nhóm hay log. Chỉ làm khi Hưng cấp task đích và nói rõ.
3. **Ảnh thật (`data/real/`) chỉ ở máy local.** Không commit, không đăng lên nhóm, không đưa vào slide.
4. **Không đọc `fixtures/*/gt/`**, không sửa evaluator, không sửa ground truth.
5. Nhãn do model sinh ra là **bản nháp (draft)**, không gọi là ground truth hay nhãn tay của nhóm.
6. Commit bằng **tài khoản GitHub của chính mình**. Không commit hộ người khác. Chỉ `git add` đúng file của mình, **không `git add .`**.
7. Không `push --force`, không `reset --hard` trên nhánh chung.
8. Không đánh dấu DONE khi chưa có lệnh chạy và output chứng minh.

## 2. Cài đặt và chạy thử (ai cũng làm, khoảng 5 phút)

Windows PowerShell, tại thư mục gốc repo:

```powershell
git pull
python -m pip install -r requirements.txt
python -m pytest -q
python -m tools.preflight
python -m src.main --config configs/lab.yaml --track t14 --pipeline src.baseline_t14 --output out/baseline --run-name baseline --offline --seed 42
python -m src.main --config configs/lab.yaml --track t14 --output out/poc --run-name poc --offline --seed 42
```

Kết quả đúng (Hưng đã chạy thử trên máy mình):

| Lệnh | Kết quả mong đợi |
|---|---|
| `pytest -q` | tất cả pass (21 test) |
| `tools.preflight` | `0 FAIL` (1 WARN về `models_optional/` là bình thường) |
| `src.main` baseline / poc | exit code `0`, có `out/<run>/predictions.csv` |
| `src.evaluate ...` | exit code `2` và dòng "ground truth file not found": **đúng thiết kế**, repo sinh viên không có GT, giảng viên mới chấm chính thức. Không cố "sửa". |

Mở `out/poc/contact_sheet.html` bằng trình duyệt để xem hàng đợi review.

### 2b. Setup cho Lane B (ảnh thật, segmentation draft) — chỉ Tech Lead và ai chạy pipeline ảnh thật

Phần 2 chỉ cần `requirements.txt` của repo. Lane B cần thêm công cụ và dữ liệu **không nằm trong repo**:

| Cần | Chi tiết | Kiểm tra |
|---|---|---|
| Python | 3.11 trở lên (tool CVAT yêu cầu `>=3.11`). Hưng chạy trên 3.14.6. | `python --version` |
| Thư viện thêm | `opencv-python`, `onnxruntime`, `ultralytics` (kéo theo `torch`), `shapely`, `psutil`, `pycocotools` (tùy chọn, để validator đọc lại COCO) | `python -c "import cv2, onnxruntime, ultralytics, shapely, psutil"` |
| Tool CVAT (chỉ đọc, không sửa) | thư mục `cvat/src` trong workspace `learning` | đặt biến `CVAT_TOOLS_SRC` |
| Weights | `tools/cvat-seg-copilot/models/segformer_b2_cityscapes.onnx` (~110 MB) và `tools/cvat-seg-copilot/yolo26s-seg.pt` (~23 MB) trong workspace `learning` | `ls` hai file trên |
| Ảnh thật | `data/real/day02_traffic/{images/, images.csv, manifest.csv}` (60 ảnh, 50 dev + 10 eval) | `Import-Csv data\real\day02_traffic\images.csv \| group split` |

Weights và ảnh thật bị `.gitignore` (`*.pt`, `*.onnx`, `data/real/`), nên **không** có sau `git pull`; xin Hưng hoặc lấy từ workspace.
Ảnh thật chỉ ở máy local, không commit, không đăng nhóm.

```powershell
python -m pip install opencv-python onnxruntime ultralytics shapely psutil pycocotools
$env:CVAT_TOOLS_SRC = "<đường dẫn tới learning>/cvat/src"
python -m tools.t14seg.runner --config configs/t14_seg_v2.yaml --limit 2      # thử 2 ảnh trước
python -m tools.t14seg.runner --config configs/t14_seg_v2.yaml                # 50 ảnh dev (~8 phút, RAM đỉnh ~5.7 GB)
```

- Lần đầu mất thêm thời gian nạp model. Chạy lại cùng ảnh và cấu hình thì dùng cache (`out/seg_v2/cache`), rất nhanh.
- Cần khoảng 6 GB RAM trống. Đừng chạy song song nhiều tiến trình nặng: một lần chạy 50 ảnh đã bị ngắt giữa chừng khi máy đang
  bận (nghi thiếu RAM).
- Kết quả ở `out/seg_v2/` (gitignore). File để import CVAT nằm ở `out/seg_v2/cvat_import/` sau khi chạy `export_cvat`
  (lệnh đầy đủ trong `docs/benchmark/README.md`, mục Tái lập). **Không tự upload lên server CVAT** khi Hưng chưa cấp task.
- Bảng số đo và các việc chưa đo: `docs/benchmark/README.md`.

## 3. Phân công

Tên lấy từ `reports/team_contributions.csv`. Việc ghép tên và vai trò A–E là Hưng đề xuất, ai muốn đổi thì nhắn Hưng.

| Vai | Người | Việc | Chỉ được sửa | Xong khi |
|---|---|---|---|---|
| A. Manual Gold | Hoàng Quốc Chung | Chọn 10–15 ảnh dev (mở rộng 20–30 nếu kịp), tự vẽ bbox chuẩn bằng tay, ghi thời gian mỗi ảnh và quy tắc vẽ | `docs/team/gold.md` | Có nhãn do chính bạn vẽ, bảng thời gian/ảnh, không commit ảnh |
| B. Human QC | Trần Nguyễn Phi Long | Review box dự đoán: approve/edit/delete. Ưu tiên ảnh không có box nào và vật nhỏ. Kiểm chéo 20% nhãn của A | `docs/team/qc.md`, `review_decisions.csv` | Mỗi dòng có reviewer, thời gian, diff thật |
| C. Tests | Nguyễn Xuân Việt | Viết test dương và âm: mapping class, bỏ lớp ngoài taxonomy, box sai, seed lặp lại được, dry-run không ghi remote, không đọc GT | `tests/test_t14_real_*.py` | `pytest -q` pass, dán output vào nhóm |
| D. Export và Benchmark | Trần Thành Hùng | Test schema CSV, import lại YOLO/COCO, bảng so sánh baseline với PoC, ghi failure cases. Không sửa evaluator | `tests/test_t14_export_*.py`, `docs/team/bench.md` | Bảng so sánh cùng seed và cùng split, số nào chưa đo được thì ghi "unavailable" |
| E. Báo cáo và Demo | Bùi Văn Thịnh | Báo cáo, sơ đồ kiến trúc, kịch bản demo, bảng đóng góp Git, slide 8–10 phút | `docs/report.md`, `docs/demo.md`, `docs/contributions.md` | Chỉ ghi việc thực sự đã xong, có commit thật |
| Tech Lead | Phạm Phú Nguyễn Hưng | Pipeline chính (select, detect, adapter, guard, rank, export), tích hợp, merge | phần còn lại | End-to-end chạy được |

Lưu ý: hiện chưa có code "ảnh thật" (Lane B). Việc A, B chuẩn bị trước bằng fixture hiện có. Hưng sẽ báo khi Lane B sẵn sàng để làm trên ảnh thật.

## 4. Báo kết quả vào nhóm

Mỗi người chạy mục 2, rồi gửi vào nhóm **theo mẫu này** (copy nguyên, điền số thật, không viết khác đi):

```
[Tên | vai A–E]
Máy: Windows/Mac/Linux, Python x.y
pytest -q: <N passed / N failed>
preflight: <N FAIL, N WARN>
src.main baseline: exit <0>
src.main poc: exit <0>
Lỗi gặp (nếu có): <dán 5 dòng cuối, bỏ đường dẫn có tên máy cá nhân>
Việc của mình: <TODO / IN_PROGRESS / BLOCKED / REVIEW / DONE> — <link commit hoặc file>
```

Gặp lỗi: dán đúng thông báo lỗi, không tự sửa file ngoài phạm vi. Đừng gửi ảnh thật hay token.

## 5. Ghi nhận tiến độ

- Mỗi mốc xong: ghi vào `PROGRESS.md` (ngày giờ, task ID, lệnh đã chạy, đường dẫn output, kết quả thực tế).
- Cập nhật cột Status trong `TASKS.md` cho dòng của mình.
- Sau khi có commit thật, điền `commit_shas` của bạn vào `reports/team_contributions.csv` (đúng với `git shortlog -sne --all`).

## 6. Ghi chú riêng của bài khác

- **Bài segmentation (Day03):** khi kéo opacity lên hết, mask phải khít mép đối tượng. Chỉ đối tượng không có class mới được để hở. T14 dùng bbox nên yêu cầu này không áp dụng ở đây.
