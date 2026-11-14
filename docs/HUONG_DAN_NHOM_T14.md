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
