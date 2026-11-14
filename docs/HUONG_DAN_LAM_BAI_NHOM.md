# Hướng dẫn làm bài nhóm T14 (đọc file này trước)

Chi tiết từng bước của từng người: `docs/CHI_TIET_TASK_THANH_VIEN.md`.

## 1. Ai làm gì (Issues #7 – #12)

| Issue | Người | Việc chính | File đầu ra |
|---|---|---|---|
| #7 | **Trần Nguyễn Phi Long** | Chấm tay 30 ảnh/box trên web review | `out/poc/review_queue.csv` (hoặc `reports/review_queue.csv`) |
| #8 | **Nguyễn Xuân Việt** | Code, viết/sửa test (`pytest` xanh), chạy lại Export & Eval | `src/t14_tools.py`, `tests/test_t14.py`, `out/poc/predictions.csv`, `reports/metrics_*.json`, `reports/table1_baseline_vs_poc.md` |
| #9 | **Hoàng Quốc Chung** | Failure Analysis (2 case bắt buộc + ảnh + root cause) & License AGPL-3.0 | `reports/group_report.md` (mục 06, 09), `reports/figures/` |
| #10 | **Trần Thành Hùng** | Kiểm tra Metric Contract, Baseline & Tổng hợp bảng Eval | `reports/metric_contract.md`, `out/baseline/predictions.csv`, `reports/group_report.md` (mục 04, 05) |
| #11 | **Phạm Phú Nguyễn Hưng** | Trang web review, tích hợp & chốt hồ sơ nhóm | `out/site/index.html`, `reports/team_contributions.csv`, `submission/README.md` |
| #12 | **Bùi Văn Thịnh** | Decision Log (`GO/HOLD/REWORK`) & rà soát hoàn thiện báo cáo nhóm | `reports/decision_log.csv`, `reports/group_report.md` |

## 2. Mỗi người nộp gì

- **1 file báo cáo `.md`**: `reports/individual/<MSSV>_<HoTen>.md`. Không làm HTML.
- Báo cáo có 4 mục ngắn: **Đã làm gì · Lệnh đã chạy · Bằng chứng · Chưa xong/hạn chế**.
- Điền mục của mình trong `reports/group_report.md` và dòng của mình trong `reports/team_contributions.csv`.
- Số liệu phải có file hoặc output chứng minh. Không điền số đoán.

## 3. Duyệt box trên web (Trần Nguyễn Phi Long — Issue #7)

Mở [out/site/index.html](out/site/index.html) (hoặc **https://t14-review.pages.dev/**).

Trang có 30 ảnh, mỗi ảnh một box cần xem. **Khung đỏ** là box cần duyệt, **khung xám** là các box khác trong ảnh.

1. Gõ **tên của bạn** (`long`) vào ô "Reviewer" trên cùng.
2. Với mỗi ảnh, chọn một quyết định trong ô xổ xuống:

   | Chọn | Khi nào |
   |---|---|
   | `approve` | Box và tên class đều đúng |
   | `reject` | Không có vật thể, hoặc box sai hẳn |
   | `edit` | Đúng vật thể nhưng sai class: gõ class đúng vào ô "override value" (`car`, `pedestrian`, `motorbike`, `sign`) |
   | `escalate` | Không chắc, nhờ người khác xem |
   | `skip` | Bỏ qua, xem sau |

3. Nếu chọn `reject` hoặc `edit`, ghi **một câu lý do** vào ô "reason / note".
4. Xong cả 30 ảnh thì bấm **Export decisions CSV** và lưu đè vào `out/poc/review_queue.csv` (và copy sang `reports/review_queue.csv` nếu cần nộp).

**Phóng to ảnh:** bấm vào ảnh (`+` / `−` để zoom, nhấp đúp zoom 3x, kéo chuột di chuyển, `Esc` để đóng).

Lưu ý:
- Trang không lưu giữa chừng. **Đừng tắt hoặc tải lại trang** trước khi bấm Export.
- Tiêu chí đúng/sai: xem `docs/HUONG_DAN_GAN_NHAN_CVAT.md`.

## 4. Kiểm tra trước khi nộp

Chạy hai lệnh, cả hai phải báo xanh:

```powershell
python -m pytest -q
python -m tools.preflight
```

## 5. Quy tắc git (làm trực tiếp trên `main`)

1. Làm việc **trực tiếp trên nhánh `main`** và push lên remote `fork` (`phamphunguyenhung/K5-Day6-L2-sIx4nhd0cthAn-DataGovernanceAndPrivacy`), **không tạo nhánh phụ** (xem `rules.md`).
2. Trước khi làm: `git pull fork main`.
3. Tên và email git phải đúng của bạn (`git config user.name`, `git config user.email`).
4. Không commit thư mục `out/`, file `.env`, `.dev.vars`. (File `review_queue.csv` cần nộp thì copy vào `reports/review_queue.csv`).

## 6. Thứ tự làm

```
Song song ngay:
  - Long (#7): chấm tay 30 ảnh trên web -> review_queue.csv
  - Việt (#8): code & sửa test label-map / round-trip (pytest xanh)
  - Chung (#9): Failure Analysis (2 case) + License AGPL-3.0
  - Hùng (#10): kiểm tra metric_contract.md + baseline
  - Hưng (#11): hỗ trợ web, tích hợp, team_contributions.csv

Sau khi Long (#7) xong CSV:
  - Việt (#8) chạy lại Export + src.evaluate + src.compare -> metrics_*.json, table1_baseline_vs_poc.md
  - Hùng (#10) điền số liệu Eval vào group_report.md
  - Thịnh (#12) chốt decision_log.csv (GO/HOLD/REWORK) & hoàn thiện group_report.md
```
