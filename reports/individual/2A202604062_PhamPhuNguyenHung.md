# Báo cáo cá nhân — Phạm Phú Nguyễn Hưng (2A202604062)

- **Nhóm:** `sIx4nhd0cthAn` · **Track:** T14 — Auto-Annotation + Human QC Exporter
- **Issue phụ trách:** [#11](https://github.com/phamphunguyenhung/K5-Day6-L2-sIx4nhd0cthAn-DataGovernanceAndPrivacy/issues/11) — Trang web review, Tích hợp & Hồ sơ nhóm

---

## 1. Việc đã làm

- Dọn codebase giữ lại duy nhất track `t14`, dựng pipeline PoC, baseline, tools và test ban đầu (`src/pipeline.py`, `src/t14_tools.py`, `src/baseline_t14.py`, `tests/test_t14.py`).
- Xây dựng trang web review tĩnh có zoom ảnh và Export CSV (`tools/t14_lowest_sheet.py`, `out/site/index.html`, `web/`) để Trần Nguyễn Phi Long (#7) chấm tay 30 box.
- Thiết lập quy ước làm việc trực tiếp trên `main` (`rules.md`), đồng bộ phân công chi tiết cho 6 thành viên (`docs/CHI_TIET_TASK_THANH_VIEN.md`, `docs/HUONG_DAN_LAM_BAI_NHOM.md`) và tạo Issues #7–#12 trên repo nhóm.
- Hoàn thiện hồ sơ nhóm: cập nhật vai trò 6 thành viên trong `submission/README.md` và `reports/team_contributions.csv`.

## 2. Lệnh đã chạy

```powershell
python -m tools.t14_lowest_sheet
python -m pytest -q
python -m tools.preflight
git shortlog -sne --all
```

## 3. Bằng chứng

- `python -m pytest -q`: **13 passed** (`13/13` xanh).
- `python -m tools.preflight`: **0 FAIL, 1 WARN** (`models_optional/ not present` — bình thường khi chạy fixture offline).
- Các file đã tạo/cập nhật:
  - `out/site/index.html` (trang duyệt 30 box kèm ảnh base64 + lightbox zoom + Export CSV).
  - `submission/README.md` (đủ 6 thành viên và vai trò).
  - `reports/team_contributions.csv` (đủ 6 dòng, khớp `git shortlog -sne --all` cho tác giả `Phạm Phú Nguyễn Hưng <ppnh10092002@gmail.com>`).
  - Commit trên `fork/main`: đã gộp thành 1 commit sạch (`Phạm Phú Nguyễn Hưng <ppnh10092002@gmail.com>`).

## 4. Hạn chế / Chưa xong

- `commit_shas` của 5 thành viên còn lại trong `reports/team_contributions.csv` sẽ được bổ sung thêm sau khi các bạn hoàn thành và push commit của mình (#7, #8, #9, #10, #12).
- `out/site/index.html` hiện hiển thị 30 box có score thấp nhất; khi Long Export CSV cần lưu đè vào `out/poc/review_queue.csv` / `reports/review_queue.csv` trước khi Việt chạy lại Export & Eval.
