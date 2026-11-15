# Bảng Tổng hợp Đóng góp Thành viên & Bằng chứng Git — Nhóm `sIx4nhd0cthAn`

Tài liệu này đối chiếu trực tiếp với `reports/team_contributions.csv`, `submission/README.md` và kết quả lệnh `git shortlog -sne --all`.

---

## 1. Bảng Phân công & Trạng thái Đóng góp (6 Thành viên)

| # | Học viên (MSSV) | GitHub / Git Author | Vai trò (Issue / Task) | File phụ trách chính | Bằng chứng thực tế trong Repo |
|---|---|---|---|---|---|
| 1 | **Trần Thành Hùng** (`2A202603247`) | `Hung-1812` | Metric Contract, Baseline & Tổng hợp Eval (#10 / D-01) | `reports/metric_contract.md`, `src/baseline_t14.py`, `reports/group_report.md` (Mục 04, 05) | Đã định nghĩa `reports/metric_contract.md` và baseline; đang chờ commit cá nhân lên nhánh chung |
| 2 | **Hoàng Quốc Chung** (`2A202604039`) | `Hoang Quoc Chung <hoangquocchung2003@gmail.com>` | Failure Analysis, 1-to-1 Label Map Validation & License (#9 / A-01) | `src/t14_tools.py`, `tests/test_t14_label_map.py`, `docs/FAILURE_CASES.md` | Commit `98ccdb0`, `ed531db` |
| 3 | **Nguyễn Xuân Việt** (`2A202604045`) | `vietnx <viet.ngxuan@gmail.com>` | Code, Unit Tests Label-map/Round-trip & Eval (#8 / C-01) | `src/t14_tools.py`, `tests/test_t14.py`, `reports/individual/2A202604045_NguyenXuanViet.md` | Commit `05ce510` (`feat(t14): guard in-taxonomy class-map errors, add queue/QA tests, add Viet's report (#8)`) |
| 4 | **Trần Nguyễn Phi Long** (`2A202604058`) | `Keishaa <shamt2004@gmail.com>` | Human Review 30 ảnh/box trên web (#7 / B-01) | `reports/review_queue.csv`, `tools/t14_box_picker.py`, `reports/individual/2A202604058_TranNguyenPhiLong.md` | Commit `12012db` (`feat(t14): human review of top-30 suspicion queue (#7)` — approve 10, edit 17, reject 3 → `GO`) |
| 5 | **Phạm Phú Nguyễn Hưng** (`2A202604062`) | `phamphunguyenhung <ppnh10092002@gmail.com>` | Tech Lead: Pipeline T14 (Lane A & Lane B), Web Review & Tích hợp (#11 / L-01..L-07) | `src/pipeline.py`, `tools/`, `web/`, `docs/`, `submission/README.md`, `reports/individual/2A202604062_PhamPhuNguyenHung.md` | Các commit `928093a`, `5b82e0e`, `90c5170`, `1a90ba8`, `5868427`, `91c114d`, `45e3098`, `77efa87` |
| 6 | **Bùi Văn Thịnh** (`2A202604040`) | `Bui Van Thinh <buivanthinhtxt@gmail.com>` | Unit Tests `decide()` REWORK, Failure Evidence, Decision Log, Báo cáo & Demo (#12 / E-01) | `tests/test_t14.py`, `configs/lab.yaml`, `reports/decision_log.csv`, `reports/group_report.md`, `reports/figures/`, `reports/individual/2A202604040_BuiVanThinh.md`, `docs/report.md`, `docs/demo.md`, `docs/contributions.md` | Commit trên `main` của `Bui Van Thinh <buivanthinhtxt@gmail.com>` · Người trình bày (Presenter) |

---

## 2. Lệnh Kiểm tra Bằng chứng Đóng góp trên Git

Giảng viên và các thành viên có thể xác minh lịch sử đóng góp bất kỳ lúc nào bằng các lệnh:

```powershell
# Thống kê số lượng commit theo từng tác giả trên toàn bộ nhánh
git shortlog -sne --all

# Xem lịch sử commit chi tiết kèm các file thay đổi
git log --oneline --stat -n 15

# Kiểm tra các báo cáo cá nhân đã nộp trong repo
Get-ChildItem reports/individual/*.md
```

