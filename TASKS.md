# TASKS — T14 Auto-Annotation + Human QC

Hướng dẫn chạy và mẫu báo cáo: [docs/HUONG_DAN_NHOM_T14.md](docs/HUONG_DAN_NHOM_T14.md)

Trạng thái: TODO | IN_PROGRESS | BLOCKED | REVIEW | DONE

## Tech Lead (core)
| ID | Task | Lane | Status |
|---|---|---|---|
| L-01 | `.gitignore` cho data/real, weights, .env.local | - | DONE |
| L-02 | Lane A: baseline + PoC offline seed 42, regression | A | IN_PROGRESS |
| L-03 | Lane B select: `real_manifest.csv` (SHA256, seed 42) | B | TODO |
| L-04 | Smoke test `cvat_tools` CLI (`--help`, JSON) | B | TODO |
| L-05 | detect adapter + taxonomy OOD audit | B | TODO |
| L-06 | guard (flag-only) + rank S (ablation trên dev) | B | TODO |
| L-07 | review file + export + QA gates | B | TODO |

## Thành viên hỗ trợ (tự làm, tự commit bằng tài khoản của mình)
| ID | Owner | Task | File được sửa | Acceptance | Status |
|---|---|---|---|---|---|
| A-01 | Hoàng Quốc Chung (A) | Vẽ nhãn gold tay 10–15 ảnh dev (mở rộng 20–30), ghi thời gian + guideline | `docs/team/gold.md` | Nhãn do người vẽ; có thời gian/ảnh; không commit ảnh | TODO |
| B-01 | Trần Nguyễn Phi Long (B) | Review box dự đoán (approve/edit/delete), ưu tiên ảnh 0 detection + vật nhỏ; cross-check 20% nhãn của A | `docs/team/qc.md`, `review_decisions.csv` | Có reviewer + timestamp + diff thật | TODO |
| C-01 | Nguyễn Xuân Việt (C) | Test mapping, OOD drop, invalid box, seed deterministic, dry-run không ghi remote, no GT leak | `tests/test_t14_real_*.py` | pytest pass + evidence | TODO |
| D-01 | Trần Thành Hùng (D) | Test CSV schema, import YOLO/COCO, so sánh baseline vs PoC, failure cases | `tests/test_t14_export_*.py`, `docs/team/bench.md` | Không sửa evaluator | TODO |
| E-01 | Bùi Văn Thịnh (E) | Report, sơ đồ kiến trúc, demo script, bảng Git evidence, slide 8–10 phút | `docs/report.md`, `docs/demo.md`, `docs/contributions.md` | Chỉ ghi việc thực sự xong | TODO |
