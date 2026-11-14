# Báo cáo cá nhân — Nguyễn Xuân Việt (2A202604045)

Track T14 · Auto-Annotation + Human QC Exporter · Issue #8: Code, Test, chạy lại Export/Eval (kèm kiểm tra review queue).

## 1. Việc đã làm

- **Test (`tests/test_t14.py`)**: thêm 8 test (13 → 21), gồm test đỏ có chủ đích cho failure mode 1 (map class sai nhưng vẫn nằm trong taxonomy), test validator, test công thức suspicion score, test review queue, test round-trip YOLO/COCO.
- **Sửa `src/t14_tools.py::build_class_map`** (thêm 8 dòng) để test đỏ thành xanh. Adapter giờ từ chối hai trường hợp:
  - alias đổi tên một class mà taxonomy đã có đúng tên đó (ví dụ `car → pedestrian`);
  - hai class của model cùng map về một class taxonomy (ví dụ `person → car` trong khi `car → car`).
- **Review queue**: chạy PoC, kiểm tra `out/poc/review_queue.csv` và tính tỉ lệ box khó trong queue so với toàn bộ.
- **Chạy lại export + eval**: chạy lại PoC và baseline ở commit `3ee254b`; thử `src.evaluate` và `tools.t14_eval_dev`.
- **Hồ sơ**: điền dòng của mình trong `reports/team_contributions.csv`; bổ sung số liệu queue, PoC và test vào mục 03, 04, 08, 10 của `reports/group_report.md`.

### Công thức suspicion score (`add_suspicion`, threshold `thr` = 0.25)

```
mid         = 1 − |conf − 0.5| × 2
near        = 1 − |conf − thr| / 0.15
uncertainty = clip(max(mid, near), 0, 1)
score       = 0.5·uncertainty + 0.3·small + 0.2·clipped + 0.3·confusion
```

- `small`: diện tích box `w·h` < 150 px².
- `clipped`: box bị cắt vì tràn ra ngoài ảnh.
- `confusion`: box chồng (IoU ≥ 0.5) lên một box **khác class** trong cùng ảnh (`flag_class_confusion`).
- Queue = 30 box có score cao nhất (`budget = 30`), sắp xếp ổn định (`mergesort`) nên cùng seed cho cùng thứ tự.
- Không có threshold riêng theo class: mọi class dùng chung `conf_threshold = 0.25`. Trọng số là giá trị mặc định trong code, `configs/lab.yaml` để `params: {}`, không chỉnh sau khi chạy.

## 2. Lệnh đã chạy

```powershell
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m tools.preflight
.venv\Scripts\python -m src.main --config configs/lab.yaml --track t14 --output out/poc --run-name poc --offline --seed 42
.venv\Scripts\python -m src.main --config configs/lab.yaml --track t14 --pipeline src.baseline_t14 --output out/baseline --run-name baseline --offline --seed 42
.venv\Scripts\python -m src.evaluate --track t14 --pred out/poc/predictions.csv --run-name poc --out reports/metrics_poc.json
.venv\Scripts\python -m tools.t14_eval_dev --gold gold/gold_dev.csv --pred out/poc/predictions.csv
```

## 3. Bằng chứng

### 3.1 Test đỏ rồi xanh

Trước khi sửa `build_class_map` (2 failed, 19 passed):

```
...FF................                                                    [100%]
FAILED tests/test_t14.py::test_wrong_alias_inside_the_taxonomy_is_refused[alias0-contradicts the taxonomy]
FAILED tests/test_t14.py::test_wrong_alias_inside_the_taxonomy_is_refused[alias1-map to the same taxonomy class]
E       Failed: DID NOT RAISE ValueError
```

Sau khi sửa:

```
.....................                                                    [100%]
```

`tools.preflight`: `0 FAIL, 1 WARN` (WARN là `models_optional/` không có, nhóm không dùng weights ngoài).

`test_validator_alone_cannot_see_a_wrong_class_map` ghi lại lý do phải chặn ở adapter: khi đã map xong, đổi chéo `person ↔ car` cho ra 0 dòng invalid ở `validate_predictions`.

### 3.2 Review queue (`out/poc/review_queue.csv`, commit `3ee254b`, config hash `6dec1da15c3f`)

| Kiểm tra | Kết quả |
|---|---|
| Số dòng | 30 (`budget = 30`) |
| Cột | `rank, item_id, score, reason, suggested_action, slice, evidence_path` + 5 cột reviewer |
| `rank` | 1..30 liên tục |
| `item_id` trùng | 0 |
| Score giảm dần | đúng, từ 0.783 xuống 0.473 |
| Box ngoài queue có score cao hơn | không có |

So sánh loại box khó trong queue với toàn bộ (tính từ `out/poc/boxes_all.csv`):

| Nhóm | N | Box nhỏ (< 150 px²) | Conf trung bình (0.35–0.65) | Dưới threshold | Clipped | Class confusion |
|---|---|---|---|---|---|---|
| Queue (top 30) | 30 | 18 (60.0%) | 23 (76.7%) | 0 | 1 | 0 |
| Toàn bộ box model | 234 | 19 (8.1%) | 78 (33.3%) | 7 | 3 | 0 |
| Box giữ lại (conf ≥ 0.25) | 227 | 18 (7.9%) | 78 (34.4%) | 0 | 3 | 0 |

- Queue chứa 18/19 box nhỏ của fixture, tỉ lệ box nhỏ gấp khoảng 7 lần mặt bằng chung.
- `reason` trong queue: `mid_conf;small` 12 · `mid_conf` 11 · `small` 6 · `clipped` 1.
- Theo class: pedestrian 13 · car 8 · sign 6 · motorbike 3. Theo split: dev 16 · eval 14.

### 3.3 Chạy lại PoC và baseline

| Run | Box model | Export | Invalid | Round-trip YOLO / COCO | Decision | Runtime |
|---|---|---|---|---|---|---|
| PoC | 234 | 227 | 0 (0.0%) | 0.00024 px / 0.0 px (tol 0.05) | `HOLD / NOT_REVIEWED` | 0.40 s |
| Baseline | 234 | 227 | 3 (1.32%) | không kiểm tra | `HOLD / BASELINE_ONLY` | 0.03 s |

Số lấy từ `out/<run>/qa_report.json`, `decision.json`, `run_info.json`. `out/` bị `.gitignore`, sinh lại bằng hai lệnh `src.main` ở mục 2.

### 3.4 Eval

- `src.evaluate`: exit 2, `ground truth file not found: fixtures/t14_auto_annotation/gt/gt_boxes.csv`. **Chưa có số chính thức.** Không có `reports/metrics_*.json` và `table1_baseline_vs_poc.md`.
- `tools.t14_eval_dev`: `FileNotFoundError: gold/gold_dev.csv`. Gold set dev của nhóm chưa có.

## 4. Chưa xong / hạn chế

1. **Trang duyệt đang lệch với review queue.** Trang https://t14-review.pages.dev/ (sinh bằng `tools.t14_lowest_sheet`) hiển thị 30 box có score **thấp nhất** (0.010–0.147), còn `review_queue.csv` là 30 box score **cao nhất**. Hai tập không trùng box nào (0/30). Thử ghép một file CSV xuất từ trang đó vào `out/poc/review_queue_reviewed.csv` thì pipeline dừng exit 2: `reviewed file has items outside the review queue: ['aa_01#0', 'aa_02#2', ...]`. Vì vậy chưa thể chạy lại export với CSV của Long cho tới khi nhóm chốt: duyệt lại trên đúng queue (`out/poc/contact_sheet.html` có sẵn nút Export), hoặc đổi pipeline để nhận box ngoài queue.
2. Chưa có CSV duyệt của Long nên decision vẫn là `HOLD / NOT_REVIEWED`.
3. Chưa có `gt/` và `gold/gold_dev.csv`: không có F1, không có bảng baseline vs PoC, chưa kiểm chứng được trọng số score trên dev.
4. Guard mới ở `build_class_map` không bắt được trường hợp hoán vị giữa các class khác tên (ví dụ `person → motorbike` cùng lúc `motorcycle → pedestrian`). Trường hợp này chỉ người duyệt hoặc gold set mới phát hiện.
5. Guard từ chối gộp hai class model vào một class taxonomy. Nếu sau này cần gộp thật (ví dụ `truck → car`) thì phải nới luật này.
6. `confusion` = 0 trên toàn fixture nên trọng số 0.3 của nó chưa có tác dụng trên dữ liệu này; chỉ được kiểm bằng unit test.
7. `commit_shas` của tôi trong `team_contributions.csv` chưa điền (Hưng gom ở #11).
