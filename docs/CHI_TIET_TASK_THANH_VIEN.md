# Chi tiết task từng thành viên — T14 Auto-Annotation + Human QC Exporter

Đọc `docs/HUONG_DAN_LAM_BAI_NHOM.md` trước (quy tắc chung, cách duyệt box trên web). File này ghi **từng bước, lệnh và khi nào gọi là xong** cho từng người theo phân công mới (Issues #7 – #12).

## Quy ước chung (áp dụng cho cả 6 người)

- Lệnh viết cho **PowerShell** (Windows). Nếu máy không có `make`, chạy trực tiếp `python -m ...` như trong từng bước.
- Cài môi trường một lần: `python -m venv .venv; .venv\Scripts\python -m pip install -r requirements.txt`, rồi dùng `.venv\Scripts\python` thay cho `python`.
- Làm việc trực tiếp trên nhánh `main` (push lên remote `fork`), **không tạo nhánh phụ** (xem `rules.md`).
- `out/` bị `.gitignore`: file nào cần nộp phải copy sang `reports/` hoặc ghi rõ lệnh sinh lại trong báo cáo.
- Seed 42, `--offline`. Không đọc ground truth (`gt/`), không tune trên split `eval`.
- Báo cáo cá nhân: `reports/individual/<MSSV>_<HoTen>.md` gồm 4 mục: Việc đã làm · Lệnh đã chạy · Bằng chứng · Hạn chế. Không làm HTML.
- Xong việc = có bằng chứng (output lệnh hoặc file), không chỉ ghi "đã viết xong".

---

## 1. Trần Nguyễn Phi Long — Issue #7: Chấm tay 30 ảnh/box trên web

**Mục tiêu:** duyệt thủ công 30 box bằng web, ghi quyết định có tên và lý do.

1. Đọc `docs/HUONG_DAN_GAN_NHAN_CVAT.md` trước (tiêu chí thế nào là box đúng, khi nào xóa, khi nào sửa).
2. Mở `out/site/index.html` (hoặc **https://t14-review.pages.dev/**) và gõ `long` vào ô "Reviewer".
3. Duyệt **đủ 30 ảnh**. Bấm vào ảnh để phóng to, nhìn kỹ box nhỏ.
4. Mỗi ảnh chọn `approve` / `reject` / `edit` / `escalate` / `skip`.
   - Nếu chọn `edit`: điền `override_value` đúng 1 trong 4 nhãn `taxonomy.yaml`: `car`, `pedestrian`, `motorbike`, `sign`.
   - Với `reject` và `edit`: ghi một câu lý do vào ô note.
5. Bấm **Export decisions CSV** (đừng tải lại trang trước đó, trang không tự lưu). Lưu đè vào `out/poc/review_queue.csv` và copy sang `reports/review_queue.csv`.
6. Báo "đã xong" trong issue #7 để Nguyễn Xuân Việt chạy lại Export & Eval.
7. Ghi vào mục 07 `reports/group_report.md`: số `approve` / `reject` / `edit`, duyệt mất bao lâu, 2–3 ảnh khó nhất kèm lý do.

**Xong khi (#7):** `reports/review_queue.csv` đủ 30 dòng có `reviewer=long` và `reviewer_decision` hợp lệ; đã điền mục 07 `group_report.md`.

---

## 2. Nguyễn Xuân Việt — Issue #8: Code, Test & Chạy lại Export/Eval

**Mục tiêu:** sửa/viết test cho label-map validation & round-trip COCO↔YOLO (`pytest` xanh), sau đó chạy lại Export & Eval khi Long xong CSV.

### Phần 1: Code & Test (làm ngay)
1. Đọc `src/t14_tools.py` (`build_class_map`, `map_classes`, `validate_predictions`, `roundtrip_check`) và `tests/test_t14.py`.
2. Kiểm tra/bổ sung test cho:
   - Map class ID COCO-style sang `taxonomy.yaml` (`car`, `pedestrian`, `motorbike`, `sign`); bắt lỗi nếu map sai hoặc class ngoài taxonomy.
   - Round-trip COCO↔YOLO và đo invalid-label rate.
3. Chạy `python -m pytest -q` và `python -m tools.preflight` đảm bảo toàn bộ xanh.

### Phần 2: Chạy lại Export & Eval (sau khi Long xong #7)
1. Áp dụng `reports/review_queue.csv` của Long để xuất lại `out/poc/predictions.csv`, `out/poc/yolo/`, `out/poc/coco.json`, `out/poc/qa_report.json`.
2. Chạy eval:
   ```powershell
   python -m src.evaluate --track t14 --pred out/baseline/predictions.csv --run-name baseline --out reports/metrics_baseline.json
   python -m src.evaluate --track t14 --pred out/poc/predictions.csv --run-name poc --out reports/metrics_poc.json
   python -m src.compare reports/metrics_baseline.json reports/metrics_poc.json --out reports/table1_baseline_vs_poc.md
   ```
   *(Nếu chưa có `gt/` của giảng viên thì chạy dev eval bằng `python -m tools.t14_eval_dev --gold gold/gold_dev.csv --pred out/poc/predictions.csv` nếu có `gold/gold_dev.csv`.)*

**Xong khi (#8):** `pytest` & `preflight` xanh; có `out/poc/predictions.csv`, `reports/metrics_*.json`, `reports/table1_baseline_vs_poc.md` (hoặc ghi rõ nếu chưa có `gt/`).

---

## 3. Hoàng Quốc Chung — Issue #9: Failure Analysis & License

**Mục tiêu:** phân tích 2 failure mode bắt buộc và ghi rõ thông tin license.

1. Xem `out/baseline/predictions.csv` và `out/poc/review_queue.csv` (hoặc chạy `python -m tools.t14_failure_evidence --help` để xuất ảnh minh hoạ).
2. **Case 1 — Map class ID sai vẫn export hợp lệ:** tìm ví dụ box bị nhầm class do map COCO→taxonomy (lấy `item_id`, ảnh minh hoạ, class dự đoán vs class đúng).
3. **Case 2 — Confidence cao nhưng thiếu object nhỏ:** tìm ảnh model tự tin ở box lớn nhưng bỏ sót object nhỏ (< 150 px² như `sign`, `pedestrian` xa).
4. Lưu ảnh minh hoạ vào `reports/figures/`, viết root cause cho từng case vào mục 06 và 09 của `reports/group_report.md`.
5. Ghi rõ **license AGPL-3.0 của YOLO26** (dùng học tập được, pipeline thương mại đóng cần Enterprise) vào `reports/group_report.md`.

**Xong khi (#9):** có ít nhất 1 case mỗi loại kèm ảnh + `item_id` + root cause và mục license AGPL-3.0 trong `reports/group_report.md`.

---

## 4. Trần Thành Hùng — Issue #10: Metric Contract, Baseline & Tổng hợp Eval

**Mục tiêu:** kiểm tra lại `metric_contract.md`, baseline và điền bảng kết quả so sánh vào báo cáo nhóm.

1. Kiểm tra `reports/metric_contract.md` khớp `configs/tracks/t14.yaml` (`Detection F1@IoU0.5`, `conf_threshold`, `budget=30`, `seed=42`).
2. Kiểm tra `out/baseline/predictions.csv` và ghi rõ nguồn/version/license từ `fixtures/t14_auto_annotation/model_card_fixture.yaml` (nêu rõ là fixture mô phỏng, không phải inference thật).
3. Sau khi Việt (#8) chạy xong Eval: copy bảng từ `reports/table1_baseline_vs_poc.md` vào mục 04 và 05 của `reports/group_report.md`, viết nhận xét ΔF1@IoU0.5 (nếu chưa có `gt/`, ghi rõ "chưa có số chính thức").

**Xong khi (#10):** `reports/metric_contract.md` chốt đúng thông số; mục 04 & 05 của `reports/group_report.md` đầy đủ số liệu hoặc ghi chú rõ ràng.

---

## 5. Phạm Phú Nguyễn Hưng — Issue #11: Trang web, Tích hợp & Hồ sơ nhóm

**Mục tiêu:** đảm bảo hạ tầng web review, tích hợp trên `main` và chốt hồ sơ đóng góp của 6 thành viên.

1. Đảm bảo `out/site/index.html` / `https://t14-review.pages.dev/` hoạt động ổn định cho Long chấm tay.
2. Cập nhật đúng tên, vai trò, file phụ trách và `commit_shas` thực tế của 6 thành viên vào `reports/team_contributions.csv` (khớp `git shortlog -sne --all`).
3. Hoàn thiện `submission/README.md` và chạy `python -m pytest -q`, `python -m tools.preflight` lần cuối trước khi nộp.

**Xong khi (#11):** `reports/team_contributions.csv` và `submission/README.md` đầy đủ thông tin 6 người; `pytest` & `preflight` xanh.

---

## 6. Bùi Văn Thịnh — Issue #12: Decision Log & Hoàn thiện Báo cáo nhóm

**Mục tiêu:** chốt quyết định GO/HOLD/REWORK và rà soát toàn bộ `reports/group_report.md`.

1. Xem `out/poc/decision.json`, `out/poc/qa_report.json` và kết quả Failure Analysis (#9), Eval (#8/#10).
2. Chốt quyết định **GO / HOLD / REWORK** kèm `reason_code` và `owner` vào `reports/decision_log.csv`.
3. Rà soát toàn bộ `reports/group_report.md` (mục 00–09) đảm bảo nhất quán, không còn placeholder trống.

**Xong khi (#12):** `reports/decision_log.csv` có quyết định + `reason_code` + `owner`; `reports/group_report.md` hoàn chỉnh sẵn sàng nộp.

---

## Checklist trước khi push lên `fork/main` (mỗi người)

- [ ] `python -m pytest -q` xanh
- [ ] `python -m tools.preflight` xanh
- [ ] Báo cáo cá nhân `.md` đủ 4 mục, mọi số liệu có bằng chứng
- [ ] Đã điền mục của mình trong `reports/group_report.md` và dòng của mình trong `reports/team_contributions.csv`
- [ ] Không commit `.env`, `.dev.vars`, `out/`
