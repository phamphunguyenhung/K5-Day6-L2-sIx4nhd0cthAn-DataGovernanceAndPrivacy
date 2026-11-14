# DECISIONS — T14 (Tech Lead chốt, không hỏi lại)

| # | Quyết định | Ghi chú |
|---|---|---|
| D1 | CVAT: dry-run + file local. KHÔNG ghi remote (không tạo task, không upload ảnh, không tìm token). | Chế độ ghi chỉ bật khi Tech Lead cấp task đích và cho phép rõ. |
| D2 | PII: T14 không yêu cầu che. Ảnh thật chỉ ở local, không commit/upload. Không phụ thuộc `src/redaction/` (T07). | `data/real/` đã gitignore. |
| D3 | Taxonomy 4 lớp: car=0, pedestrian=1, motorbike=2, sign=3. truck/bus/bicycle = OOD: drop + audit. | Xác thực theo `taxonomy.yaml` thực. |
| D4 | Provenance: `ai_draft_unreviewed` → `human_approved/edited/rejected` (+reviewer, timestamp, diff). Sidecar audit nếu format không hỗ trợ. | Không gọi draft là GT. |
| D5 | Không sửa repo `cvat`. Gọi CLI `--json` qua subprocess, retry/backoff + timeout phía Day06. Không đoán lệnh: `--help` trước. | Không đưa token vào log/argv. |
| D6 | Thứ tự: select → detect → adapter → guard → rank → review file → export → QA, toàn bộ dry-run. | Số ảnh kiểm bằng filesystem: 60 jpg (50 dev / 10 eval). |

## Ghi chú nhóm (ngoài T14)
- **Bài segmentation (Day03):** khi kéo opacity lên hết, mask phải khít mép đối tượng. ~~Chỉ đối tượng KHÔNG có class mới được để hở.~~ T14 là bbox, không áp dụng yêu cầu pixel-perfect này.
- **CẬP NHẬT (luật hiện hành cho segmentation draft):** KHÔNG HỞ NÀO (gap = 0 khi vẽ lại ở opacity 100%). Vùng/đối tượng nào mờ (conf thấp, biên không rõ) thì BỎ, phần bị bỏ được lấp bằng nền bên dưới, không để trống. Đánh đổi: bỏ nhiều thì recall giảm, nên luôn báo cả số bị bỏ.
