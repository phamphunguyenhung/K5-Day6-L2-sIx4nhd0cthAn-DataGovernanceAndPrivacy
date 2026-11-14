# Gold segmentation: cách vẽ và lưu để evaluator đọc được

Áp dụng cho A (vẽ gold) và B (kiểm chéo). Gold nằm ở `data/real/day02_traffic/gold/` (gitignored, KHÔNG commit ảnh/nhãn thật).

## Quy tắc vẽ
1. Vẽ TAY từ đầu hoặc sửa nháp AI rồi duyệt từng polygon. Nếu sửa nháp AI, ghi rõ "sửa từ nháp" trong `gold_meta.json` (thiên lệch neo vào nháp, evaluator sẽ cảnh báo).
2. Lớp theo Cityscapes 19 lớp (tên trong `tools/t14_seg_local.py::NAMES`: road, sidewalk, ..., bicycle). Giữ nguyên tên.
3. Mỗi xe/người/xe máy là 1 polygon riêng (instance). Vùng nền (đường, trời, cây...) 1 polygon/mảng.
4. Pixel không thuộc lớp nào: để trống (255). Đừng tô bừa cho kín.
5. Polygon sau chồng trước (vẽ nền trước, vật thể sau).

## Định dạng lưu (chọn 1)
- **COCO polygon**: `gold/gold_A.json` (CVAT: Export "COCO 1.0"). `file_name` phải trùng tên ảnh gốc, `width/height` đúng kích thước ảnh.
- **PNG label map**: `gold/A/<tên_ảnh>.png` 1 kênh uint8, giá trị = id lớp 0..18, 255 = trống; tùy chọn `<tên_ảnh>_inst.png` uint16 (0 = không phải instance).

B kiểm chéo lưu cùng định dạng ở `gold_B.json` hoặc `gold/B/`. Cùng tập ảnh, cùng tên, cùng kích thước với A, nếu lệch evaluator TỪ CHỐI chạy.

## Tách tập (bắt buộc)
Tạo `gold/gold_meta.json`:
```json
{"split": "heldout", "annotator": "A", "from_ai_draft": false, "minutes_per_image": {"drive_001": 14.0}}
```
- Ảnh gold phải là ảnh CHƯA dùng để chỉnh ngưỡng. Liệt kê ảnh đã dùng chỉnh ngưỡng vào `tuned_stems.txt` (mỗi dòng 1 tên ảnh); trùng thì evaluator từ chối.
- Chỉ `split = heldout` được chấm điểm.

## Ghi thời gian
Bấm giờ từng ảnh (phút, từ lúc mở đến lúc lưu), ghi vào `minutes_per_image`. Đây là số đo thật để so với thời gian vẽ nháp AI + sửa, không ước lượng.

## Chạy
```
python -m tools.t14seg.evaluate gold --pred out/.../coco.json --gold data/real/day02_traffic/gold/gold_A.json --split heldout --tuned-stems tuned_stems.txt --baseline <baseline> --out out/opt/agent8/gold.json
python -m tools.t14seg.evaluate iaa  --gold-a gold_A.json --gold-b gold_B.json
python -m tools.t14seg.evaluate proxy --pred-a <cfgA> --pred-b <cfgB>   # chưa có gold: chỉ PROXY, không phải độ chính xác
```
Với n < 30 ảnh, khoảng tin cậy rất rộng: báo cáo phải nói rõ "n nhỏ, chỉ là xu hướng".

## Cổng gap = 0 và polygon bị bỏ (luật mới)
- Pred có BẤT KỲ pixel hở (255) nào thì `gap_gate = FAIL`. Gold của A/B vẫn được phép để trống (255) ở vùng không thuộc lớp nào, cổng chỉ áp cho pred.
- Vùng mờ bị bỏ: pipeline ghi `dropped.json` cạnh pred (`{"drive_001": 3, ...}` = số polygon bỏ vì mờ). Evaluator báo `dropped_polygon_rate` cùng `instRecall`: bỏ nhiều thì recall giảm, phải đọc hai số này cùng nhau.
