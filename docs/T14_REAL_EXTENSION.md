# T14 mở rộng ảnh thật (Lane B) - ghi chú trung thực

## 1. Hai lane, không được nhập nhằng
| | Lane A (chính thức) | Lane B (mở rộng, ảnh thật) |
|---|---|---|
| Dữ liệu | fixture synthetic của bài lab | 50 ảnh UA-DETRAC (`data/real/`, gitignored) |
| Nguồn dự đoán | `simulated-detector-fixture`: KHÔNG phải inference thật | SegFormer-B2 Cityscapes (ONNX) + yolo26s-seg, chạy cục bộ |
| Giá trị | chấm điểm theo contract Day06 | bản nháp polygon để người review sửa (`ai_draft`, `source=auto`) |

Lane B không thay thế, không sửa contract Lane A. Kết quả Lane B là nháp chưa review, không phải ground truth.

## 2. Tái lập (offline, CPU)
```
python tools/t14_seg_local.py --seed 42        # ghi out/seg_local/ (cần weights cục bộ, CVAT_TOOLS_SRC)
python tools/t14_export_cvat.py                # ghi out/seg_local/cvat_import/ (COCO 1.0 + CVAT 1.1)
python -m pytest tests/test_t14seg_safety.py -q
```
Weights, ảnh thật và `.env.local` nằm ngoài git. Tham số chính xác xem `--help` của từng tool.

## 3. Bảng bằng chứng PoC
| Hạng mục | Đã đo / chưa đo / chặn bởi | Ghi chú |
|---|---|---|
| Không mạng/CVAT ghi trong code | Đã đo | `test_no_network_or_cvat_write` (quét import + động từ HTTP) |
| Không đọc `fixtures/*/gt/` | Đã đo | `test_no_gt_path_access` |
| Không đường dẫn cá nhân trong file tracked | Đã đo | test quét `git ls-files` |
| Ảnh thật/weights/.env.local bị gitignore | Đã đo | `git check-ignore` |
| Provenance `ai_draft` + `source=auto` mọi polygon | Đã đo | XML và COCO của run 49418fa7 |
| SHA256 ảnh khớp manifest | Đã đo (10 ảnh đầu) | 40 ảnh còn lại chưa kiểm trong test |
| Seed 42 | Đã đo một phần | `run_info.json` ghi seed 42; tính tất định chỉ kiểm trên dữ liệu tổng hợp, chưa chạy lại model |
| Gap = 0 | Đã đo: chưa đạt | export_report của run 49418fa7: max gap_render_pct 0.3176%, TB 0.1566% trên 50 ảnh (đo từ export_report.csv); test xfail |
| Độ chính xác so với gold | Chưa đo, chặn bởi: không có gold | chỉ có chỉ số proxy |
| Import thử vào CVAT | Chưa đo, chặn bởi: cấm gọi CVAT trong lane này | |

## 4. Giới hạn
- Không có gold; mọi con số là proxy, không phải độ chính xác.
- Chưa import thử CVAT thật.
- Ultralytics (YOLO) theo AGPL-3.0: cần xem lại nghĩa vụ trước khi phân phối.
- UA-DETRAC theo CC-BY-4.0: phải ghi nguồn; ảnh không đưa ra ngoài máy.
- Vùng 'thing' của SegFormer thô, xe dày có thể bị gộp, ảnh đêm có thể sót đối tượng nhỏ.

## 5. Luật khít mép (người dùng, bản mới)
**Không được có hở nào (gap = 0).** Vùng mờ/không chắc thì bỏ và lấp bằng nền bên dưới; không để lỗ trống. (Luật cũ "chỉ đối tượng không có class mới được hở" đã bị thay thế.)

Trạng thái: dữ liệu export hiện tại (run 49418fa7) vẫn còn gap (đo lại từ export_report.csv: TB 0.1566%, max 0.3176%), nên `test_exported_draft_has_no_gap` được đánh dấu `xfail` có lý do. Test chỉ xanh thật sau khi pipeline mới được tích hợp và export lại.
