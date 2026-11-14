# Benchmark — T14 Lane B (segmentation draft trên ảnh thật)

> **Đọc trước:** mọi số dưới đây là **số vận hành và proxy**, đo trên 50 ảnh dev của `data/real/day02_traffic`
> (UA-DETRAC, CC-BY-4.0) bằng máy CPU Windows 16 GB. **Chưa có gold nên chưa có độ chính xác nào** (mIoU, F1).
> Ô đánh dấu `⟦CHƯA ĐO⟧` là chỗ để hoàn thiện, kèm người phụ trách và lệnh để điền.
> Không dùng `drive_051..060` (split eval) để chọn ngưỡng.

## 1. Cái gì được so sánh

| | v1 (cũ) | v2 (hiện tại) |
|---|---|---|
| Lệnh | `python -m tools.t14_seg_local` | `python -m tools.t14seg.runner --config configs/t14_seg_v2.yaml` |
| Polygon | độc lập từng vùng + `approxPolyDP` | biên chung giữa 2 vùng đơn giản hóa 1 lần (`closure.py`) |
| Vùng mờ | giữ lại | bỏ (`keep_conf` 0.35, `keep_area` 100, `keep_solid` 0.8), pixel rơi về nền bên dưới |
| Luật khít mép | còn hở | **không hở nào** (gap = 0 khi vẽ lại ở opacity 100%) |
| Cache / resume | không | có (sha256 ảnh + hash cấu hình + weights) |

## 2. Kết quả chính: v1 vs v2 (50 ảnh dev, seed 42)

Sinh tự động từ file kết quả: `python -m tools.t14seg.bench_report --a out/seg_local --b out/seg_v2/run`.

| Chỉ số | v1 (run 49418fa7) | v2 (run cee60c67) |
|---|---|---|
| Số ảnh | 50 | 50 |
| Tổng thời gian (s) | 358.0 | 452.0 (46 ảnh chạy mới + 4 resume) |
| Giây/ảnh, trung bình | 7.12 | 9.02 |
| RAM đỉnh (MB) | chưa ghi | 5697.8 |
| Số instance | 1319 | 1090 |
| Số polygon | 6880 | 5766 |
| **Gap file import CVAT, trung bình (%)** | **0.1566** | **0.0** |
| **Gap file import CVAT, tối đa (%)** | **0.3176** | **0.0** |
| Gap == 0 (cổng validator) | không đạt | **đạt, 0 vi phạm** |
| Lệch polygon vẽ lại vs nhãn gốc, trung bình (%) | 0.064 ⚠ | 1.117 |
| Lệch tối đa (%) | 0.309 ⚠ | 4.752 |

**Cách đọc đúng:**
- **Gap là cải thiện đo được và chắc chắn:** từ 0.157% xuống 0.0% trên cả 50 ảnh.
- ⚠ **Hai dòng "lệch" không so sánh trực tiếp được:** v1 đo trên đường bao chính xác (eps 0), v2 đo ở eps 0.8 với
  trần 1500 điểm/ảnh để file CVAT nhẹ. Cùng điều kiện (eps 0.8) bản v1 lệch tối đa 0.727% (agent export, 50 ảnh).
  v2 lệch nhiều hơn là cái giá của việc đóng kín khe và lấp vùng bỏ.
- **Thời gian không phải bằng chứng "nhanh/chậm hơn":** mỗi giá trị là một lần chạy, máy có lúc chạy song song nhiều
  tiến trình khác. Phần ổn định là cache (mục 4).
- **Ít instance hơn ≠ tốt hơn:** v2 bỏ 229 instance (17%) vì mờ. Không có gold nên **không biết** trong đó bao nhiêu
  là đối tượng thật bị bỏ nhầm (recall giảm) hay nhiễu được loại đúng.

## 3. Các thí nghiệm thành phần (proxy, 4 ảnh dev: 001, 002, 010, 030)

Số đo gốc nằm ở `out/opt/*` (cục bộ, gitignore, không commit). Chạy lại để kiểm.

### 3.1 Cấu hình instance (YOLO-seg), tổng 4 ảnh

| Cấu hình | Instance | Instance nhỏ (<400 px²) | Ghi chú |
|---|---|---|---|
| s @960, conf 0.25 | 76 | 6 | bỏ sót nhiều xe ở ảnh kẹt xe |
| **s @1280, conf 0.25 (đang dùng)** | 103 | 16 | |
| s @1536, conf 0.25 | 114 | 21 | chậm ~3x |
| m @1280, conf 0.25 | 113 | 25 | |
| l @1280, conf 0.25 | 102 | 25 | |
| s @1280, conf 0.15 | 135 | 37 | nhiều nhiễu tiềm năng |
| s @1280, conf 0.35 | 84 | 8 | |
| s @1280, conf 0.20 + tile 320 | 131 | 34 | chậm ~3x, không bật |
| m @1280, conf 0.20 + tile 320 | 142 | 40 | |

Đếm instance là **proxy độ phủ**, không phải recall. `drive_030` (kẹt xe): YOLO@640 thấy 14 xe, @1280 thấy 32.

### 3.2 Gap polygon (closure), ngân sách ~1500 điểm/ảnh, eps 2.5

| Ảnh | Gap v1 | Gap v2 (biên dùng chung) |
|---|---|---|
| drive_001 | 0.84% | 0 |
| drive_010 | 0.77% | 0 |
| drive_030 | 0.67% | 0 |

### 3.3 Tinh chỉnh biên bằng MobileSAM (74 instance, cạnh box ≥24px)

IoU SAM–YOLO trung bình 0.82 (4 instance < 0.6). 6/74 mask bị cổng an toàn loại. Biên SAM **không mịn hơn** YOLO
(1.075 so với 1.046). 6.6 s/instance trên CPU đang tải. **Kết luận: tắt SAM**, chưa chứng minh tốt hơn YOLO.
`sam2.1_b` ⟦CHƯA ĐO⟧.

### 3.4 Xếp hạng review (S = 0.35U + 0.25D + 0.15G + 0.10C + 0.10F + 0.05N), 611 polygon

Đặc trưng D (bất đồng nguồn) chi phối; C gần như không tác động (Kendall tau 0.99). Nhiễu trọng số ±30%: tau trung bình
0.97. Ngưỡng bỏ 0.4 cho 39 polygon (6.4%). **Chưa nối vào pipeline v2**; ngưỡng là giả thuyết.

## 4. Hiệu năng vận hành (5 ảnh cố định, 2 thread)

| Lần chạy | Giây/ảnh | RAM đỉnh |
|---|---|---|
| Vòng lặp cũ, không cache | 38.1 | 4.66 GB |
| Runner, cache lần 1 | 42.4 | 3.52 GB |
| **Runner, cache lần 2** | **0.87** | 1.0 GB |
| Resume (kết quả còn nguyên, đo riêng) | 0.7 s cho cả 5 ảnh (5/5 `resumed`) | 1.0 GB |

Chạy lại cùng ảnh + cấu hình nhanh hơn khoảng 49 lần nhờ cache (42.4 s → 0.87 s/ảnh). Lần chạy đầu **không** nhanh hơn.
Lần bench resume trong `bench.json` bị lỗi ghi nhận (`resumed: 0`), nên số resume lấy từ lần chạy lại riêng của agent 9;
chưa chạy lại cả bộ bench. 2 worker song song cho kết
quả nhiễu (151 s/ảnh do tải máy), chưa kết luận; giữ `workers: 1`. Thời gian ở mục này đo khi máy đang chạy nhiều tiến
trình nên là cận trên. Ảnh `drive_057` (eval) chỉ dùng để đo thời gian, không dùng để chọn cấu hình.

## 5. Phần cần hoàn thiện (khung)

| # | Hạng mục | Trạng thái | Chặn bởi | Người | Lệnh / cách điền |
|---|---|---|---|---|---|
| B1 | mIoU theo lớp + 95% CI (bootstrap theo ảnh) v1 vs v2 | ⟦CHƯA ĐO⟧ | chưa có gold | Chung (A) vẽ, Tech Lead chạy | `python -m tools.t14seg.evaluate gold ...` (xem `docs/team/gold_format.md`) |
| B2 | Instance F1@0.5 và @0.75, boundary F | ⟦CHƯA ĐO⟧ | chưa có gold | như B1 | như B1 |
| B3 | Đồng thuận người–người (IoU giữa gold của A và B) | ⟦CHƯA ĐO⟧ | cần gold của A + kiểm chéo 20% của B | Long (B) | `python -m tools.t14seg.evaluate iaa ...` |
| B4 | Tỷ lệ instance bị bỏ vì mờ có phải đối tượng thật | ⟦CHƯA ĐO⟧ | cần gold | Tech Lead | `dropped.json` + chế độ gold |
| B5 | Tune `keep_conf/keep_area/keep_solid` trên dev (mỗi lần chỉ 1 ngưỡng, ghi bảng) | ⟦CHƯA ĐO⟧ | cần gold | Tech Lead | sweep trên dev, **không** dùng eval |
| B6 | Thời gian review mỗi ảnh (người sửa nháp vs vẽ tay) | ⟦CHƯA ĐO⟧ | cần người thật thao tác | Chung (A), Long (B) | ghi `sec_review` mỗi ảnh |
| B7 | Import thử vào CVAT (COCO 1.0 / CVAT 1.1 / mask 1.1) | ⟦CHƯA ĐO⟧ | cần task CVAT do Tech Lead cấp | Tech Lead | `export_report.json` đang ghi `cvat_import_verified: false` |
| B8 | Đo lại thời gian trên máy rảnh, 1/2/3 worker | ⟦CHƯA ĐO⟧ | máy đang tải | Tech Lead | `python -m tools.t14seg.runner --bench` |
| B9 | Nối `review_rank`, `layering`, semantic policy (agent 2) vào pipeline | ⟦CHƯA LÀM⟧ | semantic chưa đo agreement/biên | Tech Lead | hook trong `configs/t14_seg_v2.yaml` |
| B10 | Lane A: baseline vs PoC F1 trên fixture | ⟦CHƯA ĐO⟧ | cần overlay GT của giảng viên (`evaluate` exit 2 là đúng) | giảng viên | `docs/INSTRUCTOR_SETUP.md` |

Khi điền: ghi lệnh đã chạy, đường dẫn output, ngày giờ vào `PROGRESS.md`; số nào chưa đo giữ `⟦CHƯA ĐO⟧`.

## 6. Tái lập

```powershell
$env:CVAT_TOOLS_SRC = "<learning>/cvat/src"          # tool CVAT ngoài, chỉ đọc
python -m tools.t14seg.runner --config configs/t14_seg_v2.yaml          # 50 ảnh dev, có cache/resume
python -c "from tools.t14seg.cvat_export import export_cvat; export_cvat('out/seg_v2/run','out/seg_v2/cvat_import', eps=0.8, masks=True)"
python -m tools.t14seg.bench_report --a out/seg_local --b out/seg_v2/run
```

Cần weights SegFormer-B2 và `yolo26s-seg.pt` (xem `docs/HUONG_DAN_NHOM_T14.md`, mục Setup). Ảnh thật và weights **không** nằm trong repo.

## 7. Giới hạn (nói thẳng)

- Không có gold → không có số độ chính xác; mọi bảng ở trên là vận hành hoặc proxy.
- Ngưỡng bỏ vùng mờ là giả thuyết, chưa tune.
- Một lần chạy mỗi cấu hình, chưa lặp lại nên chưa có độ dao động giữa các lần.
- Chạy 50 ảnh lần đầu của v2 từng bị ngắt ở ảnh thứ 2 (nghi thiếu RAM do tiến trình khác); chạy lại thành công.
- Giấy phép: Ultralytics YOLO là AGPL-3.0; UA-DETRAC là CC-BY-4.0.
