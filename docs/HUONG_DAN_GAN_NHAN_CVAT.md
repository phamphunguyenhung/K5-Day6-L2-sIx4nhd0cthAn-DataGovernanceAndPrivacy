# Hướng dẫn gắn nhãn ảnh bằng tay (CVAT)

## 1. Mục đích

Gắn nhãn (annotation) là vẽ một **hình chữ nhật (bounding box)** quanh mỗi vật thể trong ảnh và chọn **tên class** cho nó.
Trong T14, việc này dùng cho hai mục đích:

- Tạo **gold set tự vẽ** (`gold/gold_dev.csv`) để đánh giá nội bộ trên split dev (`tools.t14_eval_dev`).
- Sửa các box máy đoán sai khi review (xem mục 6).

Taxonomy gồm 4 class, tên phải đúng từng ký tự: `car`, `pedestrian`, `motorbike`, `sign` (`fixtures/t14_auto_annotation/taxonomy.yaml`).

## 2. Chuẩn bị (người quản lý máy chủ làm một lần)

| Việc | Cách làm |
|---|---|
| Bật CVAT | Mở Docker Desktop, chờ các container `cvat_*` chạy xong. |
| Địa chỉ cho cả nhóm | Lấy IP Wi-Fi của máy chủ bằng `ipconfig`, mọi người vào `http://<IP>:8080`. IP hiện tại ghi nhận: `10.53.152.59`, có thể đổi khi đổi mạng. |
| Điều kiện | Mọi người cùng một mạng Wi-Fi với máy chủ. Chỉ người cùng mạng mới vào được, không phải toàn Internet. |
| Mở cổng | Windows có thể chặn cổng 8080. Mở tường lửa cho cổng 8080, chỉ profile Private. |
| Tài khoản | Đăng nhập admin, tạo một tài khoản riêng cho mỗi thành viên. Không chia sẻ mật khẩu admin. |

## 3. Tạo task và đưa ảnh lên

Bộ ảnh: `fixtures/t14_auto_annotation/images/` (60 ảnh `aa_00.jpg` … `aa_59.jpg`).

1. Đăng nhập CVAT → **Projects → Create new project**, đặt tên `T14 - Gắn nhãn xe`.
2. Thêm đúng 4 label: `car`, `pedestrian`, `motorbike`, `sign`.
3. Vào project → **Create a new task**, đặt tên (ví dụ `Lô 1 - 30 ảnh`).
4. **Select files → My computer** → chọn ảnh. Chia 60 ảnh thành 2 lô × 30 ảnh để 2 người làm song song.
5. Bấm **Submit & Open**.

## 4. Cách gắn nhãn

1. Mở task → bấm **Job #...** được giao.
2. Chọn công cụ hình chữ nhật (phím **N**).
3. Chọn label (ví dụ `car`) → **Shape**.
4. Bấm hai góc đối diện của vật thể: góc trên-trái, rồi góc dưới-phải.
5. Ảnh kế tiếp: phím **D**; ảnh trước: phím **F**.
6. Lưu thường xuyên bằng **Ctrl + S**.
7. Xong tất cả ảnh: **Menu → Finish the job**.

## 5. Quy tắc gắn nhãn

- Khung **ôm sát** vật thể, không rộng thừa, không cắt mất một phần.
- Vật nhìn thấy rõ thì **phải gắn**, kể cả vật nhỏ.
- Một vật chỉ một khung, không vẽ trùng.
- Không chắc là vật gì thì **không đoán**. Hỏi người phụ trách hoặc chọn `escalate` khi review.
- Ảnh có mặt người hoặc biển số rõ nét: không ghi lại hay chia sẻ thông tin cá nhân ra ngoài nhóm.

## 6. Tiêu chí khi review box máy đoán

Áp dụng khi duyệt trên trang review (`approve / reject / edit / escalate / skip`):

| Quyết định | Khi nào |
|---|---|
| `approve` | Box ôm đúng vật thể **và** tên class đúng. |
| `reject` | Không có vật thể trong box, hoặc box lệch hẳn khỏi vật thể. |
| `edit` | Có vật thể nhưng sai class (ghi class đúng vào ô override), hoặc box lệch nhẹ. |
| `escalate` | Không xác định được vật thể (mờ, che khuất, quá nhỏ). |
| `skip` | Chưa xem. Phải quay lại xem trước khi xuất CSV. |

Với `reject` và `edit`, luôn ghi **một câu lý do** để người chốt cuối kiểm tra lại được.

## 7. Kiểm tra và xuất kết quả (người phụ trách)

1. Mở lại vài ảnh ngẫu nhiên của mỗi người, kiểm tra khung và class.
2. Task → **Actions → Export task dataset** → chọn **YOLO 1.1** (hoặc COCO 1.0).
3. Giải nén file `.zip`, đổi tên class cho khớp `taxonomy.yaml` nếu cần.

## 8. Xử lý sự cố

| Hiện tượng | Cách xử lý |
|---|---|
| Máy khác không vào được | Kiểm tra cùng Wi-Fi, đúng IP, đã mở cổng 8080. |
| Trang báo lỗi 502 | Docker chưa chạy xong. Chờ 1–2 phút rồi tải lại; vẫn lỗi thì báo người quản lý máy. |
| Mất kết nối khi đang vẽ | Bấm Ctrl+S trước, tải lại trang. Phần đã lưu vẫn còn. |
| Máy chủ chậm | Giảm số người làm cùng lúc, đóng bớt chương trình khác. |
