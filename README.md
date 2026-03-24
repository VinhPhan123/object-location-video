# YOLO Dual Camera Detection & Crop Application

Ứng dụng Python sử dụng YOLO để phát hiện vật thể từ 2 camera, hiển thị tracking ID và crop images.

## Tính năng

- 🎥 Xử lý video từ 2 camera đồng thời
- 🔍 Phát hiện vật thể và tracking ID bằng YOLO
- 📋 Danh sách ID được phát hiện
- 🖼️ Hiển thị crop images từ mỗi tracking ID
- 🎨 Giao diện GUI với PyQt5

## Cấu trúc dự án

```
demo_app/
├── main.py           # File chính chạy ứng dụng GUI
├── detection.py      # Module xử lý YOLO detection
├── camera.py         # Module xử lý camera capture
├── requirements.txt  # Dependencies
└── README.md         # File này
```

## Cài đặt

### 1. Tạo virtual environment (tuỳ chọn nhưng khuyến nghị)

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 3. Tải model YOLO (nếu chưa có)

Model sẽ được tải tự động lần đầu chạy. Có thể tải trước:

```bash
python -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"
```

## Sử dụng

### Chạy ứng dụng

```bash
python main.py
```

### Công dụng các phần giao diện

1. **Cột trái (ID List)**
   - Hiển thị danh sách tất cả tracking ID được phát hiện từ 2 camera
   - Click vào ID để chọn xem crop image
   - Hiển thị số lượng active tracks

2. **Phần trung tâm (Video Streams)**
   - Cửa sổ Camera 1: Luồng video từ camera 1 với bounding boxes
   - Cửa sổ Camera 2: Luồng video từ camera 2 với bounding boxes
   - Mỗi vật thể được đánh dấu bằng ID, class name, và confidence score

3. **Phần dưới (Crop Images)**
   - Crop từ Camera 1: Hình ảnh crop của tracking ID được chọn từ camera 1
   - Crop từ Camera 2: Hình ảnh crop của tracking ID được chọn từ camera 2

## Cấu hình

Mở file `main.py` và chỉnh sửa các tham số trong hàm `main()`:

```python
window = YOLODualCameraApp(
    camera1_idx=2,        # Index camera 1 (thay đổi theo số camera)
    camera2_idx=0,        # Index camera 2
    model_path="yolo26n.pt"  # Model YOLO (n, s, m, l, x)
)
```

### Các model YOLO có sẵn
- `yolo26n.pt` - Nano (nhanh nhất, độ chính xác thấp)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large (chậm nhất, độ chính xác cao)

## Cấu trúc code

### detection.py

- `YOLODetector`: Class chính xử lý YOLO detection và tracking
  - `detect()`: Phát hiện vật thể trên frame
  - `get_all_track_ids()`: Lấy danh sách ID
  - `get_roi()`: Lấy crop image của ID

### camera.py

- `CameraCapture`: Quản lý capture từ 1 camera trong thread riêng
- `DualCameraController`: Quản lý 2 camera cùng lúc

### main.py

- `YOLODualCameraApp`: Ứng dụng GUI chính
  - `update_video_frames()`: Cập nhật luồng video
  - `update_id_list()`: Cập nhật danh sách ID
  - `update_crop_images()`: Cập nhật crop images

## Khắc phục sự cố

### Lỗi "không mở được camera"
- Kiểm tra index camera: Sử dụng `cv2.VideoCapture(i)` để test từng index
- Đảm bảo camera không bị chiếm bởi ứng dụng khác

### Hiệu suất thấp
- Giảm resolution camera
- Sử dụng model nhỏ hơn (yolov8n thay vì yolov8x)
- Sử dụng GPU nếu có (CUDA)

### Lỗi import PyQt5
```bash
pip install PyQt5
```

## Ghi chú

- Ứng dụng xử lý 2 camera trong 2 thread riêng biệt để tối đa hóa hiệu suất
- Tracking ID được cập nhật theo real-time
- Crop images được lưu trong memory (không lưu file)
- Có thể thoát ứng dụng bằng cách đóng cửa sổ

## Phát triển thêm

Có thể mở rộng dự án:
- Lưu crop images vào file
- Thêm chế độ record video
- Tích hợp database để lưu tracking history
- Thêm tính năng search/filter by class
- Export thống kê tracking

---

**Tác giả**: YOLO Demo Team  
**Phiên bản**: 1.0  
**Ngày tạo**: 2026-03-22
"# object-location-video" 
