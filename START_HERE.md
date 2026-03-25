# 🚀 BẮT ĐẦU NHANH - YOLO Dual Camera Application

## 📋 Yêu cầu hệ thống

- Python 3.8+ 
- Windows/Linux/MacOS
- 2 camera USB hoặc built-in (tuỳ chọn)
- GPU với CUDA (tuỳ chọn nhưng hỗ trợ xử lý nhanh hơn)

---

## ⚡ Cài đặt nhanh (5 phút)

### Bước 1: Kiểm tra camera và dependencies
```bash
python test.py
```

Nếu tất cả là "✓ PASS", sang bước 2.

### Bước 2: Cài đặt dependencies (nếu chưa)
```bash
pip install -r requirements.txt
```

### Bước 3: Chạy ứng dụng
```bash
python main.py
```

Hoặc trên Windows, double-click file:
- `run.bat`

---

## 🎮 Sử dụng ứng dụng

### Giao diện
```
┌──────────────────────────────────────────────┐
│ ID List  │         Camera 1    │   Camera 2  │
│ [ID: 1]  │       [Video1]      │  [Video2]   │
│ [ID: 2]  │       [Video1]      │  [Video2]   │
│ [ID: 3]  │       [Video1]      │  [Video2]   │
├──────────┴────────────────────────────────────┤
│        Crop ID 1 Cam1  │  Crop ID 1 Cam2     │
└────────────────────────────────────────────────┘
```

### Hướng dẫn
1. **Chọn ID**: Click vào một ID trong danh sách bên trái
2. **Xem crop**: Hình ảnh crop sẽ hiển thị bên dưới
3. **Xem video**: Video từ 2 camera hiển thị bên phải với bounding boxes

---

## ⚙️ Tuỳ chỉnh

### Đổi camera index
Mở `config.py` và sửa:
```python
CAMERA1_INDEX = 2    # Thay 2 bằng số camera khác
CAMERA2_INDEX = 0    # Thay 0 bằng số camera khác
```

Để biết số index của camera, chạy:
```bash
python test.py
```

### Đổi model YOLO
Mở `config.py` và sửa:
```python
YOLO_MODEL = "yolo26n.pt"  # Các tuỳ chọn:
# yolo26n.pt (nhanh nhất)
# yolov8s.pt
# yolov8m.pt
# yolov8l.pt
# yolov8x.pt (chính xác nhất nhưng chậm)
```

### Điều chỉnh threshold
Mở `config.py`:
```python
CONFIDENCE_THRESHOLD = 0.4   # Tăng = ít phát hiện hơn (0.0-1.0)
IOU_THRESHOLD = 0.45         # Giảm = ít NMS hơn
```

---

## 🐛 Khắc phục sự cố

### ❌ "Camera not found"
- Chạy `python test.py` để kiểm tra camera index
- Đổi `CAMERA1_INDEX` và `CAMERA2_INDEX` trong `config.py`

### ❌ "Module not found"
```bash
pip install -r requirements.txt
```

### ❌ "Out of memory"
- Sử dụng model nhỏ: `yolo26n.pt`
- Giảm resolution camera trong `config.py`:
```python
CAMERA_WIDTH = 640   # Thay 1280
CAMERA_HEIGHT = 480  # Thay 720
```

### ❌ Chậm/Lag
- Giảm FPS: `CAMERA_FPS = 15` (thay 30 trong `config.py`)
- Hoặc sử dụng GPU (ensure CUDA available)

### ❌ PyQt5 error
```bash
pip install PyQt5
```

---

## 📂 Cấu trúc tệp

```
demo_app/
├── main.py              # Chạy cái này!
├── config.py            # Đổi cấu hình ở đây
├── detection.py         # Module YOLO (không cần sửa)
├── camera.py            # Module camera (không cần sửa)
├── test.py              # Kiểm tra setup
├── requirements.txt     # Dependencies
├── run.bat              # Double-click trên Windows
├── README.md            # Hướng dẫn chi tiết
└── START_HERE.md        # File này
```

---

## 🎯 Ví dụ sử dụng

### Ví dụ 1: Detect từ webcam mặc định
Chỉ cần chạy:
```bash
python main.py
```

### Ví dụ 2: Detect từ camera 1 và 2
Mở `config.py` và sửa:
```python
CAMERA1_INDEX = 0
CAMERA2_INDEX = 1
```
Rồi chạy: `python main.py`

### Ví dụ 3: Sử dụng model chính xác hơn
Mở `config.py`:
```python
YOLO_MODEL = "yolov8m.pt"  # Medium thay vì Nano
CONFIDENCE_THRESHOLD = 0.5  # Lọc cây vào
```

### Ví dụ 4: Chạy với cấu hình advanced
```bash
python main_advanced.py
```

---

## 💡 Tips & Tricks

1. **Giảm lag**: Camera detection xảy ra async, nên lag sẽ ít hơn
2. **Màu tracking**: Mỗi ID có màu khác nhau để dễ tracking
3. **Crop quality**: Crop lấy từ bounding box, nên quality tùy vào detection chính xác
4. **Monitor FPS**: Xem FPS hiển thị trên video để đánh giá performance

---

## 📞 Hỗ trợ

### Cần thêm tính năng?
- Lưu crop images: Xem `main.py` hàm `display_crop_image()`
- Record video: Thêm `cv2.VideoWriter` vào `camera.py`
- Export tracking data: Thêm file logging vào `detection.py`

---

## ✨ Tiếp theo

Sau khi quen thuộc:
1. Đọc `README.md` để hiểu chi tiết hơn
2. Xem code trong `main.py`, `detection.py`, `camera.py`
3. Customize theo nhu cầu

---

**Happy tracking! 🎉**
