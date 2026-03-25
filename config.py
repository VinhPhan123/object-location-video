"""
Configuration file for YOLO Dual Camera Application
Chỉnh sửa các giá trị dưới đây để tuỳ chỉnh ứng dụng
"""

# ============================================
# CAMERA CONFIGURATION
# ============================================
CAMERA1_INDEX = 0           # Index camera 1 (0, 1, 2, etc.)
CAMERA2_INDEX = 1           # Index camera 2

CAMERA_WIDTH = 1280         # Độ rộng capture
CAMERA_HEIGHT = 720         # Độ cao capture
CAMERA_FPS = 30             # Frame per second mong muốn

# ============================================
# YOLO CONFIGURATION
# ============================================
YOLO_MODEL = "yolo26n.pt"   # Model YOLO có sẵn trong folder
                            # Options: yolo26n.pt (local), hoặc download: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
CONFIDENCE_THRESHOLD = 0.4  # Confidence threshold (0.0 - 1.0)
IOU_THRESHOLD = 0.45        # IoU threshold for NMS (0.0 - 1.0)

# Device configuration (auto sẽ tự chọn GPU nếu có)
DEVICE = "auto"             # Options: auto, cpu, cuda, cuda:0

# ============================================
# GUI CONFIGURATION
# ============================================
WINDOW_WIDTH = 1600         # Độ rộng cửa sổ chính
WINDOW_HEIGHT = 1000        # Độ cao cửa sổ chính
WINDOW_TITLE = "YOLO Dual Camera - ID Detection & Crop"

# Video display size
VIDEO_DISPLAY_WIDTH = 600
VIDEO_DISPLAY_HEIGHT = 450

# Crop display size
CROP_DISPLAY_WIDTH = 250
CROP_DISPLAY_HEIGHT = 200

# ============================================
# UPDATE RATE (Frames per second)
# ============================================
VIDEO_UPDATE_FPS = 30       # Update rate cho video streams
ID_LIST_UPDATE_FPS = 10     # Update rate cho ID list
CROP_UPDATE_FPS = 10        # Update rate cho crop images

# ============================================
# TRACKING CONFIGURATION
# ============================================
TRACKING_TIMEOUT = 5.0      # Thời gian (giây) để xóa track không còn xuất hiện
ID_COLOR_COUNT = 9          # Số lượng màu sử dụng cho các ID khác nhau

# ============================================
# LOGGING
# ============================================
DEBUG_MODE = False          # Hiển thị debug info
LOG_FILE = "app_debug.log"  # File log

# ============================================
# PERFORMANCE
# ============================================
MAX_QUEUE_SIZE = 5          # Kích thước tối đa của frame queue
