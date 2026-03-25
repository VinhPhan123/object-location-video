import cv2
import threading
import numpy as np
from queue import Queue
from detection import YOLODetector


class CameraCapture:
    def __init__(self, camera_index, camera_name, detector, width=1280, height=720, fps=30):
        """
        Initialize camera capture thread
        
        Args:
            camera_index: Camera index
            camera_name: Camera name (Camera 1, Camera 2)
            detector: YOLODetector instance
            width: Capture width
            height: Capture height
            fps: Desired frames per second
        """
        self.camera_index = camera_index
        self.camera_name = camera_name
        self.detector = detector
        self.width = width
        self.height = height
        self.fps = fps
        
        self.frame_queue = Queue(maxsize=5)
        self.detection_queue = Queue(maxsize=5)
        self.running = False
        self.thread = None
        
        self.cap = None
        self.current_frame = None
        self.current_detections = None
        self.frame_count = 0
        self.fps_counter = 0
        self.fps_display = 0
        
    def start(self):
        """Start capture thread"""
        self.running = True
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_index}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop capture thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        if self.cap:
            self.cap.release()
    
    def _capture_loop(self):
        """Main capture loop running in thread"""
        import time
        
        frame_time = 1.0 / self.fps
        last_frame_time = time.time()
        fps_time = time.time()
        fps_frame_count = 0
        
        while self.running:
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            # Perform detection
            result, detections = self.detector.detect(frame)
            
            self.current_frame = frame
            self.current_detections = detections
            self.frame_count += 1
            fps_frame_count += 1
            
            # Calculate FPS
            current_time = time.time()
            if current_time - fps_time >= 1.0:
                self.fps_display = fps_frame_count
                fps_frame_count = 0
                fps_time = current_time
            
            # Adjust capture speed
            elapsed = time.time() - last_frame_time
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
            last_frame_time = time.time()
    
    def get_frame(self):
        """Get current frame"""
        return self.current_frame
    
    def get_track_ids(self):
        """Get list of current track IDs"""
        return self.detector.get_all_track_ids()
    
    def get_roi(self, track_id):
        """Get crop image of track ID"""
        return self.detector.get_roi(track_id)


class DualCameraController:
    def __init__(self, camera1_index=0, camera2_index=2, model_path="yolo26n.pt", 
                 camera_width=1280, camera_height=720):
        """
        Control 2 cameras simultaneously
        
        Args:
            camera1_index: Index of camera 1
            camera2_index: Index of camera 2
            model_path: Path to YOLO model
            camera_width: Camera width
            camera_height: Camera height
        """
        self.detector1 = YOLODetector(model_path=model_path)
        self.detector2 = YOLODetector(model_path=model_path)
        
        self.camera1 = CameraCapture(camera1_index, "Camera 1", self.detector1, camera_width, camera_height)
        self.camera2 = CameraCapture(camera2_index, "Camera 2", self.detector2, camera_width, camera_height)
    
    def start(self):
        """Start both cameras"""
        self.camera1.start()
        self.camera2.start()
    
    def stop(self):
        """Stop both cameras"""
        self.camera1.stop()
        self.camera2.stop()
    
    def get_frame(self, camera_num):
        """Get frame from camera (1 or 2)"""
        if camera_num == 1:
            return self.camera1.get_frame()
        else:
            return self.camera2.get_frame()
    
    def get_track_ids(self, camera_num):
        """Get list of track IDs"""
        if camera_num == 1:
            return self.camera1.get_track_ids()
        else:
            return self.camera2.get_track_ids()
    
    def get_roi(self, camera_num, track_id):
        """Get crop image of track"""
        if camera_num == 1:
            return self.camera1.get_roi(track_id)
        else:
            return self.camera2.get_roi(track_id)
