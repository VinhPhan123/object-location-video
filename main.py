import sys
import cv2
import numpy as np
import time
from datetime import datetime
from typing import Tuple, Optional, Dict, List
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QListWidget, QListWidgetItem, QLabel, QTextEdit
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor
from PyQt5.QtCore import Qt, QTimer
from camera import DualCameraController

# Import feature extraction and matching modules
try:
    from feature_extraction import SuperPointExtractor
    FEATURE_EXTRACTION_AVAILABLE = True
except ImportError as e:
    print(f"[Main] Warning: Could not import SuperPointExtractor: {e}")
    FEATURE_EXTRACTION_AVAILABLE = False

try:
    from feature_matching import LightGlueMatcher, IDAssociator
    FEATURE_MATCHING_AVAILABLE = True
except ImportError as e:
    print(f"[Main] Warning: Could not import LightGlueMatcher: {e}")
    FEATURE_MATCHING_AVAILABLE = False

try:
    from triangulation import StereoTriangulator
    TRIANGULATION_AVAILABLE = True
except ImportError as e:
    print(f"[Main] Warning: Could not import StereoTriangulator: {e}")
    TRIANGULATION_AVAILABLE = False

try:
    from tracked_object import TrackedObject, ObjectTracker
    TRACKED_OBJECT_AVAILABLE = True
except ImportError as e:
    print(f"[Main] Warning: Could not import TrackedObject: {e}")
    TRACKED_OBJECT_AVAILABLE = False

# Import utils for contour filtering and drawing
try:
    from utils import get_contour_mask, filter_keypoints_by_contour, draw_contour_on_frame, draw_contours_for_all_objects
    CONTOUR_FILTERING_AVAILABLE = True
except ImportError as e:
    print(f"[Main] Warning: Could not import contour utilities: {e}")
    CONTOUR_FILTERING_AVAILABLE = False


class YOLODualCameraApp(QMainWindow):
    def __init__(self, camera1_idx=0, camera2_idx=1, model_path="yolov8n.pt"):
        super().__init__()
        self.camera_controller = DualCameraController(
            camera1_index=camera1_idx,
            camera2_index=camera2_idx,
            model_path=model_path
        )
        
        self.selected_track_id = None
        self.selected_camera = None  # 1 = Camera 1, 2 = Camera 2
        self.selected_crop_snapshot = None
        
        # Cache for crop images
        self.crop_cache_cam1 = {}
        self.crop_cache_cam2 = {}
        
        # Track previous IDs to detect new IDs
        self.previous_ids_cam1 = set()
        self.previous_ids_cam2 = set()
        
        # FPS tracking
        self.frame_times_cam1 = []  # List of recent frame timestamps for FPS calculation
        self.frame_times_cam2 = []
        self.current_fps_cam1 = 0
        self.current_fps_cam2 = 0
        self.max_fps_history = 30  # Keep last 30 frames for smooth FPS calculation
        
        # Log buffers for feature extraction/matching and triangulation
        self.feature_log_buffer = []  # Max 50 lines
        self.triangulation_log_buffer = []  # Max 50 lines
        self.max_log_lines = 50
        
        # Store matched points for visualization
        self.matched_pts_cam1 = None  # Full frame matched points for cam1
        self.matched_pts_cam2 = None  # Full frame matched points for cam2
        
        # Object-oriented tracking system
        self.object_tracker = ObjectTracker() if TRACKED_OBJECT_AVAILABLE else None
        self.current_objects_cam1 = {}  # {track_id: TrackedObject}
        self.current_objects_cam2 = {}  # {track_id: TrackedObject}
        
        # Initialize feature extraction and matching modules
        self.feature_extractor = None
        self.feature_matcher = None
        self.id_associator = None
        self.triangulator = None
        self.features_cache_cam1 = {}  # Cache features: {track_id: features_dict}
        self.features_cache_cam2 = {}
        self.matched_pairs = {}  # Cache matched ID pairs: {(id_cam1, id_cam2): MatchResult}
        
        if FEATURE_EXTRACTION_AVAILABLE:
            try:
                self.feature_extractor = SuperPointExtractor(device="cuda")
                self.add_feature_log("SuperPointExtractor initialized successfully")
            except Exception as e:
                self.add_feature_log(f"ERROR: Failed to initialize SuperPointExtractor: {e}")
        
        if FEATURE_MATCHING_AVAILABLE:
            try:
                self.feature_matcher = LightGlueMatcher(device="cuda")
                self.id_associator = IDAssociator()
                self.add_feature_log("LightGlueMatcher and IDAssociator initialized successfully")
            except Exception as e:
                self.add_feature_log(f"ERROR: Failed to initialize matching modules: {e}")
        
        if TRIANGULATION_AVAILABLE:
            try:
                self.triangulator = StereoTriangulator()
                self.add_triangulation_log("StereoTriangulator initialized successfully")
            except Exception as e:
                self.add_triangulation_log(f"ERROR: Failed to initialize triangulator: {e}")
        
        self.initUI()
        self.setup_timers()
        
        # Start cameras
        try:
            self.camera_controller.start()
        except Exception as e:
            print(f"Error starting cameras: {e}")
            self.statusBar().showMessage(f"Error: {e}")
    
    def _create_section_label(self, text, font_size=10, bold=True):
        """Helper to create consistent section labels"""
        label = QLabel(text)
        font = QFont()
        font.setBold(bold)
        font.setPointSize(font_size)
        label.setFont(font)
        return label
    
    def _is_point_on_contour(self, frame: np.ndarray, point: Tuple[int, int], 
                             bbox: Tuple[int, int, int, int],
                             tolerance: int = 5) -> bool:
        """
        Kiểm tra xem một điểm có nằm trên contour/biên của vật thể không.
        
        Args:
            frame: Full frame
            point: (x, y) point in frame coordinates
            bbox: (x1, y1, x2, y2) bounding box
            tolerance: Pixel tolerance around contour
            
        Returns:
            True nếu điểm nằm trên contour, False nếu trong background
        """
        if not CONTOUR_FILTERING_AVAILABLE:
            return True  # If filtering not available, keep all points
        
        try:
            x, y = int(point[0]), int(point[1])
            x1, y1, x2, y2 = bbox
            
            # Crop region from frame
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            if x1 >= x2 or y1 >= y2:
                return True
            
            region = frame[y1:y2, x1:x2].copy()
            
            # Get contour mask for this region
            contour_mask = get_contour_mask(
                region,
                canny_threshold1=50,
                canny_threshold2=150,
                dilation_kernel_size=3,
                dilation_iterations=2
            )
            
            if contour_mask is None:
                return True
            
            # Check if point falls on contour in region coordinates
            px_region = x - x1
            py_region = y - y1
            
            h, w = contour_mask.shape
            if 0 <= px_region < w and 0 <= py_region < h:
                return contour_mask[int(py_region), int(px_region)] > 0
            
            return True
        except Exception as e:
            # If error, keep point
            return True
    
    def update_tracked_objects(self, camera_num):
        """
        Create/update TrackedObject instances for current detections.
        
        Args:
            camera_num: Camera number (1 or 2)
        """
        if not TRACKED_OBJECT_AVAILABLE or self.object_tracker is None:
            return
        
        detector = self.camera_controller.detector1 if camera_num == 1 else self.camera_controller.detector2
        track_ids = self.camera_controller.get_track_ids(camera_num)
        current_objects = self.current_objects_cam1 if camera_num == 1 else self.current_objects_cam2
        features_cache = self.features_cache_cam1 if camera_num == 1 else self.features_cache_cam2
        
        # Track which IDs are still active
        active_ids = set(track_ids)
        
        # Remove dead objects
        dead_ids = set(current_objects.keys()) - active_ids
        for dead_id in dead_ids:
            if self.object_tracker:
                self.object_tracker.remove_object(dead_id, camera_num)
            current_objects.pop(dead_id, None)
        
        # Create or update tracked objects
        for track_id in track_ids:
            track_info = detector.get_track_info(track_id)
            if not track_info:
                continue
            
            bbox = track_info["bbox"]
            class_name = track_info["class_name"]
            confidence = track_info["conf"]
            color = self.get_color_for_id(track_id)  # Get color for consistent visualization
            
            if track_id not in current_objects:
                # Create new TrackedObject with proper color
                obj = TrackedObject(
                    track_id=track_id,
                    camera_id=camera_num,
                    bbox=bbox,
                    class_name=class_name,
                    confidence=confidence,
                    color=color
                )
                current_objects[track_id] = obj
                if self.object_tracker:
                    self.object_tracker.add_object(obj)
            else:
                # Update existing object
                obj = current_objects[track_id]
                obj.bbox = bbox
                obj.confidence = confidence
            
            # Add matched points from cache if available
            if track_id in features_cache:
                features = features_cache[track_id]
                if 'matched_points' in features:
                    matched_pts = features['matched_points']
                    descriptors = features.get('matched_descriptors', None)
                    
                    # Clear previous matched points
                    obj.matched_points_cam = []
                    obj.descriptors = []
                    
                    # Add new matched points
                    if descriptors is not None and len(descriptors) > 0:
                        obj.add_matched_points_batch(matched_pts, descriptors)
                    else:
                        obj.add_matched_points_batch(matched_pts)
            
            # Set gray color for unmatched objects
            if not obj.is_matched():
                obj.set_unmatched_color()
    
    def get_tracked_objects_summary(self):
        """Get summary of all currently tracked objects."""
        if not TRACKED_OBJECT_AVAILABLE or self.object_tracker is None:
            return None
        
        summary = {
            'total_objects': len(self.current_objects_cam1) + len(self.current_objects_cam2),
            'cam1_objects': len(self.current_objects_cam1),
            'cam2_objects': len(self.current_objects_cam2),
            'matched_pairs': len(self.object_tracker.matched_pairs),
            'cam1_details': [obj.to_dict() for obj in self.current_objects_cam1.values()],
            'cam2_details': [obj.to_dict() for obj in self.current_objects_cam2.values()],
        }
        return summary
    
    def print_tracked_objects_info(self):
        """Print information about all tracked objects."""
        if not TRACKED_OBJECT_AVAILABLE or self.object_tracker is None:
            print("[ObjectTracker] Tracked object system not available")
            return
        
        print("\n" + "="*80)
        print("TRACKED OBJECTS INFORMATION")
        print("="*80)
        
        summary = self.get_tracked_objects_summary()
        if not summary:
            print("No objects tracked")
            return
        
        print(f"\n--- SUMMARY ---")
        print(f"Total Objects: {summary['total_objects']}")
        print(f"Camera 1 Objects: {summary['cam1_objects']}")
        print(f"Camera 2 Objects: {summary['cam2_objects']}")
        print(f"Matched Pairs: {summary['matched_pairs']}")
        
        print(f"\n--- CAMERA 1 OBJECTS ---")
        for obj_info in summary['cam1_details']:
            print(f"  ID: {obj_info['id']}, Class: {obj_info['class_name']}, "
                  f"Conf: {obj_info['confidence']:.2f}, Points: {obj_info['matched_points_count']}, "
                  f"3D: {obj_info['valid_3d_points_count']}")
        
        print(f"\n--- CAMERA 2 OBJECTS ---")
        for obj_info in summary['cam2_details']:
            print(f"  ID: {obj_info['id']}, Class: {obj_info['class_name']}, "
                  f"Conf: {obj_info['confidence']:.2f}, Points: {obj_info['matched_points_count']}, "
                  f"3D: {obj_info['valid_3d_points_count']}")
        
        print(f"\n--- MATCHED PAIRS ---")
        if self.object_tracker.matched_pairs:
            for obj1, obj2 in self.object_tracker.matched_pairs:
                print(f"  Cam1 ID{obj1.id} ({obj1.class_name}) <-> Cam2 ID{obj2.id} ({obj2.class_name})")
        else:
            print("  No matched pairs")
        
        print("="*80 + "\n")


    def initUI(self):
        """Initialize the GUI"""
        self.setWindowTitle("YOLO Dual Camera - ID Detection & Crop")
        self.setGeometry(100, 100, 1600, 1000)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        
        # ===== LEFT PANEL: ID Lists =====
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(280)
        
        # Camera 1 List
        left_layout.addWidget(self._create_section_label("Camera 1 IDs", 10, True))
        self.id_list_cam1 = QListWidget()
        self.id_list_cam1.itemClicked.connect(lambda item: self.on_id_selected(item, camera=1))
        left_layout.addWidget(self.id_list_cam1)
        
        # Camera 2 List
        left_layout.addWidget(self._create_section_label("Camera 2 IDs", 10, True))
        self.id_list_cam2 = QListWidget()
        self.id_list_cam2.itemClicked.connect(lambda item: self.on_id_selected(item, camera=2))
        left_layout.addWidget(self.id_list_cam2)
        
        # Info label
        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("background-color: #f0f0f0; padding: 5px;")
        left_layout.addWidget(self.info_label)
        
        main_layout.addWidget(left_panel)
        
        # ===== RIGHT PANEL: Video streams and crops =====
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)
        
        # Video streams layout
        video_layout = QHBoxLayout()
        
        # Camera 1 video
        cam1_container = self._create_video_container("Camera 1")
        self.camera1_label = cam1_container['label']
        video_layout.addWidget(cam1_container['widget'])
        
        # Camera 2 video
        cam2_container = self._create_video_container("Camera 2")
        self.camera2_label = cam2_container['label']
        video_layout.addWidget(cam2_container['widget'])
        
        right_layout.addLayout(video_layout, 2)
        
        # Log displays layout
        log_layout = QHBoxLayout()
        
        # Left log: Feature extraction and matching
        left_log_container = self._create_log_container("Feature Extraction & Matching Log")
        self.log_left_display = left_log_container['text_edit']
        log_layout.addWidget(left_log_container['widget'])
        
        # Right log: Triangulation and 3D computation
        right_log_container = self._create_log_container("3D Triangulation & Position Log")
        self.log_right_display = right_log_container['text_edit']
        log_layout.addWidget(right_log_container['widget'])
        
        right_layout.addLayout(log_layout, 1)
        main_layout.addWidget(right_panel, 1)
        
        self.statusBar().showMessage("Initializing...")
    
    def _create_video_container(self, title):
        """Create a video display container"""
        container = QWidget()
        layout = QVBoxLayout()
        container.setLayout(layout)
        
        title_label = self._create_section_label(title, 11, True)
        layout.addWidget(title_label)
        
        video_label = QLabel()
        video_label.setMinimumSize(600, 450)
        video_label.setScaledContents(False)
        video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(video_label)
        
        return {'widget': container, 'label': video_label}
    
    def _create_log_container(self, title):
        """Create a log display container with text area"""
        container = QWidget()
        layout = QVBoxLayout()
        container.setLayout(layout)
        
        title_label = self._create_section_label(title, 9, True)
        layout.addWidget(title_label)
        
        text_edit = QTextEdit()
        text_edit.setMinimumSize(300, 200)
        text_edit.setMaximumHeight(250)
        text_edit.setReadOnly(True)
        text_edit.setStyleSheet("border: 1px solid gray; background-color: #f8f8f8; font-family: Courier; font-size: 9pt;")
        text_edit.setLineWrapMode(QTextEdit.NoWrap)
        layout.addWidget(text_edit)
        
        return {'widget': container, 'text_edit': text_edit}
    
    def setup_timers(self):
        """Setup update timers"""
        # Video frames timer (30 FPS)
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.update_video_frames)
        self.video_timer.start(33)
        
        # ID list timer (10 FPS)
        self.id_timer = QTimer()
        self.id_timer.timeout.connect(self.update_id_list)
        self.id_timer.start(100)
        
        # Feature processing timer (2 FPS - process features every 500ms)
        self.feature_timer = QTimer()
        self.feature_timer.timeout.connect(self.process_features_and_matching)
        self.feature_timer.start(500)
        
        # Logs timer (5 FPS for display updates)
        self.log_timer = QTimer()
        self.log_timer.timeout.connect(self.update_log_displays)
        self.log_timer.start(200)
    
    def update_video_frames(self):
        """Update video streams from both cameras with FPS tracking"""
        # Track FPS for camera 1
        self.frame_times_cam1.append(time.time())
        if len(self.frame_times_cam1) > self.max_fps_history:
            self.frame_times_cam1.pop(0)
        
        if len(self.frame_times_cam1) > 1:
            time_diff = self.frame_times_cam1[-1] - self.frame_times_cam1[0]
            if time_diff > 0:
                self.current_fps_cam1 = (len(self.frame_times_cam1) - 1) / time_diff
        
        frame1 = self.camera_controller.get_frame(1)
        if frame1 is not None:
            self.update_tracked_objects(1)  # Update TrackedObject instances for camera 1
            self.display_frame_with_detections(frame1, self.camera1_label, 1)
        
        # Track FPS for camera 2
        self.frame_times_cam2.append(time.time())
        if len(self.frame_times_cam2) > self.max_fps_history:
            self.frame_times_cam2.pop(0)
        
        if len(self.frame_times_cam2) > 1:
            time_diff = self.frame_times_cam2[-1] - self.frame_times_cam2[0]
            if time_diff > 0:
                self.current_fps_cam2 = (len(self.frame_times_cam2) - 1) / time_diff
        
        frame2 = self.camera_controller.get_frame(2)
        if frame2 is not None:
            self.update_tracked_objects(2)  # Update TrackedObject instances for camera 2
            self.display_frame_with_detections(frame2, self.camera2_label, 2)
    
    def display_frame_with_detections(self, frame, label, camera_num):
        """Display frame with bounding boxes, keypoints, matched points, and contours"""
        frame_display = frame.copy()
        track_ids = self.camera_controller.get_track_ids(camera_num)
        current_objects = self.current_objects_cam1 if camera_num == 1 else self.current_objects_cam2
        
        # Determine which cache to use
        features_cache = self.features_cache_cam1 if camera_num == 1 else self.features_cache_cam2
        
        for track_id in track_ids:
            detector = self.camera_controller.detector1 if camera_num == 1 else self.camera_controller.detector2
            track_info = detector.get_track_info(track_id)
            
            if track_info:
                bbox = track_info["bbox"]
                x1, y1, x2, y2 = bbox
                class_name = track_info["class_name"]
                conf = track_info["conf"]
                
                # Get color from TrackedObject or fall back to ID-based color
                if track_id in current_objects:
                    color = current_objects[track_id].color
                else:
                    color = self.get_color_for_id(track_id)
                
                # Draw bounding box
                cv2.rectangle(frame_display, (x1, y1), (x2, y2), color, 2)
                
                # Draw contour of the object (using color intensity differences)
                if CONTOUR_FILTERING_AVAILABLE:
                    try:
                        frame_display = draw_contour_on_frame(
                            frame_display,
                            bbox,
                            color=color,
                            line_thickness=1,
                            canny_threshold1=40,
                            canny_threshold2=120
                        )
                    except Exception as e:
                        print(f"[Display] Error drawing contour for track {track_id}: {e}")
                
                # Draw label
                label_text = f"ID:{track_id} {class_name} {conf:.2f}"
                (text_w, text_h), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                cv2.rectangle(
                    frame_display,
                    (x1, y1 - text_h - baseline - 5),
                    (x1 + text_w + 5, y1),
                    color,
                    thickness=-1,
                )
                cv2.putText(
                    frame_display,
                    label_text,
                    (x1 + 2, y1 - baseline - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                
                # Draw keypoints if available
                if track_id in features_cache:
                    features = features_cache[track_id]
                    keypoints = features.get('keypoints', [])
                    scores = features.get('scores', [])
                    
                    # Offset keypoints by bbox top-left corner (if they're in crop coords)
                    for idx, kpt in enumerate(keypoints):
                        kx, ky = int(kpt[0]), int(kpt[1])
                        
                        # Try to draw in frame coordinates if within bbox
                        if 0 <= kx < (x2-x1) and 0 <= ky < (y2-y1):
                            frame_kx = x1 + kx
                            frame_ky = y1 + ky
                        else:
                            frame_kx, frame_ky = kx, ky
                        
                        # Draw keypoint circle
                        cv2.circle(frame_display, (frame_kx, frame_ky), 3, (0, 255, 0), -1)
                        cv2.circle(frame_display, (frame_kx, frame_ky), 4, (0, 255, 255), 1)
                        
                        # Draw score
                        if idx < len(scores):
                            score = scores[idx]
                            cv2.putText(frame_display, f"{score:.2f}", (frame_kx+5, frame_ky-5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 1)
        
        # Draw matched points on full frame, colored by bounding box
        matched_pts = self.matched_pts_cam1 if camera_num == 1 else self.matched_pts_cam2
        
        if matched_pts is not None:
            # Build bounding box map for faster lookup: pt_index -> (color, bbox)
            bbox_map = {}  # pt_index -> (color, bbox)
            for track_id in track_ids:
                detector = self.camera_controller.detector1 if camera_num == 1 else self.camera_controller.detector2
                track_info = detector.get_track_info(track_id)
                if track_info:
                    bbox = track_info["bbox"]
                    x1, y1, x2, y2 = bbox
                    
                    # Get color from TrackedObject or fall back to ID-based color
                    if track_id in current_objects:
                        color = current_objects[track_id].color
                    else:
                        color = self.get_color_for_id(track_id)
                    
                    # Find which matched points fall within this bbox
                    for pt_idx, pt in enumerate(matched_pts):
                        x, y = int(pt[0]), int(pt[1])
                        if x1 <= x <= x2 and y1 <= y <= y2:
                            bbox_map[pt_idx] = (color, bbox)
            
            # Draw matched points ONLY if they are within bounding boxes AND on contour
            points_drawn = 0
            points_filtered = 0
            
            for pt_idx in bbox_map:  # Only iterate through points that are inside bboxes
                pt = matched_pts[pt_idx]
                pt_color, bbox = bbox_map[pt_idx]
                x, y = int(pt[0]), int(pt[1])
                
                if 0 <= x < frame_display.shape[1] and 0 <= y < frame_display.shape[0]:
                    # Check if point is on contour (if filtering enabled)
                    if self._is_point_on_contour(frame, pt, bbox, tolerance=5):
                        # Draw matched point marker: larger circle with cross
                        cv2.circle(frame_display, (x, y), 5, pt_color, 2)
                        cv2.drawMarker(frame_display, (x, y), pt_color, 
                                      markerType=cv2.MARKER_CROSS, markerSize=8, thickness=2)
                        points_drawn += 1
                    else:
                        points_filtered += 1
            
            if points_filtered > 0:
                # Display filtering stats
                filter_text = f"Points: {points_drawn} drawn, {points_filtered} filtered (background)"
                cv2.putText(frame_display, filter_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (0, 200, 255), 1, cv2.LINE_AA)
        
        # Add FPS display to frame
        fps_text = f"FPS: {self.current_fps_cam1:.1f}" if camera_num == 1 else f"FPS: {self.current_fps_cam2:.1f}"
        cv2.putText(frame_display, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Resize frame for display
        frame_resized = self._resize_frame(frame_display, 600, 450)
        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        qt_image = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qt_image))
    
    def _resize_frame(self, frame, target_width, target_height):
        """Resize frame maintaining aspect ratio"""
        height, width = frame.shape[:2]
        ratio = min(target_width / width, target_height / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        return cv2.resize(frame, (new_width, new_height))
    
    def update_id_list(self):
        """Update ID lists for each camera"""
        ids_cam1 = self.camera_controller.get_track_ids(1)
        ids_cam2 = self.camera_controller.get_track_ids(2)
        
        # Auto-crop new IDs
        new_ids_cam1 = set(ids_cam1) - self.previous_ids_cam1
        new_ids_cam2 = set(ids_cam2) - self.previous_ids_cam2
        
        for track_id in new_ids_cam1:
            roi = self.camera_controller.get_roi(1, track_id)
            if roi is not None:
                self.crop_cache_cam1[track_id] = roi.copy()
        
        for track_id in new_ids_cam2:
            roi = self.camera_controller.get_roi(2, track_id)
            if roi is not None:
                self.crop_cache_cam2[track_id] = roi.copy()
        
        # Clean up non-existent track IDs from caches
        ids_cam1_set = set(ids_cam1)
        ids_cam2_set = set(ids_cam2)
        
        # Remove cache entries for tracks that no longer exist (camera 1)
        removed_ids_cam1 = list(self.crop_cache_cam1.keys()) + list(self.features_cache_cam1.keys())
        for track_id in removed_ids_cam1:
            if track_id != "full_frame" and track_id not in ids_cam1_set:
                self.crop_cache_cam1.pop(track_id, None)
                self.features_cache_cam1.pop(track_id, None)
        
        # Remove cache entries for tracks that no longer exist (camera 2)
        removed_ids_cam2 = list(self.crop_cache_cam2.keys()) + list(self.features_cache_cam2.keys())
        for track_id in removed_ids_cam2:
            if track_id != "full_frame" and track_id not in ids_cam2_set:
                self.crop_cache_cam2.pop(track_id, None)
                self.features_cache_cam2.pop(track_id, None)
        
        # Update tracking
        self.previous_ids_cam1 = set(ids_cam1)
        self.previous_ids_cam2 = set(ids_cam2)
        
        # Update list widgets with objects info
        self._update_list_widget(self.id_list_cam1, ids_cam1, 1)
        self._update_list_widget(self.id_list_cam2, ids_cam2, 2)
        
        # Update info
        info_text = f"Total IDs: {len(ids_cam1) + len(ids_cam2)}\n"
        info_text += f"Camera 1: {len(ids_cam1)}\n"
        info_text += f"Camera 2: {len(ids_cam2)}"
        self.info_label.setText(info_text)
    
    def _update_list_widget(self, list_widget, track_ids, camera_num):
        """Helper to update a list widget with track IDs and object names with colors"""
        current_objects = self.current_objects_cam1 if camera_num == 1 else self.current_objects_cam2
        current_items = set()
        
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            try:
                current_items.add(int(item.text().split(":")[0].strip()))
            except:
                pass
        
        # Add new IDs
        for track_id in track_ids:
            if track_id not in current_items:
                obj = current_objects.get(track_id)
                if obj:
                    # Get object name and color
                    obj_name = obj.class_name
                    color = obj.color  # BGR tuple
                    item_text = f"{track_id}: {obj_name}"
                else:
                    obj_name = "Unknown"
                    color = (128, 128, 128)  # Gray if object not found
                    item_text = f"{track_id}: {obj_name}"
                
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, track_id)
                
                # Set text color from BGR to RGB
                b, g, r = color
                rgb_color = QColor(r, g, b)
                item.setForeground(rgb_color)
                
                list_widget.addItem(item)
        
        # Update existing items and remove IDs no longer present
        for track_id in current_items:
            if track_id not in track_ids:
                for i in range(list_widget.count()):
                    item = list_widget.item(i)
                    try:
                        if int(item.text().split(":")[0].strip()) == track_id:
                            list_widget.takeItem(i)
                            break
                    except:
                        pass
            else:
                # Update existing item with current color and name
                for i in range(list_widget.count()):
                    item = list_widget.item(i)
                    try:
                        item_id = int(item.text().split(":")[0].strip())
                        if item_id == track_id:
                            obj = current_objects.get(track_id)
                            if obj:
                                # Update text and color
                                item_text = f"{track_id}: {obj.class_name}"
                                item.setText(item_text)
                                
                                # Update color
                                b, g, r = obj.color
                                rgb_color = QColor(r, g, b)
                                item.setForeground(rgb_color)
                            break
                    except:
                        pass
    
    def process_features_and_matching(self):
        """Extract features from FULL FRAMES and match with bounding box filtering"""
        if not FEATURE_EXTRACTION_AVAILABLE or not FEATURE_MATCHING_AVAILABLE:
            return
        
        try:
            # Get current frames
            frame1 = self.camera_controller.get_frame(1)
            frame2 = self.camera_controller.get_frame(2)
            
            if frame1 is None or frame2 is None:
                return
            
            ids_cam1 = self.camera_controller.get_track_ids(1)
            ids_cam2 = self.camera_controller.get_track_ids(2)
            
            if not ids_cam1 or not ids_cam2:
                self.add_feature_log("Waiting for detections from both cameras...")
                return
            
            # Check if we need to extract features:
            # Skip extraction if all detected objects are already matched (optimization)
            if TRACKED_OBJECT_AVAILABLE and self.object_tracker:
                unmatched_cam1 = [id_ for id_ in ids_cam1 
                                 if id_ in self.current_objects_cam1 
                                 and not self.current_objects_cam1[id_].is_matched()]
                unmatched_cam2 = [id_ for id_ in ids_cam2 
                                 if id_ in self.current_objects_cam2 
                                 and not self.current_objects_cam2[id_].is_matched()]
                
                # If only matched objects exist, skip feature extraction
                if not unmatched_cam1 and not unmatched_cam2:
                    self.add_feature_log("All objects matched - skipping feature extraction")
                    return
            
            # Extract features from FULL FRAMES once
            cache_key = "full_frame"
            if cache_key not in self.features_cache_cam1:
                try:
                    features1 = self.feature_extractor.extract_features(frame1)
                    self.features_cache_cam1[cache_key] = features1
                    num_kpts = len(features1.get('keypoints', []))
                    self.add_feature_log(f"Cam1 Full Frame: Extracted {num_kpts} keypoints")
                except Exception as e:
                    self.add_feature_log(f"Cam1 feature extraction error: {str(e)[:50]}")
                    return
            
            if cache_key not in self.features_cache_cam2:
                try:
                    features2 = self.feature_extractor.extract_features(frame2)
                    self.features_cache_cam2[cache_key] = features2
                    num_kpts = len(features2.get('keypoints', []))
                    self.add_feature_log(f"Cam2 Full Frame: Extracted {num_kpts} keypoints")
                except Exception as e:
                    self.add_feature_log(f"Cam2 feature extraction error: {str(e)[:50]}")
                    return
            
            # Match features on full frames
            try:
                feat1 = self.features_cache_cam1[cache_key]
                feat2 = self.features_cache_cam2[cache_key]
                
                self.add_feature_log(f"Matching full frames...")
                match_result = self.feature_matcher.match_features(feat1, feat2)
                
                if not match_result or match_result.num_matches == 0:
                    self.add_feature_log(f"No matches found on full frames")
                    return
                
                total_matches = match_result.num_matches
                self.add_feature_log(f"Full frame matches: {total_matches} points")
                
                # Now filter matches by bounding boxes
                matched_pts_cam0 = match_result.matched_pts_cam0
                matched_pts_cam1 = match_result.matched_pts_cam1
                
                # Store matched points for visualization on video frames
                self.matched_pts_cam1 = matched_pts_cam0
                self.matched_pts_cam2 = matched_pts_cam1
                
                # Compare ID pairs and find which bboxes have most matched points
                # Skip already-matched pairs to avoid unnecessary re-matching (optimization)
                from feature_matching import filter_matches_by_bboxes
                
                best_matches = {}  # (id_cam1, id_cam2) -> (valid_pts_cam0, valid_pts_cam1, count)
                
                # Build set of already-matched pairs for quick lookup
                matched_pairs_set = set()
                if TRACKED_OBJECT_AVAILABLE and self.object_tracker:
                    for obj1, obj2 in self.object_tracker.matched_pairs:
                        matched_pairs_set.add((obj1.id, obj2.id))
                
                for id_cam1 in ids_cam1:
                    track_info1 = self.camera_controller.detector1.get_track_info(id_cam1)
                    if not track_info1:
                        continue
                    
                    bbox1 = track_info1["bbox"]
                    
                    for id_cam2 in ids_cam2:
                        # Skip if this pair is already matched
                        if (id_cam1, id_cam2) in matched_pairs_set:
                            self.add_feature_log(f"ID:{id_cam1}↔ID:{id_cam2} already matched - skipping")
                            continue
                        
                        track_info2 = self.camera_controller.detector2.get_track_info(id_cam2)
                        if not track_info2:
                            continue
                        
                        bbox2 = track_info2["bbox"]
                        
                        # Filter matches within these bboxes (only for unmatched pairs)
                        valid_pts_cam0, valid_pts_cam1, num_valid = filter_matches_by_bboxes(
                            matched_pts_cam0, matched_pts_cam1, bbox1, bbox2
                        )
                        
                        if num_valid >= 1:  # Need at least 1 match for triangulation
                            best_matches[(id_cam1, id_cam2)] = (valid_pts_cam0, valid_pts_cam1, num_valid)
                
                # Log results and link matched objects
                if best_matches:
                    for (id_cam1, id_cam2), (pts0, pts1, count) in best_matches.items():
                        obj1_class = self.camera_controller.detector1.get_track_info(id_cam1).get("class_name", "?")
                        obj2_class = self.camera_controller.detector2.get_track_info(id_cam2).get("class_name", "?")
                        self.add_feature_log(f"✓ ID:{id_cam1}({obj1_class})↔ID:{id_cam2}({obj2_class}) - {count} inlier points")
                        
                        # Link matched objects and assign sequential color
                        if TRACKED_OBJECT_AVAILABLE and self.object_tracker:
                            obj1 = self.current_objects_cam1.get(id_cam1)
                            obj2 = self.current_objects_cam2.get(id_cam2)
                            if obj1 and obj2:
                                self.object_tracker.match_objects(obj1, obj2)
                        
                        # Triangulate
                        if TRIANGULATION_AVAILABLE and self.triangulator:
                            try:
                                self.add_triangulation_log(f"Triangulating {count} points from ID:{id_cam1}↔ID:{id_cam2}...")
                                
                                # Compute 3D points
                                result = self.triangulator.triangulate_matched_points(pts0, pts1)
                                
                                if result and result.get('points_3d') is not None and len(result['points_3d']) > 0:
                                    pts_3d = result['points_3d']
                                    errors = result.get('reprojection_errors', [])
                                    valid_mask = result.get('valid_points', np.ones(len(pts_3d), dtype=bool))
                                    
                                    # Calculate centroid from valid points
                                    if np.any(valid_mask):
                                        valid_pts = pts_3d[valid_mask]
                                        centroid_3d = np.mean(valid_pts, axis=0)
                                        mean_error = np.mean(errors[valid_mask]) if len(errors) > 0 else 0
                                        num_valid = np.sum(valid_mask)
                                    else:
                                        centroid_3d = np.mean(pts_3d, axis=0)
                                        mean_error = np.mean(errors) if len(errors) > 0 else 0
                                        num_valid = len(pts_3d)
                                    
                                    self.add_triangulation_log(
                                        f"3D Position: X={centroid_3d[0]:.1f}, Y={centroid_3d[1]:.1f}, Z={centroid_3d[2]:.1f} mm"
                                    )
                                    self.add_triangulation_log(
                                        f"Valid points: {num_valid}/{count}, Error: {mean_error:.2f}px"
                                    )
                            except Exception as e:
                                self.add_triangulation_log(f"Triangulation error: {str(e)[:60]}")
                else:
                    self.add_feature_log(f"No object pairs matched (need ≥3 inlier points)")
                    
            except Exception as e:
                self.add_feature_log(f"Matching error: {str(e)[:60]}")
                
        except Exception as e:
            self.add_feature_log(f"Feature processing error: {str(e)[:70]}")
    
    def add_feature_log(self, message):
        """Add message to feature extraction/matching log"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_message = f"[{timestamp}] {message}"
        self.feature_log_buffer.append(log_message)
        
        # Keep only last N lines
        if len(self.feature_log_buffer) > self.max_log_lines:
            self.feature_log_buffer = self.feature_log_buffer[-self.max_log_lines:]
    
    def add_triangulation_log(self, message):
        """Add message to triangulation log"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_message = f"[{timestamp}] {message}"
        self.triangulation_log_buffer.append(log_message)
        
        # Keep only last N lines
        if len(self.triangulation_log_buffer) > self.max_log_lines:
            self.triangulation_log_buffer = self.triangulation_log_buffer[-self.max_log_lines:]
    
    def update_log_displays(self):
        """Update log displays in UI"""
        # Update left log (feature extraction/matching)
        if self.feature_log_buffer:
            left_log_text = "\n".join(self.feature_log_buffer)
            self.log_left_display.setText(left_log_text)
            # Auto-scroll to bottom
            scrollbar = self.log_left_display.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
        
        # Update right log (triangulation)
        if self.triangulation_log_buffer:
            right_log_text = "\n".join(self.triangulation_log_buffer)
            self.log_right_display.setText(right_log_text)
            # Auto-scroll to bottom
            scrollbar = self.log_right_display.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
    
    def on_id_selected(self, item, camera=None):
        """Handle ID selection from list"""
        try:
            track_id = int(item.text().split(":")[0].strip())
            self.selected_track_id = track_id
            self.selected_camera = camera
            camera_name = f"Camera {camera}" if camera else "Any"
            self.statusBar().showMessage(f"Selected ID: {track_id} from {camera_name}")
            
            # Get snapshot at time of selection
            if camera == 1:
                self.selected_crop_snapshot = self.crop_cache_cam1.get(track_id, None)
            elif camera == 2:
                self.selected_crop_snapshot = self.crop_cache_cam2.get(track_id, None)
        except:
            pass
    
    def get_color_for_id(self, track_id):
        """Get color based on track ID for consistent visualization"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128),
        ]
        return colors[track_id % len(colors)]
    
    def closeEvent(self, event):
        """Handle application closing"""
        self.video_timer.stop()
        self.id_timer.stop()
        self.feature_timer.stop()
        self.log_timer.stop()
        self.camera_controller.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    
    try:
        window = YOLODualCameraApp(
            camera1_idx=2,
            camera2_idx=0,
            model_path="yolo26n.pt"
        )
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Application error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
