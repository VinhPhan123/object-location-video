import torch
import numpy as np
from ultralytics import YOLO
import time


class YOLODetector:
    def __init__(self, model_path="yolo26n.pt", device="auto", conf=0.4, iou=0.45):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model file
            device: Device to use (auto, cpu, cuda, cuda:0)
            conf: Confidence threshold
            iou: IoU threshold for NMS
        """
        if device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.track_data = {}  # {track_id: {bbox, frame_info, last_seen}}
        
    def detect(self, frame):
        """
        Perform detection and tracking on frame
        
        Args:
            frame: Input frame from camera
            
        Returns:
            result: YOLO result object
            detections: List of detection info {id, class, conf, bbox, roi}
        """
        results = self.model.track(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            persist=True,
            verbose=False,
        )
        
        result = results[0]
        detections = []
        current_time = time.time()
        detected_ids = set()
        
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                track_id = int(box.id[0]) if box.id is not None else -1
                
                if track_id >= 0:
                    detected_ids.add(track_id)
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
                    
                    cls_id = int(box.cls[0]) if box.cls is not None else -1
                    conf = float(box.conf[0]) if box.conf is not None else 0.0
                    class_name = self.model.names.get(cls_id, str(cls_id))
                    
                    # Extract ROI
                    roi = frame[y1:y2, x1:x2].copy() if (x1 < x2 and y1 < y2) else None
                    
                    # Update track data
                    self.track_data[track_id] = {
                        "bbox": (x1, y1, x2, y2),
                        "class_name": class_name,
                        "class_id": cls_id,
                        "conf": conf,
                        "roi": roi,
                        "last_seen": current_time,
                        "frame_count": self.track_data.get(track_id, {}).get("frame_count", 0) + 1
                    }
                    
                    detections.append({
                        "id": track_id,
                        "class_name": class_name,
                        "class_id": cls_id,
                        "conf": conf,
                        "bbox": (x1, y1, x2, y2),
                        "roi": roi
                    })
        
        # Remove tracks no longer detected
        ids_to_remove = [tid for tid in self.track_data if tid not in detected_ids]
        for tid in ids_to_remove:
            del self.track_data[tid]
        
        return result, detections
    
    def get_all_track_ids(self):
        """Get list of all current track IDs"""
        return sorted(list(self.track_data.keys()))
    
    def get_track_info(self, track_id):
        """Get detailed information of a track"""
        return self.track_data.get(track_id, None)
    
    def get_roi(self, track_id):
        """Get ROI (crop image) of a track ID"""
        if track_id in self.track_data:
            return self.track_data[track_id].get("roi", None)
        return None
