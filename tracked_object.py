"""
TrackedObject Class - Object-Oriented Representation of Tracked Entities
==========================================================================
Encapsulates all data associated with a tracked object across cameras:
- Unique identifier and tracking metadata
- Bounding box and spatial information
- Matched feature points within the bounding box
- Reference to matched object on other camera stream
- Feature descriptors and triangulation data

Author: AI Assistant
Date: March 24, 2026
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Point2D:
    """2D point representation with optional descriptor."""
    x: float
    y: float
    descriptor: Optional[np.ndarray] = None
    confidence: float = 0.0
    
    def to_tuple(self) -> Tuple[float, float]:
        """Convert to (x, y) tuple."""
        return (self.x, self.y)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array [x, y]."""
        return np.array([self.x, self.y])


@dataclass
class Point3D:
    """3D point representation with validity information."""
    x: float
    y: float
    z: float
    is_valid: bool = True
    reprojection_error: float = 0.0
    confidence: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array [x, y, z]."""
        return np.array([self.x, self.y, self.z])
    
    def distance_to(self, other: 'Point3D') -> float:
        """Calculate Euclidean distance to another 3D point."""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)


class TrackedObject:
    """
    Represents a tracked object detected in video stream.
    
    Attributes:
        id: Unique track ID
        camera_id: Source camera (1 or 2)
        bbox: Bounding box (x1, y1, x2, y2) where (x1, y1) is top-left, (x2, y2) is bottom-right
        class_name: Object class name from YOLO
        confidence: YOLO detection confidence
        matched_points_cam: List of 2D matched feature points in this camera
        descriptors: Feature descriptors for matched points
        3d_points: 3D triangulated points corresponding to matched points
        matched_object_other_camera: Reference to corresponding object on other camera
        center: Centroid of bounding box (x, y)
        area: Bounding box area in pixels
        timestamp: When this object was created/updated
        metadata: Additional optional metadata
    """
    
    def __init__(
        self,
        track_id: int,
        camera_id: int,
        bbox: Tuple[float, float, float, float],
        class_name: str,
        confidence: float,
        timestamp: Optional[datetime] = None,
        color: Optional[Tuple[int, int, int]] = None
    ):
        """
        Initialize a tracked object.
        
        Args:
            track_id: Unique track identifier
            camera_id: Source camera (1 or 2)
            bbox: Bounding box (x1, y1, x2, y2)
            class_name: Object class name
            confidence: Detection confidence score
            timestamp: Creation timestamp (defaults to current time)
            color: BGR color tuple (B, G, R) for display; if None, auto-generated from ID
        """
        self.id: int = track_id
        self.camera_id: int = camera_id
        self.bbox: Tuple[float, float, float, float] = bbox
        self.class_name: str = class_name
        self.confidence: float = confidence
        self.timestamp: datetime = timestamp or datetime.now()
        self.color: Tuple[int, int, int] = color or self._generate_color_from_id(track_id)
        
        # Feature points and descriptors
        self.matched_points_cam: List[Point2D] = []
        self.descriptors: List[np.ndarray] = []
        
        # 3D triangulation results
        self.triangulated_3d_points: List[Point3D] = []
        
        # Cross-camera matching
        self.matched_object_other_camera: Optional['TrackedObject'] = None
        
        # Additional metadata
        self.metadata: Dict[str, Any] = {}
        self._update_derived_properties()
    
    @staticmethod
    def _generate_color_from_id(track_id: int) -> Tuple[int, int, int]:
        """Generate BGR color from track ID for consistent coloring."""
        colors = [
            (255, 0, 0),      # Blue
            (0, 255, 0),      # Green
            (0, 0, 255),      # Red
            (255, 255, 0),    # Cyan
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Yellow
            (128, 0, 0),      # Dark Blue
            (0, 128, 0),      # Dark Green
            (0, 0, 128),      # Dark Red
            (128, 128, 0),    # Dark Cyan
            (128, 0, 128),    # Dark Magenta
            (0, 128, 128),    # Dark Yellow
            (192, 192, 192),  # Light Gray
            (128, 128, 128),  # Gray
            (255, 128, 0),    # Orange
            (0, 255, 128),    # Spring Green
        ]
        return colors[track_id % len(colors)]
    
    def _update_derived_properties(self) -> None:
        """Update derived properties (center, area, etc.)."""
        x1, y1, x2, y2 = self.bbox
        self.center: Tuple[float, float] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        self.width: float = x2 - x1
        self.height: float = y2 - y1
        self.area: float = self.width * self.height
    
    def add_matched_point(
        self,
        x: float,
        y: float,
        descriptor: Optional[np.ndarray] = None,
        confidence: float = 0.0
    ) -> None:
        """
        Add a matched feature point to this object.
        
        Args:
            x: X coordinate
            y: Y coordinate
            descriptor: 256-dimensional feature descriptor (optional)
            confidence: Point confidence score
        """
        point = Point2D(x=x, y=y, descriptor=descriptor, confidence=confidence)
        self.matched_points_cam.append(point)
        if descriptor is not None:
            self.descriptors.append(descriptor)
    
    def add_matched_points_batch(self, points: np.ndarray, descriptors: Optional[np.ndarray] = None) -> None:
        """
        Add multiple matched feature points at once.
        
        Args:
            points: Array of shape (N, 2) containing (x, y) coordinates
            descriptors: Array of shape (N, 256) containing feature descriptors (optional)
        """
        for i, pt in enumerate(points):
            desc = descriptors[i] if descriptors is not None else None
            self.add_matched_point(pt[0], pt[1], descriptor=desc)
    
    def add_3d_point(
        self,
        x: float,
        y: float,
        z: float,
        is_valid: bool = True,
        reprojection_error: float = 0.0,
        confidence: float = 0.0
    ) -> None:
        """
        Add a 3D triangulated point.
        
        Args:
            x, y, z: 3D coordinates
            is_valid: Whether the point passes validity checks
            reprojection_error: Reprojection error for this point
            confidence: Point confidence score
        """
        point_3d = Point3D(x=x, y=y, z=z, is_valid=is_valid, 
                          reprojection_error=reprojection_error, confidence=confidence)
        self.triangulated_3d_points.append(point_3d)
    
    def add_3d_points_batch(self, points_3d: np.ndarray, validity_mask: Optional[np.ndarray] = None,
                           errors: Optional[np.ndarray] = None) -> None:
        """
        Add multiple 3D points at once.
        
        Args:
            points_3d: Array of shape (N, 3) containing (x, y, z) coordinates
            validity_mask: Array of shape (N,) indicating valid points
            errors: Array of shape (N,) containing reprojection errors
        """
        for i, pt in enumerate(points_3d):
            is_valid = validity_mask[i] if validity_mask is not None else True
            error = errors[i] if errors is not None else 0.0
            self.add_3d_point(pt[0], pt[1], pt[2], is_valid=is_valid, reprojection_error=error)
    
    def set_matched_object(self, obj: Optional['TrackedObject']) -> None:
        """
        Set reference to matched object on other camera stream.
        Both matched objects will have the same color.
        
        Args:
            obj: TrackedObject from the other camera, or None to clear
        """
        self.matched_object_other_camera = obj
        if obj is not None:
            # Make both objects have the same color (use this object's color)
            obj.color = self.color
            # Establish bidirectional reference
            obj.matched_object_other_camera = self
    
    def is_matched(self) -> bool:
        """
        Check if this object has been matched with an object on the other camera.
        
        Returns:
            True if matched, False otherwise
        """
        return self.matched_object_other_camera is not None
    
    def set_unmatched_color(self, gray_color: Tuple[int, int, int] = (128, 128, 128)) -> None:
        """
        Set color to gray (unmatched state).
        Use this to mark objects that haven't been matched with objects on the other camera.
        
        Args:
            gray_color: BGR color tuple for unmatched objects (default: gray)
        """
        self.color = gray_color
    
    def get_matched_points_array(self) -> np.ndarray:
        """
        Get all matched points as numpy array.
        
        Returns:
            Array of shape (N, 2) containing (x, y) coordinates
        """
        if not self.matched_points_cam:
            return np.empty((0, 2), dtype=np.float32)
        return np.array([pt.to_array() for pt in self.matched_points_cam], dtype=np.float32)
    
    def get_descriptors_array(self) -> np.ndarray:
        """
        Get all descriptors as numpy array.
        
        Returns:
            Array of shape (N, 256) containing feature descriptors
        """
        if not self.descriptors:
            return np.empty((0, 256), dtype=np.float32)
        return np.array(self.descriptors, dtype=np.float32)
    
    def get_3d_points_array(self) -> np.ndarray:
        """
        Get all valid 3D points as numpy array.
        
        Returns:
            Array of shape (N, 3) containing valid (x, y, z) coordinates
        """
        valid_points = [pt.to_array() for pt in self.triangulated_3d_points if pt.is_valid]
        if not valid_points:
            return np.empty((0, 3), dtype=np.float32)
        return np.array(valid_points, dtype=np.float32)
    
    def get_bbox_tuple(self) -> Tuple[int, int, int, int]:
        """Get bounding box as integer tuple (x1, y1, x2, y2)."""
        x1, y1, x2, y2 = self.bbox
        return (int(x1), int(y1), int(x2), int(y2))
    
    def bbox_contains_point(self, x: float, y: float) -> bool:
        """
        Check if a point is inside this object's bounding box.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            True if point is inside bbox, False otherwise
        """
        x1, y1, x2, y2 = self.bbox
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistical information about this tracked object."""
        valid_3d_points = [pt for pt in self.triangulated_3d_points if pt.is_valid]
        
        stats = {
            'id': self.id,
            'camera_id': self.camera_id,
            'class_name': self.class_name,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'center': self.center,
            'area': self.area,
            'matched_points_count': len(self.matched_points_cam),
            'descriptors_count': len(self.descriptors),
            'triangulated_3d_points_total': len(self.triangulated_3d_points),
            'triangulated_3d_points_valid': len(valid_3d_points),
            'has_matched_object': self.matched_object_other_camera is not None,
            'matched_object_id': self.matched_object_other_camera.id if self.matched_object_other_camera else None,
            'timestamp': self.timestamp.isoformat()
        }
        
        if valid_3d_points:
            reprojection_errors = [pt.reprojection_error for pt in valid_3d_points]
            stats['mean_reprojection_error'] = float(np.mean(reprojection_errors))
            stats['max_reprojection_error'] = float(np.max(reprojection_errors))
            
            # Calculate mean depth (Z coordinate)
            depths = [pt.z for pt in valid_3d_points]
            stats['mean_depth'] = float(np.mean(depths))
            stats['depth_range'] = (float(np.min(depths)), float(np.max(depths)))
        
        return stats
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert object to dictionary representation."""
        return {
            'id': self.id,
            'camera_id': self.camera_id,
            'class_name': self.class_name,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'center': self.center,
            'width': self.width,
            'height': self.height,
            'area': self.area,
            'color': self.color,
            'matched_points_count': len(self.matched_points_cam),
            'descriptors_count': len(self.descriptors),
            'triangulated_3d_points_count': len(self.triangulated_3d_points),
            'valid_3d_points_count': sum(1 for pt in self.triangulated_3d_points if pt.is_valid),
            'matched_object_camera': self.matched_object_other_camera.camera_id if self.matched_object_other_camera else None,
            'matched_object_id': self.matched_object_other_camera.id if self.matched_object_other_camera else None,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
    
    def __repr__(self) -> str:
        """String representation of tracked object."""
        matched_info = f" -> Matched with Cam{self.matched_object_other_camera.camera_id} ID{self.matched_object_other_camera.id}" if self.matched_object_other_camera else " (No match)"
        return (f"TrackedObject(id={self.id}, camera={self.camera_id}, class='{self.class_name}', "
                f"conf={self.confidence:.2f}, points={len(self.matched_points_cam)}, "
                f"3d_pts={len(self.triangulated_3d_points)}{matched_info})")
    
    def __hash__(self) -> int:
        """Hash based on id and camera_id."""
        return hash((self.id, self.camera_id))
    
    def __eq__(self, other: Any) -> bool:
        """Equality comparison."""
        if not isinstance(other, TrackedObject):
            return False
        return self.id == other.id and self.camera_id == other.camera_id


class ObjectTracker:
    """
    Container for managing tracked objects across both cameras.
    
    Attributes:
        objects_cam1: Dict of camera 1 tracked objects {track_id: TrackedObject}
        objects_cam2: Dict of camera 2 tracked objects {track_id: TrackedObject}
        matched_pairs: List of matched object pairs across cameras
    """
    
    def __init__(self):
        """Initialize object tracker."""
        self.objects_cam1: Dict[int, TrackedObject] = {}
        self.objects_cam2: Dict[int, TrackedObject] = {}
        self.matched_pairs: List[Tuple[TrackedObject, TrackedObject]] = []
        self._current_color_index: int = 0  # Track color index for sequential assignment
    
    def add_object(self, obj: TrackedObject) -> None:
        """Add a tracked object."""
        if obj.camera_id == 1:
            self.objects_cam1[obj.id] = obj
        elif obj.camera_id == 2:
            self.objects_cam2[obj.id] = obj
    
    def get_object(self, track_id: int, camera_id: int) -> Optional[TrackedObject]:
        """Get tracked object by ID and camera."""
        if camera_id == 1:
            return self.objects_cam1.get(track_id)
        elif camera_id == 2:
            return self.objects_cam2.get(track_id)
        return None
    
    def remove_object(self, track_id: int, camera_id: int) -> None:
        """Remove a tracked object (dead track)."""
        if camera_id == 1:
            if track_id in self.objects_cam1:
                obj = self.objects_cam1.pop(track_id)
                # Remove from matched pairs if exists
                self.matched_pairs = [(o1, o2) for o1, o2 in self.matched_pairs if o1 != obj]
        elif camera_id == 2:
            if track_id in self.objects_cam2:
                obj = self.objects_cam2.pop(track_id)
                # Remove from matched pairs if exists
                self.matched_pairs = [(o1, o2) for o1, o2 in self.matched_pairs if o2 != obj]
    
    def match_objects(self, obj1: TrackedObject, obj2: TrackedObject) -> None:
        """Match two objects across cameras with sequential color assignment."""
        if obj1.camera_id == obj2.camera_id:
            raise ValueError("Cannot match objects from the same camera")
        
        # Get the next sequential color for matched pair
        matched_color = self._get_next_matched_color()
        
        # Assign color to both matched objects
        obj1.color = matched_color
        obj2.color = matched_color
        
        # Set bidirectional reference
        obj1.matched_object_other_camera = obj2
        obj2.matched_object_other_camera = obj1
        
        if (obj1, obj2) not in self.matched_pairs and (obj2, obj1) not in self.matched_pairs:
            self.matched_pairs.append((obj1, obj2))
    
    def get_all_objects(self) -> List[TrackedObject]:
        """Get all tracked objects from both cameras."""
        return list(self.objects_cam1.values()) + list(self.objects_cam2.values())
    
    def _get_next_matched_color(self) -> Tuple[int, int, int]:
        """
        Get the next sequential color from the palette for matched objects.
        Colors cycle through the palette in order.
        
        Returns:
            BGR color tuple
        """
        colors = [
            (255, 0, 0),      # Blue
            (0, 255, 0),      # Green
            (0, 0, 255),      # Red
            (255, 255, 0),    # Cyan
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Yellow
            (128, 0, 0),      # Dark Blue
            (0, 128, 0),      # Dark Green
            (0, 0, 128),      # Dark Red
            (128, 128, 0),    # Dark Cyan
            (128, 0, 128),    # Dark Magenta
            (0, 128, 128),    # Dark Yellow
        ]
        color = colors[self._current_color_index % len(colors)]
        self._current_color_index += 1
        return color
    
    def get_matched_pairs(self) -> List[Tuple[TrackedObject, TrackedObject]]:
        """Get all matched object pairs."""
        return self.matched_pairs.copy()
    
    def clear(self) -> None:
        """Clear all tracked objects."""
        self.objects_cam1.clear()
        self.objects_cam2.clear()
        self.matched_pairs.clear()
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"ObjectTracker(cam1={len(self.objects_cam1)}, cam2={len(self.objects_cam2)}, "
                f"matched_pairs={len(self.matched_pairs)})")
