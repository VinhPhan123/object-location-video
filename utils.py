"""
Utility functions for YOLO Dual Camera Application
"""

import cv2
import numpy as np
from typing import Tuple, List


def get_color_palette(num_colors: int = 10) -> List[Tuple[int, int, int]]:
    """
    Generate a palette of distinct colors for tracking
    
    Args:
        num_colors: Number of desired colors
        
    Returns:
        List of BGR colors
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
        (128, 128, 0),    # Teal
        (128, 0, 128),    # Purple
        (0, 128, 128),    # Olive
    ]
    return colors[:num_colors] if num_colors <= len(colors) else colors * (num_colors // len(colors) + 1)


def get_color_for_id(track_id: int) -> Tuple[int, int, int]:
    """
    Get color for a tracking ID
    
    Args:
        track_id: Track ID
        
    Returns:
        BGR color tuple
    """
    colors = get_color_palette()
    return colors[track_id % len(colors)]


def get_contour_mask(region: np.ndarray, 
                     canny_threshold1: int = 50, 
                     canny_threshold2: int = 150, 
                     dilation_kernel_size: int = 3,
                     dilation_iterations: int = 2) -> np.ndarray:
    """
    Create a mask for object contour/edges based on color intensity differences.
    
    Method:
    1. Convert to grayscale
    2. Use Canny edge detection to find boundaries
    3. Use morphological operations to clean and expand boundaries
    4. Create mask to indicate contour points
    
    Args:
        region: Cropped image (bbox region)
        canny_threshold1: Low threshold for Canny edge detection
        canny_threshold2: High threshold for Canny edge detection
        dilation_kernel_size: Kernel size for morphological operations
        dilation_iterations: Number of dilation iterations to expand contour
        
    Returns:
        Binary mask: 255 if point on contour, 0 otherwise
    """
    if region is None or region.size == 0:
        return None
    
    # Convert to grayscale
    if len(region.shape) == 3:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    else:
        gray = region.copy()
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    
    # Canny edge detection
    edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)
    
    # Dilation to expand edges (to catch points near boundaries)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, 
        (dilation_kernel_size, dilation_kernel_size)
    )
    contour_mask = cv2.dilate(edges, kernel, iterations=dilation_iterations)
    
    return contour_mask


def filter_keypoints_by_contour(keypoints: np.ndarray, 
                                descriptors: np.ndarray,
                                region: np.ndarray,
                                bbox: Tuple[int, int, int, int],
                                keypoint_coords: str = 'region',
                                canny_threshold1: int = 50,
                                canny_threshold2: int = 150,
                                contour_tolerance: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter keypoints to keep only those on the contour/edge of the object.
    
    Background points are removed. Only points on object boundaries (with color
    intensity changes) are kept.
    
    Args:
        keypoints: Keypoint array (N, 2) with coordinates [x, y]
        descriptors: Corresponding descriptor array (N, D)
        region: Cropped image from bbox
        bbox: Tuple (x1, y1, x2, y2) bbox coordinates
        keypoint_coords: 'region' if keypoints in crop coords, 'frame' if frame coords
        canny_threshold1: Low threshold for Canny (default: 50)
        canny_threshold2: High threshold for Canny (default: 150)
        contour_tolerance: Pixel tolerance to consider point on contour (default: 3)
        
    Returns:
        Tuple (filtered_keypoints, filtered_descriptors)
        - filtered_keypoints: Only contour points
        - filtered_descriptors: Corresponding descriptors
    """
    if keypoints is None or len(keypoints) == 0:
        if descriptors is None:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        else:
            return np.array([], dtype=np.float32), descriptors[:0]
    
    if region is None:
        return keypoints, descriptors
    
    # Create contour mask
    contour_mask = get_contour_mask(
        region,
        canny_threshold1=canny_threshold1,
        canny_threshold2=canny_threshold2,
        dilation_kernel_size=3,
        dilation_iterations=2
    )
    
    if contour_mask is None:
        return keypoints, descriptors
    
    # Convert keypoint coordinates if needed
    if keypoint_coords == 'frame':
        x1, y1 = bbox[0], bbox[1]
        kpts_in_region = keypoints.copy()
        kpts_in_region[:, 0] -= x1
        kpts_in_region[:, 1] -= y1
    else:
        kpts_in_region = keypoints.copy()
    
    # Check each keypoint for contour location
    h, w = contour_mask.shape
    valid_mask = np.zeros(len(kpts_in_region), dtype=bool)
    
    for i, (kx, ky) in enumerate(kpts_in_region):
        x, y = int(round(kx)), int(round(ky))
        
        # Check if point within region bounds
        if 0 <= x < w and 0 <= y < h:
            # Check if point on contour (value > 0 in mask)
            if contour_mask[y, x] > 0:
                valid_mask[i] = True
    
    # If too few points kept (< 20% original), lower threshold for more edges
    if np.sum(valid_mask) < max(1, len(valid_mask) // 5):
        # Lower Canny threshold to get more edges
        contour_mask = get_contour_mask(
            region,
            canny_threshold1=max(20, canny_threshold1 // 2),
            canny_threshold2=canny_threshold2,
            dilation_kernel_size=5,
            dilation_iterations=3
        )
        
        valid_mask = np.zeros(len(kpts_in_region), dtype=bool)
        for i, (kx, ky) in enumerate(kpts_in_region):
            x, y = int(round(kx)), int(round(ky))
            if 0 <= x < w and 0 <= y < h:
                if contour_mask[y, x] > 0:
                    valid_mask[i] = True
    
    # Return filtered arrays
    filtered_keypoints = keypoints[valid_mask].astype(np.float32)
    
    if descriptors is not None:
        filtered_descriptors = descriptors[valid_mask].astype(np.float32)
    else:
        filtered_descriptors = np.array([], dtype=np.float32)
    
    return filtered_keypoints, filtered_descriptors


def draw_contour_on_frame(frame: np.ndarray,
                          bbox: Tuple[int, int, int, int],
                          color: Tuple[int, int, int] = (0, 255, 0),
                          line_thickness: int = 2,
                          canny_threshold1: int = 50,
                          canny_threshold2: int = 150) -> np.ndarray:
    """
    Draw object contour (edges) on frame.
    
    This function:
    1. Crops region from bounding box
    2. Detects edges/contours based on color intensity changes
    3. Draws contours on original frame
    
    Args:
        frame: Input frame (original image)
        bbox: Tuple (x1, y1, x2, y2) bbox coordinates
        color: BGR color to draw contours
        line_thickness: Line drawing thickness
        canny_threshold1: Low threshold for Canny edge detection
        canny_threshold2: High threshold for Canny edge detection
        
    Returns:
        Frame with contours drawn
    """
    if frame is None or bbox is None:
        return frame
    
    frame_display = frame.copy()
    x1, y1, x2, y2 = bbox
    
    # Crop region from bbox
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    
    if x1 >= x2 or y1 >= y2:
        return frame_display
    
    region = frame[y1:y2, x1:x2]
    
    if region.size == 0:
        return frame_display
    
    # Get contour mask
    contour_mask = get_contour_mask(
        region,
        canny_threshold1=canny_threshold1,
        canny_threshold2=canny_threshold2,
        dilation_kernel_size=3,
        dilation_iterations=1
    )
    
    if contour_mask is None or contour_mask.size == 0:
        return frame_display
    
    # Find contours from mask
    contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return frame_display
    
    # Draw contours on original frame
    # Convert contour coordinates from crop space to frame space
    for contour in contours:
        # Filter out small contours (noise)
        area = cv2.contourArea(contour)
        if area < 10:  # Skip contours < 10 px²
            continue
        
        # Add offset from bbox to convert from crop coords to frame coords
        contour_frame = contour.copy()
        contour_frame[:, 0, 0] += x1
        contour_frame[:, 0, 1] += y1
        
        # Draw contour
        cv2.drawContours(frame_display, [contour_frame], 0, color, line_thickness, cv2.LINE_AA)
    
    return frame_display


def draw_contours_for_all_objects(frame: np.ndarray,
                                  track_ids: List[int],
                                  track_info_getter,
                                  color_getter,
                                  line_thickness: int = 2) -> np.ndarray:
    """
    Draw contours for all tracked objects.
    
    Args:
        frame: Input frame
        track_ids: List of track IDs
        track_info_getter: Callback to get track info: track_info_getter(track_id) -> dict with 'bbox'
        color_getter: Callback to get color: color_getter(track_id) -> (B, G, R)
        line_thickness: Line drawing thickness
        
    Returns:
        Frame with all contours drawn
    """
    if not track_ids:
        return frame
    
    frame_display = frame.copy()
    
    for track_id in track_ids:
        try:
            track_info = track_info_getter(track_id)
            if track_info and 'bbox' in track_info:
                bbox = track_info['bbox']
                color = color_getter(track_id)
                frame_display = draw_contour_on_frame(
                    frame_display,
                    bbox,
                    color=color,
                    line_thickness=line_thickness
                )
        except Exception as e:
            print(f"[draw_contours_for_all_objects] Error drawing contour for track {track_id}: {e}")
            continue
    
    return frame_display
