"""
Stereo Triangulation Module
============================
Compute 3D coordinates using stereo triangulation.

Author: AI Assistant
Date: March 23, 2026
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List
from pathlib import Path


class StereoTriangulator:
    """
    Perform stereo triangulation to compute 3D coordinates.
    
    Uses camera intrinsics and extrinsics to compute world coordinates
    from matched image points in two cameras.
    """
    
    def __init__(self, calibration_data_path: str = "calibration_data/"):
        """
        Initialize stereo triangulator with calibration data.
        
        Args:
            calibration_data_path: Path to calibration data folder
        """
        self.calib_path = Path(calibration_data_path)
        
        # Load calibration matrices
        self.K_cam0 = None
        self.K_cam1 = None
        self.D_cam0 = None
        self.D_cam1 = None
        self.R_rel = None  # Rotation from cam0 to cam1
        self.t_rel = None  # Translation from cam0 to cam1
        self.P_cam0 = None  # Projection matrix cam0
        self.P_cam1 = None  # Projection matrix cam1
        
        # TODO: Load calibration data
        # self._load_calibration_data()
        
        self._init_placeholder_calibration()
        print(f"[StereoTriangulator] Initialized from {calibration_data_path}")
    
    def _init_placeholder_calibration(self):
        """Initialize with placeholder calibration for testing."""
        # Typical calibration values for USB cameras
        fx_cam0, fy_cam0 = 800, 800
        cx_cam0, cy_cam0 = 320, 240
        self.K_cam0 = np.array([
            [fx_cam0, 0, cx_cam0],
            [0, fy_cam0, cy_cam0],
            [0, 0, 1]
        ], dtype=np.float64)
        
        fx_cam1, fy_cam1 = 800, 800
        cx_cam1, cy_cam1 = 320, 240
        self.K_cam1 = np.array([
            [fx_cam1, 0, cx_cam1],
            [0, fy_cam1, cy_cam1],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Distortion coefficients (typically small for USB cameras)
        self.D_cam0 = np.array([0.1, 0.01, 0, 0, 0], dtype=np.float64)
        self.D_cam1 = np.array([0.1, 0.01, 0, 0, 0], dtype=np.float64)
        
        # Relative pose (cam1 w.r.t. cam0)
        # Assume ~120mm baseline
        self.R_rel = np.eye(3, dtype=np.float64)  # No rotation
        self.t_rel = np.array([[-0.12], [0], [0]], dtype=np.float64)  # 120mm baseline
        
        # Create projection matrices
        self.P_cam0 = self.K_cam0 @ np.hstack([np.eye(3), np.zeros((3, 1))])
        self.P_cam1 = self.K_cam1 @ np.hstack([self.R_rel, self.t_rel])
    
    def _load_calibration_data(self):
        """Load calibration data from files."""
        try:
            self.K_cam0 = np.load(self.calib_path / "camera0_intrinsics.npy")
            self.K_cam1 = np.load(self.calib_path / "camera1_intrinsics.npy")
            self.D_cam0 = np.load(self.calib_path / "camera0_distortion.npy")
            self.D_cam1 = np.load(self.calib_path / "camera1_distortion.npy")
            
            pose_data = np.load(self.calib_path / "relative_pose.npy", allow_pickle=True)
            self.R_rel = pose_data[0]
            self.t_rel = pose_data[1]
            
            # Create projection matrices
            self.P_cam0 = self.K_cam0 @ np.hstack([np.eye(3), np.zeros((3, 1))])
            self.P_cam1 = self.K_cam1 @ np.hstack([self.R_rel, self.t_rel])
            
            print("[StereoTriangulator] Calibration data loaded successfully")
        except Exception as e:
            print(f"[StereoTriangulator] Error loading calibration: {e}")
            self._init_placeholder_calibration()
    
    def triangulate(self, 
                   pt_cam0: Tuple[float, float], 
                   pt_cam1: Tuple[float, float]) -> Dict:
        """
        Triangulate a single point pair from two cameras.
        
        Args:
            pt_cam0: (x, y) in camera 0 image coordinates
            pt_cam1: (x, y) in camera 1 image coordinates
            
        Returns:
            dict: {
                'point_3d': [X, Y, Z],           # World coordinates (meters)
                'reprojection_error': float,     # Pixel error
                'confidence': float,             # 0-1 confidence score
                'success': bool                  # Whether triangulation succeeded
            }
        """
        pts_4d = cv2.triangulatePoints(
            self.P_cam0, 
            self.P_cam1,
            np.array([[pt_cam0[0]], [pt_cam0[1]]], dtype=np.float64),
            np.array([[pt_cam1[0]], [pt_cam1[1]]], dtype=np.float64)
        )
        
        # Convert from homogeneous to 3D coordinates
        point_3d = pts_4d[:3] / pts_4d[3]
        point_3d = point_3d.flatten()
        
        # Compute reprojection error
        reproj_cam0 = self.P_cam0 @ np.append(point_3d, 1)
        reproj_cam0 = reproj_cam0[:2] / reproj_cam0[2]
        error0 = np.linalg.norm(reproj_cam0 - np.array(pt_cam0))
        
        reproj_cam1 = self.P_cam1 @ np.append(point_3d, 1)
        reproj_cam1 = reproj_cam1[:2] / reproj_cam1[2]
        error1 = np.linalg.norm(reproj_cam1 - np.array(pt_cam1))
        
        reprojection_error = (error0 + error1) / 2
        
        # Confidence based on reprojection error (threshold ~2 pixels)
        confidence = max(0, 1 - reprojection_error / 10.0)
        
        # Check if point is in front of both cameras
        success = point_3d[2] > 0
        
        return {
            'point_3d': point_3d,
            'reprojection_error': float(reprojection_error),
            'confidence': float(confidence),
            'success': bool(success)
        }
    
    def triangulate_matched_points(self, 
                                  pts_cam0: np.ndarray, 
                                  pts_cam1: np.ndarray) -> Dict:
        """
        Triangulate multiple matched point pairs.
        
        Args:
            pts_cam0: np.array (N, 2) - points in camera 0
            pts_cam1: np.array (N, 2) - points in camera 1
            
        Returns:
            dict: {
                'points_3d': np.array (N, 3) - 3D world coordinates,
                'reprojection_errors': np.array (N,),
                'confidence': float - mean of valid points,
                'valid_points': np.array (bool) - validity mask,
                'avg_depth': float - average Z coordinate
            }
        """
        if len(pts_cam0) == 0:
            return {
                'points_3d': np.array([], dtype=np.float32).reshape(0, 3),
                'reprojection_errors': np.array([], dtype=np.float32),
                'confidence': 0.0,
                'valid_points': np.array([], dtype=bool),
                'avg_depth': 0.0
            }
        
        points_3d = []
        errors = []
        valid_mask = []
        
        for pt0, pt1 in zip(pts_cam0, pts_cam1):
            result = self.triangulate(pt0, pt1)
            points_3d.append(result['point_3d'])
            errors.append(result['reprojection_error'])
            valid_mask.append(result['success'])
        
        points_3d = np.array(points_3d, dtype=np.float32)
        errors = np.array(errors, dtype=np.float32)
        valid_mask = np.array(valid_mask, dtype=bool)
        
        # Calculate statistics
        if np.any(valid_mask):
            avg_confidence = np.mean([
                1 - err / 10.0 for err in errors[valid_mask]
            ])
            avg_depth = np.mean(points_3d[valid_mask, 2])
        else:
            avg_confidence = 0.0
            avg_depth = 0.0
        
        return {
            'points_3d': points_3d,
            'reprojection_errors': errors,
            'confidence': float(avg_confidence),
            'valid_points': valid_mask,
            'avg_depth': float(avg_depth),
            'num_valid': int(np.sum(valid_mask))
        }
    
    def compute_point_from_features(self, 
                                   matched_features_cam0: Dict, 
                                   matched_features_cam1: Dict) -> Dict:
        """
        Compute 3D coordinates from matched feature keypoints.
        
        Uses the centers and matched keypoints from both cameras.
        
        Args:
            matched_features_cam0: dict with 'keypoints', 'descriptors', 'center'
            matched_features_cam1: similar structure
            
        Returns:
            dict: Similar to triangulate_matched_points
        """
        if 'keypoints' not in matched_features_cam0 or 'keypoints' not in matched_features_cam1:
            return {
                'points_3d': np.array([], dtype=np.float32).reshape(0, 3),
                'reprojection_errors': np.array([], dtype=np.float32),
                'confidence': 0.0,
                'valid_points': np.array([], dtype=bool),
                'avg_depth': 0.0
            }
        
        # Use matched feature keypoints
        pts_cam0 = matched_features_cam0['keypoints']  # (N, 2)
        pts_cam1 = matched_features_cam1['keypoints']  # (N, 2)
        
        return self.triangulate_matched_points(pts_cam0, pts_cam1)
    
    def set_calibration(self, K0: np.ndarray, D0: np.ndarray, 
                       K1: np.ndarray, D1: np.ndarray, 
                       R: np.ndarray, t: np.ndarray):
        """
        Set calibration matrices manually.
        
        Args:
            K0: Camera 0 intrinsics (3, 3)
            D0: Camera 0 distortion (5,) or (8,)
            K1: Camera 1 intrinsics (3, 3)
            D1: Camera 1 distortion (5,) or (8,)
            R: Rotation matrix (3, 3)
            t: Translation vector (3, 1) or (3,)
        """
        self.K_cam0 = K0.astype(np.float64)
        self.K_cam1 = K1.astype(np.float64)
        self.D_cam0 = D0.astype(np.float64)
        self.D_cam1 = D1.astype(np.float64)
        self.R_rel = R.astype(np.float64)
        if t.shape == (3,):
            self.t_rel = t.reshape(3, 1).astype(np.float64)
        else:
            self.t_rel = t.astype(np.float64)
        
        # Recreate projection matrices
        self.P_cam0 = self.K_cam0 @ np.hstack([np.eye(3), np.zeros((3, 1))])
        self.P_cam1 = self.K_cam1 @ np.hstack([self.R_rel, self.t_rel])


# Example usage
if __name__ == "__main__":
    triangulator = StereoTriangulator()
    
    # Test single point triangulation
    pt_cam0 = (320, 240)
    pt_cam1 = (300, 240)
    result = triangulator.triangulate(pt_cam0, pt_cam1)
    print(f"3D point: {result['point_3d']}")
    print(f"Reprojection error: {result['reprojection_error']:.2f} pixels")
