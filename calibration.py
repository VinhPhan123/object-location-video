"""
Camera Calibration Module
==========================
Load and manage camera calibration data.

Author: AI Assistant
Date: March 23, 2026
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Tuple, Optional


class CalibrationManager:
    """
    Manage camera calibration data loading and saving.
    """
    
    def __init__(self, calib_path: str = "calibration_data/"):
        """
        Initialize calibration manager.
        
        Args:
            calib_path: Path to calibration data folder
        """
        self.calib_path = Path(calib_path)
        self.calib_path.mkdir(parents=True, exist_ok=True)
        
        self.calibration = {
            'K_cam0': None,
            'K_cam1': None,
            'D_cam0': None,
            'D_cam1': None,
            'R': None,
            't': None,
            'baseline': None,
        }
    
    def load_all_calibration(self) -> Dict:
        """
        Load all calibration matrices from disk.
        
        Returns:
            dict: Calibration data
                {
                    'K_cam0': (3, 3),
                    'K_cam1': (3, 3),
                    'D_cam0': (5,) or (8,),
                    'D_cam1': (5,) or (8,),
                    'R': (3, 3),
                    't': (3, 1),
                    'baseline': float
                }
        """
        try:
            self.calibration['K_cam0'] = np.load(self.calib_path / "camera0_intrinsics.npy")
            self.calibration['K_cam1'] = np.load(self.calib_path / "camera1_intrinsics.npy")
            self.calibration['D_cam0'] = np.load(self.calib_path / "camera0_distortion.npy")
            self.calibration['D_cam1'] = np.load(self.calib_path / "camera1_distortion.npy")
            
            pose_data = np.load(self.calib_path / "relative_pose.npy", allow_pickle=True)
            self.calibration['R'] = pose_data[0]
            self.calibration['t'] = pose_data[1]
            
            # Try to load baseline
            baseline_file = self.calib_path / "baseline.txt"
            if baseline_file.exists():
                self.calibration['baseline'] = float(baseline_file.read_text().strip())
            
            print("[CalibrationManager] Calibration data loaded successfully")
            return self.calibration
        
        except Exception as e:
            print(f"[CalibrationManager] Error loading calibration: {e}")
            return None
    
    def save_calibration(self, 
                        K_cam0: np.ndarray, D_cam0: np.ndarray,
                        K_cam1: np.ndarray, D_cam1: np.ndarray,
                        R: np.ndarray, t: np.ndarray,
                        baseline: Optional[float] = None):
        """
        Save calibration matrices to disk.
        
        Args:
            K_cam0: Camera 0 intrinsics (3, 3)
            D_cam0: Camera 0 distortion (5,) or (8,)
            K_cam1: Camera 1 intrinsics (3, 3)
            D_cam1: Camera 1 distortion (5,) or (8,)
            R: Rotation matrix (3, 3)
            t: Translation vector (3, 1)
            baseline: Baseline distance in meters (optional)
        """
        try:
            np.save(self.calib_path / "camera0_intrinsics.npy", K_cam0)
            np.save(self.calib_path / "camera1_intrinsics.npy", K_cam1)
            np.save(self.calib_path / "camera0_distortion.npy", D_cam0)
            np.save(self.calib_path / "camera1_distortion.npy", D_cam1)
            np.save(self.calib_path / "relative_pose.npy", np.array([R, t], dtype=object))
            
            if baseline is not None:
                (self.calib_path / "baseline.txt").write_text(str(baseline))
            
            # Update internal calibration
            self.calibration = {
                'K_cam0': K_cam0, 'K_cam1': K_cam1,
                'D_cam0': D_cam0, 'D_cam1': D_cam1,
                'R': R, 't': t, 'baseline': baseline
            }
            
            print("[CalibrationManager] Calibration data saved successfully")
        except Exception as e:
            print(f"[CalibrationManager] Error saving calibration: {e}")
    
    def create_projection_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create projection matrices P and P' from calibration data.
        
        Returns:
            Tuple: (P_cam0, P_cam1)
                P_cam0 = K_cam0 @ [I | 0]  (3, 4)
                P_cam1 = K_cam1 @ [R | t]  (3, 4)
        """
        if any(v is None for v in [self.calibration['K_cam0'], self.calibration['R'], self.calibration['t']]):
            print("[CalibrationManager] Calibration data not loaded")
            return None, None
        
        K_cam0 = self.calibration['K_cam0']
        K_cam1 = self.calibration['K_cam1']
        R = self.calibration['R']
        t = self.calibration['t']
        
        P_cam0 = K_cam0 @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P_cam1 = K_cam1 @ np.hstack([R, t])
        
        return P_cam0, P_cam1
    
    def rectify_stereo(self, img0: np.ndarray, img1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rectify stereo image pair.
        
        Args:
            img0: Image from camera 0
            img1: Image from camera 1
            
        Returns:
            Tuple: (img0_rectified, img1_rectified)
        """
        K_cam0 = self.calibration['K_cam0']
        K_cam1 = self.calibration['K_cam1']
        D_cam0 = self.calibration['D_cam0']
        D_cam1 = self.calibration['D_cam1']
        R = self.calibration['R']
        t = self.calibration['t']
        
        if any(v is None for v in [K_cam0, K_cam1, D_cam0, D_cam1, R, t]):
            print("[CalibrationManager] Calibration data not loaded")
            return img0, img1
        
        # Stereorectify
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            K_cam0, D_cam0, K_cam1, D_cam1,
            img0.shape[:2], R, t,
            alpha=0
        )
        
        # Compute rectification maps
        map1_0, map2_0 = cv2.initUndistortRectifyMap(
            K_cam0, D_cam0, R1, P1, img0.shape[:2], cv2.CV_32F
        )
        map1_1, map2_1 = cv2.initUndistortRectifyMap(
            K_cam1, D_cam1, R2, P2, img1.shape[:2], cv2.CV_32F
        )
        
        # Apply rectification
        img0_rect = cv2.remap(img0, map1_0, map2_0, cv2.INTER_LINEAR)
        img1_rect = cv2.remap(img1, map1_1, map2_1, cv2.INTER_LINEAR)
        
        return img0_rect, img1_rect
    
    def get_calibration(self) -> Dict:
        """Get current calibration data."""
        return self.calibration.copy()
    
    def print_calibration_info(self):
        """Print calibration information."""
        print("\n=== Camera Calibration Info ===")
        
        if self.calibration['K_cam0'] is not None:
            K = self.calibration['K_cam0']
            print(f"Camera 0 Intrinsics (K):")
            print(f"  fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
            print(f"  cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
        
        if self.calibration['K_cam1'] is not None:
            K = self.calibration['K_cam1']
            print(f"Camera 1 Intrinsics (K):")
            print(f"  fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
            print(f"  cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
        
        if self.calibration['t'] is not None:
            t = self.calibration['t'].flatten()
            baseline = np.linalg.norm(t)
            print(f"Baseline: {baseline:.3f} meters")
            print(f"Translation: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]")
        
        print()


def create_default_calibration(width: int = 640, 
                              height: int = 480,
                              baseline: float = 0.12) -> Dict:
    """
    Create default calibration matrices for testing.
    
    Args:
        width: Image width
        height: Image height
        baseline: Baseline distance in meters
        
    Returns:
        dict: Calibration data
    """
    # Estimate focal length (assuming 50-degree field of view)
    fov = 50 * np.pi / 180
    fx = fy = (width / 2) / np.tan(fov / 2)
    cx, cy = width / 2, height / 2
    
    K_cam0 = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    
    K_cam1 = K_cam0.copy()
    
    D_cam0 = np.array([0.1, 0.05, 0, 0, 0], dtype=np.float64)
    D_cam1 = np.array([0.1, 0.05, 0, 0, 0], dtype=np.float64)
    
    R = np.eye(3, dtype=np.float64)
    t = np.array([[-baseline], [0], [0]], dtype=np.float64)
    
    return {
        'K_cam0': K_cam0, 'K_cam1': K_cam1,
        'D_cam0': D_cam0, 'D_cam1': D_cam1,
        'R': R, 't': t, 'baseline': baseline
    }


# Example usage
if __name__ == "__main__":
    # Create and use calibration manager
    calib_mgr = CalibrationManager()
    
    # Load or create default calibration
    calib = calib_mgr.load_all_calibration()
    if calib is None:
        print("Creating default calibration for testing...")
        calib = create_default_calibration()
        calib_mgr.calibration = calib
    
    calib_mgr.print_calibration_info()
    
    # Create projection matrices
    P0, P1 = calib_mgr.create_projection_matrices()
    print(f"Projection matrix P0 shape: {P0.shape}")
    print(f"Projection matrix P1 shape: {P1.shape}")
