"""
SuperPoint Feature Extraction Module
======================================
Extract keypoints and descriptors from crop images using SuperPoint.
Uses local LightGlue package for SuperPoint implementation.

Author: AI Assistant
Date: March 23, 2026
"""

import sys
import torch
import numpy as np
import cv2
from typing import Dict, Tuple, Optional, Union
from pathlib import Path

# Add LightGlue path for local import
LIGHTGLUE_PATH = Path(__file__).parent / "LightGlue"
if LIGHTGLUE_PATH.exists():
    sys.path.insert(0, str(LIGHTGLUE_PATH))
    print(f"[SuperPointExtractor] Added LightGlue path: {LIGHTGLUE_PATH}")
else:
    print(f"[SuperPointExtractor] Warning: LightGlue path not found at {LIGHTGLUE_PATH}")

try:
    from lightglue import SuperPoint as SuperPointModel
    LIGHTGLUE_AVAILABLE = True
except ImportError as e:
    LIGHTGLUE_AVAILABLE = False
    print(f"[SuperPointExtractor] Warning: LightGlue SuperPoint not available: {e}")
    print("[SuperPointExtractor] Using placeholder extraction")

# Import contour filtering from utils
try:
    from utils import filter_keypoints_by_contour, get_contour_mask
    CONTOUR_FILTERING_AVAILABLE = True
except ImportError:
    CONTOUR_FILTERING_AVAILABLE = False
    print("[SuperPointExtractor] Warning: Contour filtering not available (utils module not found)")


class SuperPointExtractor:
    """
    Extract SuperPoint features from images using local LightGlue implementation.
    
    Attributes:
        model: SuperPoint model from LightGlue
        device: torch device (cuda or cpu)
        confidence_threshold: minimum keypoint confidence score
        use_contour_filtering: Apply contour-based filtering to remove background points
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda", 
                 use_contour_filtering: bool = True):
        """
        Initialize SuperPoint extractor using local LightGlue.
        
        Args:
            model_path: Path to SuperPoint model file (ignored, uses LightGlue)
            device: Device to use ('cuda' or 'cpu')
            use_contour_filtering: Enable contour-based filtering to remove background points
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = 0.0
        self.model = None
        self.use_contour_filtering = use_contour_filtering and CONTOUR_FILTERING_AVAILABLE
        
        if LIGHTGLUE_AVAILABLE:
            try:
                # Load SuperPoint model from LightGlue
                self.model = SuperPointModel(max_num_keypoints=512).eval().to(self.device)
                print(f"[SuperPointExtractor] SuperPoint loaded from LightGlue on device: {self.device}")
            except Exception as e:
                print(f"[SuperPointExtractor] Error loading SuperPoint: {e}")
                self.model = None
        else:
            print(f"[SuperPointExtractor] LightGlue not available, using placeholder extraction")
        
        if self.use_contour_filtering:
            print("[SuperPointExtractor] Contour-based filtering ENABLED - background points will be removed")
    
    def extract_features(self, 
                        image: Union[np.ndarray, torch.Tensor], 
                        confidence_threshold: float = 0.0,
                        bbox: Optional[Tuple[int, int, int, int]] = None,
                        apply_contour_filter: bool = True) -> Dict:
        """
        Extract keypoints and descriptors from a single image using SuperPoint.
        
        Optionally filters keypoints to only keep those on object contours,
        removing background points using color/intensity edge detection.
        
        Args:
            image: Input image (H, W, 3) or (H, W, 1) or torch.Tensor (1, 1, H, W)
            confidence_threshold: Minimum confidence score for keypoints
            bbox: Optional tuple (x1, y1, x2, y2) for contour-based filtering reference
            apply_contour_filter: Whether to apply contour-based filtering (if enabled)
            
        Returns:
            dict: {
                'keypoints': np.array (N, 2) - (x, y) coordinates,
                'descriptors': np.array (N, 256) - SuperPoint descriptors,
                'scores': np.array (N,) - Confidence scores,
                'center': tuple (cx, cy) - Center of all keypoints,
                'image': torch.Tensor - Normalized tensor for LightGlue,
                'num_filtered': int - Number of points removed by contour filter (if applied)
            }
        """
        if self.model is None:
            return self._extract_features_placeholder(image, confidence_threshold)
        
        # Store original image for contour filtering
        original_image = None
        if apply_contour_filter and self.use_contour_filtering:
            if isinstance(image, np.ndarray):
                original_image = image.copy()
            elif isinstance(image, torch.Tensor):
                # Convert tensor back to numpy for contour filtering
                original_image = image[0].permute(1, 2, 0).numpy() if len(image.shape) == 4 else image
        
        # Convert to tensor if needed
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:  # Grayscale
                image = np.expand_dims(image, axis=2)
            # Normalize to [0, 1]
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
        else:
            image_tensor = image.to(self.device)
        
        try:
            with torch.no_grad():
                # Extract features using LightGlue SuperPoint
                outputs = self.model({"image": image_tensor})
                
                keypoints = outputs["keypoints"].cpu().numpy()  # (1, N, 2)
                descriptors = outputs["descriptors"].cpu().numpy()  # (1, N, 256)
                
                # Try to get scores, fallback to ones if not available
                if "scores" in outputs:
                    scores = outputs["scores"].cpu().numpy()  # (1, N)
                else:
                    # SuperPoint may not return scores, use ones as default
                    scores = np.ones(keypoints.shape[:-1])
                
                # Remove batch dimension
                if len(keypoints.shape) == 3:
                    keypoints = keypoints[0]  # (N, 2)
                if len(descriptors.shape) == 3:
                    descriptors = descriptors[0]  # (N, 256)
                if len(scores.shape) == 2:
                    scores = scores[0]  # (N,)
                elif len(scores.shape) == 1 and len(scores) != len(keypoints):
                    # If scores shape doesn't match keypoints, use ones
                    scores = np.ones(len(keypoints))
                
                # Filter by confidence threshold
                valid_mask = scores >= confidence_threshold
                keypoints = keypoints[valid_mask].astype(np.float32)
                descriptors = descriptors[valid_mask].astype(np.float32)
                scores = scores[valid_mask].astype(np.float32)
                
                num_filtered = 0
                
                # Apply contour-based filtering to remove background points
                if (apply_contour_filter and self.use_contour_filtering and 
                    original_image is not None and bbox is not None and len(keypoints) > 0):
                    
                    try:
                        num_before = len(keypoints)
                        keypoints, descriptors = filter_keypoints_by_contour(
                            keypoints=keypoints,
                            descriptors=descriptors,
                            region=original_image,
                            bbox=bbox,
                            keypoint_coords='region',
                            canny_threshold1=50,
                            canny_threshold2=150,
                            contour_tolerance=3
                        )
                        num_filtered = num_before - len(keypoints)
                        
                        if num_filtered > 0:
                            print(f"[SuperPointExtractor] Contour filter: {num_before} -> {len(keypoints)} points "
                                  f"(removed {num_filtered} background points)")
                    except Exception as e:
                        print(f"[SuperPointExtractor] Warning: Contour filtering failed: {e}")
                        num_filtered = 0
                
                # Calculate center of keypoints
                if len(keypoints) > 0:
                    center = np.mean(keypoints, axis=0)
                else:
                    center = (0, 0)
                
                return {
                    'keypoints': keypoints,        # (N, 2)
                    'descriptors': descriptors,    # (N, 256)
                    'scores': scores,              # (N,)
                    'center': tuple(center),       # (cx, cy)
                    'image': image_tensor.cpu(),   # Normalized tensor for LightGlue
                    'num_filtered': num_filtered   # Number of points removed by contour filter
                }
        except Exception as e:
            print(f"[SuperPointExtractor] Error during feature extraction: {e}")
            import traceback
            traceback.print_exc()
            return self._extract_features_placeholder(image, confidence_threshold)
    
    def _extract_features_placeholder(self, 
                                     image: Union[np.ndarray, torch.Tensor],
                                     confidence_threshold: float = 0.0) -> Dict:
        """Placeholder implementation when LightGlue is not available."""
        # Placeholder implementation
        keypoints = np.array([[100, 100], [200, 150], [350, 200]], dtype=np.float32)
        descriptors = np.random.randn(3, 256).astype(np.float32)
        scores = np.array([0.95, 0.87, 0.91], dtype=np.float32)
        
        # Filter by confidence threshold
        valid_mask = scores >= confidence_threshold
        keypoints = keypoints[valid_mask]
        descriptors = descriptors[valid_mask]
        scores = scores[valid_mask]
        
        # Calculate center of keypoints
        if len(keypoints) > 0:
            center = np.mean(keypoints, axis=0)
        else:
            center = (0, 0)
        
        return {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'scores': scores,
            'center': center,
            'image': torch.from_numpy(np.zeros((1, 3, 100, 100), dtype=np.float32))
        }
    
    def batch_extract(self, 
                     crops_dict: Dict[int, np.ndarray], 
                     camera_id: int = 0) -> Dict[int, Dict]:
        """
        Extract features from multiple crops at once.
        
        Args:
            crops_dict: Dictionary mapping ID -> crop image
            camera_id: Camera identifier (0 or 1)
            
        Returns:
            dict: {
                id: {
                    'keypoints': np.array (N, 2),
                    'descriptors': np.array (N, 256),
                    'scores': np.array (N,),
                    'center': tuple (cx, cy)
                }
            }
        """
        features = {}
        
        for track_id, crop_image in crops_dict.items():
            if crop_image is None:
                continue
            
            try:
                features[track_id] = self.extract_features(crop_image)
            except Exception as e:
                print(f"[SuperPointExtractor] Error extracting features for ID {track_id}: {e}")
                features[track_id] = {
                    'keypoints': np.array([], dtype=np.float32).reshape(0, 2),
                    'descriptors': np.array([], dtype=np.float32).reshape(0, 256),
                    'scores': np.array([], dtype=np.float32),
                    'center': (0, 0)
                }
        
        print(f"[SuperPointExtractor] Extracted features for {len(features)} crops from camera {camera_id}")
        return features
    
    def visualize_keypoints(self, 
                           image: np.ndarray, 
                           features: Dict) -> np.ndarray:
        """
        Visualize keypoints on the image with circles and confidence scores.
        
        Args:
            image: Input image
            features: Extracted features dict
            
        Returns:
            np.ndarray: Image with keypoints drawn
        """
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
            if image.shape[0] in [1, 3]:  # CHW format
                image = np.transpose(image, (1, 2, 0))
        
        if image.dtype == np.float32:
            image = (image * 255).astype(np.uint8)
        
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
        
        keypoints = features.get('keypoints', [])
        scores = features.get('scores', [])
        
        for idx, (x, y) in enumerate(keypoints):
            # Green circle for keypoint
            cv2.circle(vis_image, (int(x), int(y)), 4, (0, 255, 0), -1)
            cv2.circle(vis_image, (int(x), int(y)), 5, (0, 255, 255), 2)
            
            # Draw confidence score
            if idx < len(scores):
                score = scores[idx]
                cv2.putText(vis_image, f"{score:.2f}", (int(x)+8, int(y)-8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        return vis_image


# Example usage
if __name__ == "__main__":
    # Test basic functionality
    extractor = SuperPointExtractor()
    
    # Create a dummy image
    dummy_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    
    # Extract features
    features = extractor.extract_features(dummy_image)
    print(f"Extracted {len(features['keypoints'])} keypoints")
    print(f"Feature shape: {features['descriptors'].shape}")
