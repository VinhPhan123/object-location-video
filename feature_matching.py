"""
LightGlue Feature Matching Module
==================================
Match features between crop images using local LightGlue package.

Author: AI Assistant
Date: March 23, 2026
"""

import sys
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass

# Add LightGlue path for local import
LIGHTGLUE_PATH = Path(__file__).parent / "LightGlue"
if LIGHTGLUE_PATH.exists():
    sys.path.insert(0, str(LIGHTGLUE_PATH))
    print(f"[LightGlueMatcher] Added LightGlue path: {LIGHTGLUE_PATH}")
else:
    print(f"[LightGlueMatcher] Warning: LightGlue path not found at {LIGHTGLUE_PATH}")

try:
    from lightglue import LightGlue as LightGlueModel
    from lightglue import match_pair
    LIGHTGLUE_AVAILABLE = True
except ImportError as e:
    LIGHTGLUE_AVAILABLE = False
    print(f"[LightGlueMatcher] Warning: LightGlue not available: {e}")
    print("[LightGlueMatcher] Using placeholder matching")


@dataclass
class MatchResult:
    """Store feature matching result."""
    id_cam0: int
    id_cam1: int
    confidence: float
    num_matches: int
    matched_indices: List[Tuple[int, int]]
    matched_pts_cam0: np.ndarray
    matched_pts_cam1: np.ndarray
    center_cam0: Tuple[float, float]
    center_cam1: Tuple[float, float]


class LightGlueMatcher:
    """
    Match features between two crop images using LightGlue.
    Uses local LightGlue package implementation.
    
    Attributes:
        model: LightGlue matcher model
        device: torch device (cuda or cpu)
        confidence_threshold: minimum matching confidence
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        """
        Initialize LightGlue matcher using local package.
        
        Args:
            model_path: Path to LightGlue model file (ignored, uses LightGlue package)
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = 0.7
        self.model = None
        
        if LIGHTGLUE_AVAILABLE:
            try:
                # Load LightGlue model configured for SuperPoint
                self.model = LightGlueModel(
                    features="superpoint",
                    depth_confidence=-1,  # -1 means all confidence levels
                    width_confidence=-1
                ).eval().to(self.device)
                print(f"[LightGlueMatcher] LightGlue loaded from local package on device: {self.device}")
            except Exception as e:
                print(f"[LightGlueMatcher] Error loading LightGlue: {e}")
                self.model = None
        else:
            print(f"[LightGlueMatcher] LightGlue not available, using placeholder matching")
    
    def match_features(self, 
                      features_cam0: Dict, 
                      features_cam1: Dict,
                      confidence_threshold: float = 0.7) -> Optional[MatchResult]:
        """
        Match features between two crops from different cameras.
        
        Args:
            features_cam0: Features dict from camera 0 {'keypoints', 'descriptors', 'scores', 'center'}
            features_cam1: Features dict from camera 1 (same structure)
            confidence_threshold: Minimum confidence for a valid match
            
        Returns:
            MatchResult: Matching results, or None if no good matches
        """
        if self.model is None:
            return self._match_features_placeholder(features_cam0, features_cam1, confidence_threshold)
        
        kpts0 = features_cam0.get('keypoints')  # (N, 2)
        kpts1 = features_cam1.get('keypoints')  # (M, 2)
        desc0 = features_cam0.get('descriptors')  # (N, 256)
        desc1 = features_cam1.get('descriptors')  # (M, 256)
        img0 = features_cam0.get('image')  # Tensor or None
        img1 = features_cam1.get('image')  # Tensor or None
        
        if kpts0 is None or kpts1 is None or len(kpts0) == 0 or len(kpts1) == 0:
            return None
        
        try:
            # Ensure keypoints and descriptors are numpy arrays with correct shape
            if isinstance(kpts0, torch.Tensor):
                kpts0 = kpts0.cpu().numpy()
            if isinstance(kpts1, torch.Tensor):
                kpts1 = kpts1.cpu().numpy()
            if isinstance(desc0, torch.Tensor):
                desc0 = desc0.cpu().numpy()
            if isinstance(desc1, torch.Tensor):
                desc1 = desc1.cpu().numpy()
            
            # Remove batch dimension if present (SuperPoint may add it)
            if len(kpts0.shape) == 3 and kpts0.shape[0] == 1:
                kpts0 = kpts0[0]  # (1, N, 2) -> (N, 2)
            if len(kpts1.shape) == 3 and kpts1.shape[0] == 1:
                kpts1 = kpts1[0]
            if len(desc0.shape) == 3 and desc0.shape[0] == 1:
                desc0 = desc0[0]  # (1, N, 256) -> (N, 256)
            if len(desc1.shape) == 3 and desc1.shape[0] == 1:
                desc1 = desc1[0]
            
            # Prepare data dict for LightGlue with CORRECT NESTED STRUCTURE
            # LightGlue expects: {image0: {keypoints, descriptors}, image1: {keypoints, descriptors}}
            data = {
                "image0": {
                    "keypoints": torch.from_numpy(kpts0).float().unsqueeze(0).to(self.device),  # (1, N, 2)
                    "descriptors": torch.from_numpy(desc0).float().unsqueeze(0).to(self.device),  # (1, N, 256)
                },
                "image1": {
                    "keypoints": torch.from_numpy(kpts1).float().unsqueeze(0).to(self.device),  # (1, M, 2)
                    "descriptors": torch.from_numpy(desc1).float().unsqueeze(0).to(self.device),  # (1, M, 256)
                }
            }
            
            # Add images if available
            if img0 is not None:
                if isinstance(img0, torch.Tensor):
                    img0 = img0.to(self.device)
                    # Ensure correct shape (C, H, W) -> (1, C, H, W)
                    if len(img0.shape) == 3:
                        img0 = img0.unsqueeze(0)
                data["image0"]["image"] = img0
            
            if img1 is not None:
                if isinstance(img1, torch.Tensor):
                    img1 = img1.to(self.device)
                    if len(img1.shape) == 3:
                        img1 = img1.unsqueeze(0)
                data["image1"]["image"] = img1
            
            with torch.no_grad():
                # Run LightGlue matching
                matches = self.model(data)
                matches0 = matches.get("matches0")  # (B, N) -> index in kpts1 or -1
                
                if matches0 is None:
                    return self._match_features_placeholder(features_cam0, features_cam1, confidence_threshold)
                
                # Extract from batch dimension if needed
                if len(matches0.shape) > 1:
                    matches0 = matches0[0]  # (B, N) -> (N,)
                
                matches0 = matches0.cpu().numpy().astype(int)
                
                # Extract valid matches
                matched_indices = []
                matched_pts_cam0 = []
                matched_pts_cam1 = []
                
                for i, match_idx in enumerate(matches0):
                    if match_idx >= 0:  # Valid match
                        matched_indices.append((i, int(match_idx)))
                        matched_pts_cam0.append(kpts0[i])
                        matched_pts_cam1.append(kpts1[int(match_idx)])
                
                if len(matched_indices) == 0:
                    return None
                
                matched_pts_cam0 = np.array(matched_pts_cam0, dtype=np.float32)
                matched_pts_cam1 = np.array(matched_pts_cam1, dtype=np.float32)
                
                # Get match confidence
                if "confidence" in matches:
                    avg_confidence = float(matches["confidence"].cpu().mean())
                else:
                    avg_confidence = min(len(matched_indices) / max(len(kpts0), len(kpts1)), 1.0)
                
                return MatchResult(
                    id_cam0=None,
                    id_cam1=None,
                    confidence=avg_confidence,
                    num_matches=len(matched_indices),
                    matched_indices=matched_indices,
                    matched_pts_cam0=matched_pts_cam0,
                    matched_pts_cam1=matched_pts_cam1,
                    center_cam0=features_cam0.get('center', (0, 0)),
                    center_cam1=features_cam1.get('center', (0, 0))
                )
        except Exception as e:
            print(f"[LightGlueMatcher] Error during matching: {e}")
            import traceback
            traceback.print_exc()
            return self._match_features_placeholder(features_cam0, features_cam1, confidence_threshold)
    
    def _match_features_placeholder(self, 
                                   features_cam0: Dict, 
                                   features_cam1: Dict,
                                   confidence_threshold: float) -> Optional[MatchResult]:
        """Placeholder matching when LightGlue is not available."""
        kpts0 = features_cam0['keypoints']
        kpts1 = features_cam1['keypoints']
        desc0 = features_cam0['descriptors']
        desc1 = features_cam1['descriptors']
        
        if len(kpts0) == 0 or len(kpts1) == 0:
            return None
        
        # Simple nearest neighbor matching
        from scipy.spatial.distance import cdist
        distances = cdist(desc0, desc1, metric='euclidean')
        matched_indices = []
        matched_pts_cam0 = []
        matched_pts_cam1 = []
        
        for i in range(len(desc0)):
            best_j = np.argmin(distances[i])
            # Simple threshold
            if distances[i, best_j] < 0.8:
                matched_indices.append((i, best_j))
                matched_pts_cam0.append(kpts0[i])
                matched_pts_cam1.append(kpts1[best_j])
        
        if len(matched_indices) == 0:
            return None
        
        matched_pts_cam0 = np.array(matched_pts_cam0, dtype=np.float32)
        matched_pts_cam1 = np.array(matched_pts_cam1, dtype=np.float32)
        
        avg_distance = np.mean([distances[i, j] for i, j in matched_indices])
        confidence = max(0, 1 - avg_distance)
        
        return MatchResult(
            id_cam0=None,
            id_cam1=None,
            confidence=confidence,
            num_matches=len(matched_indices),
            matched_indices=matched_indices,
            matched_pts_cam0=matched_pts_cam0,
            matched_pts_cam1=matched_pts_cam1,
            center_cam0=features_cam0['center'],
            center_cam1=features_cam1['center']
        )
    
    def match_crop_sets(self, 
                       features_dict_cam0: Dict[int, Dict], 
                       features_dict_cam1: Dict[int, Dict],
                       confidence_threshold: float = 0.7) -> List[Tuple]:
        """
        Match all crops between two cameras.
        
        Args:
            features_dict_cam0: {id: features_dict} for camera 0
            features_dict_cam1: {id: features_dict} for camera 1
            confidence_threshold: Minimum confidence
            
        Returns:
            List of tuples: [(id_cam0, id_cam1, match_result), ...]
        """
        matching_results = []
        
        for id_cam0, features_cam0 in features_dict_cam0.items():
            best_match = None
            best_confidence = 0
            best_id_cam1 = None
            
            for id_cam1, features_cam1 in features_dict_cam1.items():
                match_result = self.match_features(
                    features_cam0, 
                    features_cam1, 
                    confidence_threshold
                )
                
                if match_result and match_result.confidence > best_confidence:
                    best_confidence = match_result.confidence
                    best_match = match_result
                    best_id_cam1 = id_cam1
            
            if best_match and best_confidence >= confidence_threshold:
                best_match.id_cam0 = id_cam0
                best_match.id_cam1 = best_id_cam1
                matching_results.append((id_cam0, best_id_cam1, best_match))
        
        print(f"[LightGlueMatcher] Found {len(matching_results)} matches")
        return matching_results


class IDAssociator:
    """
    Associate matched IDs across cameras.
    
    Create a consolidated list of matched ID pairs from matching results.
    """
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize ID associator.
        
        Args:
            confidence_threshold: Minimum confidence for valid match
        """
        self.confidence_threshold = confidence_threshold
    
    def associate_ids(self, 
                     matching_results: List[Tuple],
                     confidence_threshold: Optional[float] = None) -> Dict:
        """
        Create matched ID pairs from matching results.
        
        Args:
            matching_results: List of (id_cam0, id_cam1, match_result) tuples
            confidence_threshold: Override minimum confidence if provided
            
        Returns:
            dict: {
                'matched_pairs': [(id_0, id_1), ...],
                'id_map': {id_cam0: id_cam1},
                'confidence_map': {(id_0, id_1): confidence},
                'match_details': {(id_0, id_1): match_result},
                'unmatched_cam0': [id_0, ...],
                'unmatched_cam1': [id_1, ...],
            }
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        
        matched_pairs = []
        id_map = {}
        confidence_map = {}
        match_details = {}
        
        for id_cam0, id_cam1, match_result in matching_results:
            if match_result.confidence >= confidence_threshold:
                matched_pairs.append((id_cam0, id_cam1))
                id_map[id_cam0] = id_cam1
                confidence_map[(id_cam0, id_cam1)] = match_result.confidence
                match_details[(id_cam0, id_cam1)] = match_result
        
        # Find unmatched IDs
        all_cam0_ids = set(id_map.keys())
        all_cam1_ids = set(id_map.get(id0) for id0 in all_cam0_ids if id0 in id_map)
        
        # This is simplified - would need full ID lists from camera to compute unmatched
        unmatched_cam0 = []
        unmatched_cam1 = []
        
        return {
            'matched_pairs': matched_pairs,
            'id_map': id_map,
            'confidence_map': confidence_map,
            'match_details': match_details,
            'num_matched': len(matched_pairs),
            'unmatched_cam0': unmatched_cam0,
            'unmatched_cam1': unmatched_cam1,
        }


def filter_matches_by_bboxes(matched_points_cam0: np.ndarray,
                            matched_points_cam1: np.ndarray,
                            bbox_cam0: Tuple[int, int, int, int],
                            bbox_cam1: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Filter matched points to keep only those within both bounding boxes.
    
    Args:
        matched_points_cam0: Matched points in camera 0 (N, 2) format
        matched_points_cam1: Matched points in camera 1 (N, 2) format
        bbox_cam0: Bounding box in cam0 (x1, y1, x2, y2)
        bbox_cam1: Bounding box in cam1 (x1, y1, x2, y2)
        
    Returns:
        (filtered_pts_cam0, filtered_pts_cam1, num_valid_matches)
    """
    if len(matched_points_cam0) == 0 or len(matched_points_cam1) == 0:
        return np.empty((0, 2)), np.empty((0, 2)), 0
    
    x1_0, y1_0, x2_0, y2_0 = bbox_cam0
    x1_1, y1_1, x2_1, y2_1 = bbox_cam1
    
    # Check which points are inside both bboxes
    valid_mask = (
        (matched_points_cam0[:, 0] >= x1_0) & (matched_points_cam0[:, 0] <= x2_0) &
        (matched_points_cam0[:, 1] >= y1_0) & (matched_points_cam0[:, 1] <= y2_0) &
        (matched_points_cam1[:, 0] >= x1_1) & (matched_points_cam1[:, 0] <= x2_1) &
        (matched_points_cam1[:, 1] >= y1_1) & (matched_points_cam1[:, 1] <= y2_1)
    )
    
    filtered_cam0 = matched_points_cam0[valid_mask]
    filtered_cam1 = matched_points_cam1[valid_mask]
    
    return filtered_cam0, filtered_cam1, np.sum(valid_mask)


# Example usage
if __name__ == "__main__":
    matcher = LightGlueMatcher()
    associator = IDAssociator()
    
    # Placeholder feature dicts
    features_cam0 = {
        1: {
            'keypoints': np.random.randn(10, 2),
            'descriptors': np.random.randn(10, 256),
            'scores': np.ones(10),
            'center': (100, 100)
        }
    }
    
    features_cam1 = {
        1: {
            'keypoints': np.random.randn(10, 2),
            'descriptors': np.random.randn(10, 256),
            'scores': np.ones(10),
            'center': (105, 105)
        }
    }
    
    # Test matching
    results = matcher.match_crop_sets(features_cam0, features_cam1)
    print(f"Matching results: {len(results)} matches")
