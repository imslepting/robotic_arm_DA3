"""
Depth Decoder — Post-processing for DA3 inference output.

Extracts the current time-step (t) depth maps for left and right views
from the 6-frame batch output, applies confidence filtering, and
optionally fuses the stereo depth maps.
"""

import numpy as np


class DepthDecoder:
    """Post-process DA3 inference output to extract usable depth maps.

    The 6-frame batch is ordered as:
        [L_{t-2}, L_{t-1}, L_t, R_{t-2}, R_{t-1}, R_t]
        indices:  0        1      2       3        4      5

    We extract index 2 (left at time t) and index 5 (right at time t)
    as the primary depth maps.

    Args:
        confidence_threshold: Minimum confidence to keep a depth value.
    """

    # Batch indices for current time step
    IDX_LEFT_T = 2
    IDX_RIGHT_T = 5

    def __init__(self, confidence_threshold: float = 0.8):
        self.confidence_threshold = confidence_threshold

    def decode(self, inference_result: dict) -> dict:
        """Decode inference output to extract current-frame depth maps.

        Args:
            inference_result: Dictionary from InferenceEngine with:
                'depth': (6, H, W) float32 depth maps
                'conf':  (6, H, W) float32 confidence maps (optional)
                'time_ms': inference time

        Returns:
            Dictionary with:
                'depth_left': (H, W) float32 depth map for left camera at time t
                'depth_right': (H, W) float32 depth map for right camera at time t
                'conf_left': (H, W) float32 confidence map (or ones if unavailable)
                'conf_right': (H, W) float32 confidence map
                'mask_left': (H, W) bool mask where conf >= threshold
                'mask_right': (H, W) bool mask
                'depth_all': (6, H, W) all depth maps
                'time_ms': inference time
        """
        depth_all = inference_result["depth"]  # (6, H, W)
        time_ms = inference_result.get("time_ms", 0)

        # Extract current time step
        depth_left = depth_all[self.IDX_LEFT_T]   # (H, W)
        depth_right = depth_all[self.IDX_RIGHT_T]  # (H, W)

        # Confidence maps
        if "conf" in inference_result and inference_result["conf"] is not None:
            conf_all = inference_result["conf"]
            conf_left = conf_all[self.IDX_LEFT_T]
            conf_right = conf_all[self.IDX_RIGHT_T]
        else:
            conf_left = np.ones_like(depth_left)
            conf_right = np.ones_like(depth_right)

        # Confidence masks
        mask_left = conf_left >= self.confidence_threshold
        mask_right = conf_right >= self.confidence_threshold

        # Ensure depth is positive
        depth_left = np.maximum(depth_left, 0.0)
        depth_right = np.maximum(depth_right, 0.0)

        return {
            "depth_left": depth_left,
            "depth_right": depth_right,
            "conf_left": conf_left,
            "conf_right": conf_right,
            "mask_left": mask_left,
            "mask_right": mask_right,
            "depth_all": depth_all,
            "time_ms": time_ms,
        }

    def fuse_stereo_depth(
        self,
        depth_left: np.ndarray,
        depth_right: np.ndarray,
        conf_left: np.ndarray,
        conf_right: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fuse left and right depth maps using confidence-weighted average.

        This is a simple fusion for overlapping regions. Since both cameras
        observe the same scene from different angles, we use confidence
        to weight each depth estimate.

        Args:
            depth_left: (H, W) left depth map.
            depth_right: (H, W) right depth map (in left camera frame).
            conf_left: (H, W) left confidence map.
            conf_right: (H, W) right confidence map.

        Returns:
            Tuple of (fused_depth, fused_confidence), both (H, W).
        """
        # Normalize confidence for weighting
        total_conf = conf_left + conf_right + 1e-8
        w_left = conf_left / total_conf
        w_right = conf_right / total_conf

        fused_depth = w_left * depth_left + w_right * depth_right
        fused_conf = np.maximum(conf_left, conf_right)

        return fused_depth, fused_conf
