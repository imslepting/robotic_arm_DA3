"""
Confidence-based Filter for depth maps and Gaussian Splatting.

Removes low-confidence regions and edge noise to produce
clean point clouds / Gaussians for rendering.
"""

import cv2
import numpy as np


class ConfidenceFilter:
    """Filter depth/point data by confidence score.

    Args:
        threshold: Minimum confidence score to keep (default: 0.8).
        use_morphology: Apply morphological operations to remove edge noise.
        morph_kernel_size: Kernel size for morphological operations.
    """

    def __init__(
        self,
        threshold: float = 0.8,
        use_morphology: bool = True,
        morph_kernel_size: int = 3,
    ):
        self.threshold = threshold
        self.use_morphology = use_morphology
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )

    def compute_mask(self, confidence: np.ndarray) -> np.ndarray:
        """Compute a binary mask from confidence map.

        Args:
            confidence: (H, W) float32 confidence map.

        Returns:
            (H, W) bool mask where True = keep, False = discard.
        """
        mask = confidence >= self.threshold

        if self.use_morphology:
            # Convert to uint8 for morphological ops
            mask_u8 = mask.astype(np.uint8) * 255
            # Erosion removes thin noisy edges
            mask_u8 = cv2.erode(mask_u8, self._kernel, iterations=1)
            # Dilation restores slightly
            mask_u8 = cv2.dilate(mask_u8, self._kernel, iterations=1)
            mask = mask_u8 > 0

        return mask

    def filter_points(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        confidence: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter 3D points by confidence.

        Args:
            points: (H, W, 3) or (N, 3) world-space points.
            colors: (H, W, 3) or (N, 3) RGB colors.
            confidence: (H, W) or (N,) confidence values.

        Returns:
            Filtered (points, colors, confidence), all (M, 3) / (M,).
        """
        # Flatten if spatial
        if points.ndim == 3:
            H, W = points.shape[:2]
            points = points.reshape(-1, 3)
            colors = colors.reshape(-1, 3)
            confidence = confidence.reshape(-1)

        mask = confidence >= self.threshold
        return points[mask], colors[mask], confidence[mask]

    def filter_gaussians(
        self,
        means: np.ndarray,
        scales: np.ndarray,
        rotations: np.ndarray,
        colors: np.ndarray,
        opacities: np.ndarray,
        confidence: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        """Filter Gaussian splat parameters by confidence.

        Args:
            means: (N, 3) Gaussian centers.
            scales: (N, 3) Gaussian scales.
            rotations: (N, 4) Gaussian quaternions.
            colors: (N, 3) RGB colors.
            opacities: (N,) opacities.
            confidence: (N,) confidence scores.

        Returns:
            Filtered (means, scales, rotations, colors, opacities).
        """
        mask = confidence >= self.threshold
        return (
            means[mask],
            scales[mask],
            rotations[mask],
            colors[mask],
            opacities[mask],
        )
