"""
Stereo Rectification using calibrated camera parameters.

Loads intrinsics (K), distortion (D), rotation (R), and translation (T)
from an OpenCV-format YAML file and precomputes undistort+rectify maps
for real-time cv2.remap().
"""

import cv2
import numpy as np


class StereoRectifier:
    """Perform stereo rectification using precomputed maps.

    Loads calibration parameters from an OpenCV YAML file and precomputes
    rectification maps for both left and right cameras. Subsequent calls
    to `rectify()` use fast `cv2.remap()`.

    Args:
        calibration_path: Path to the OpenCV YAML calibration file.
        image_size: (width, height) of input images.
    """

    def __init__(self, calibration_path: str, image_size: tuple[int, int] = (640, 480)):
        self.image_size = image_size  # (W, H)

        # Load calibration parameters
        self.K_l, self.D_l, self.K_r, self.D_r, self.R, self.T = (
            self._load_calibration(calibration_path)
        )

        # Compute rectification transforms
        self.R1, self.R2, self.P1, self.P2, self.Q, self.roi1, self.roi2 = (
            cv2.stereoRectify(
                self.K_l, self.D_l,
                self.K_r, self.D_r,
                image_size,   # (W, H)
                self.R, self.T,
                flags=cv2.CALIB_ZERO_DISPARITY,
                alpha=0,      # Crop to valid region
            )
        )

        # Precompute undistort + rectify maps (once at init)
        self.map1_l, self.map2_l = cv2.initUndistortRectifyMap(
            self.K_l, self.D_l, self.R1, self.P1,
            image_size, cv2.CV_32FC1,
        )
        self.map1_r, self.map2_r = cv2.initUndistortRectifyMap(
            self.K_r, self.D_r, self.R2, self.P2,
            image_size, cv2.CV_32FC1,
        )

        print(f"[StereoRectifier] 已初始化，影像尺寸: {image_size}")
        print(f"  左相機校正後內參 P1:\n{self.P1}")
        print(f"  右相機校正後內參 P2:\n{self.P2}")

    def rectify(
        self,
        img_left: np.ndarray,
        img_right: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply stereo rectification to a left-right image pair.

        Args:
            img_left: Left camera image (H, W, 3), uint8.
            img_right: Right camera image (H, W, 3), uint8.

        Returns:
            Tuple of (rectified_left, rectified_right), both (H, W, 3) uint8.
        """
        rect_l = cv2.remap(
            img_left, self.map1_l, self.map2_l, cv2.INTER_LINEAR,
        )
        rect_r = cv2.remap(
            img_right, self.map1_r, self.map2_r, cv2.INTER_LINEAR,
        )
        return rect_l, rect_r

    def get_rectified_intrinsics(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the rectified intrinsic matrices (3x3) for left and right cameras.

        These are extracted from the projection matrices P1 and P2 output
        by cv2.stereoRectify(). The 3×3 intrinsic block is the same for both
        cameras after rectification.

        Returns:
            Tuple of (K_rect_left, K_rect_right), both (3, 3) float64.
        """
        K_rect_l = self.P1[:3, :3].copy()
        K_rect_r = self.P2[:3, :3].copy()
        return K_rect_l, K_rect_r

    def get_baseline(self) -> float:
        """Get the stereo baseline in the same unit as calibration (usually mm).

        Returns:
            Baseline distance (positive scalar).
        """
        # P2[0,3] = -fx * Tx  =>  Tx = -P2[0,3] / P2[0,0]
        baseline = abs(self.P2[0, 3] / self.P2[0, 0])
        return baseline

    @staticmethod
    def _load_calibration(path: str):
        """Load stereo calibration from OpenCV YAML file.

        Returns:
            (K_l, D_l, K_r, D_r, R, T) as numpy arrays.
        """
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise FileNotFoundError(f"無法開啟校正檔案: {path}")

        K_l = fs.getNode("K_l").mat()
        D_l = fs.getNode("D_l").mat()
        K_r = fs.getNode("K_r").mat()
        D_r = fs.getNode("D_r").mat()
        R = fs.getNode("R").mat()
        T = fs.getNode("T").mat()
        fs.release()

        print(f"[StereoRectifier] 已載入校正參數: {path}")
        return K_l, D_l, K_r, D_r, R, T
