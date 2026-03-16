"""
Camera Pose Manager — Precomputed constant tensors on GPU.

Since the robotic arm cameras are fixed, intrinsics K and extrinsics [R|T]
are constant. This module precomputes them as CUDA tensors once and serves
batch-sized copies with zero CPU overhead.
"""

import cv2
import numpy as np
import torch


class PoseManager:
    """Manage precomputed camera pose tensors on GPU.

    Loads calibration, computes rectified intrinsics and extrinsics,
    and stores them as constant CUDA tensors for the 6-frame batch:
        {L_{t-2}, L_{t-1}, L_t, R_{t-2}, R_{t-1}, R_t}

    The first 3 frames share the left camera parameters,
    the last 3 frames share the right camera parameters.

    Args:
        calibration_path: Path to OpenCV YAML calibration file.
        image_size: (width, height) of input images.
        device: CUDA device string.
    """

    def __init__(
        self,
        calibration_path: str,
        image_size: tuple[int, int] = (640, 480),
        device: str = "cuda",
        temporal_frames: int = 3,
    ):
        self.device = torch.device(device)
        self.image_size = image_size

        # Load raw calibration
        K_l, D_l, K_r, D_r, R, T = self._load_calibration(calibration_path)

        # Compute rectified parameters
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            K_l, D_l, K_r, D_r, image_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
        )

        # Rectified intrinsics (3x3)
        K_rect_l = P1[:3, :3].astype(np.float32)
        K_rect_r = P2[:3, :3].astype(np.float32)

        # Build extrinsics: world-to-camera (4x4) for left and right
        # Left camera is the world origin after rectification
        ext_l = np.eye(4, dtype=np.float32)

        # Right camera extrinsic
        ext_r = np.eye(4, dtype=np.float32)
        ext_r[:3, :3] = R.astype(np.float32)
        ext_r[:3, 3] = T.flatten().astype(np.float32)

        # Build batch tensors: temporal_frames * 2 frames
        # Intrinsics: (2*N, 3, 3)
        intrinsics = np.stack([K_rect_l] * temporal_frames + [K_rect_r] * temporal_frames, axis=0)
        # Extrinsics: (2*N, 4, 4)
        extrinsics = np.stack([ext_l] * temporal_frames + [ext_r] * temporal_frames, axis=0)

        # Convert to GPU constant tensors
        self._intrinsics = torch.from_numpy(intrinsics).to(self.device)
        self._extrinsics = torch.from_numpy(extrinsics).to(self.device)

        # Store individual camera params for other uses
        self._K_l = torch.from_numpy(K_rect_l).to(self.device)
        self._K_r = torch.from_numpy(K_rect_r).to(self.device)
        self._ext_l = torch.from_numpy(ext_l).to(self.device)
        self._ext_r = torch.from_numpy(ext_r).to(self.device)

        print(f"[PoseManager] 已在 {device} 上預計算位姿張量")
        print(f"  Intrinsics batch shape: {self._intrinsics.shape}")
        print(f"  Extrinsics batch shape: {self._extrinsics.shape}")

    def get_batch_intrinsics(self) -> torch.Tensor:
        """Get batch intrinsics tensor.

        Returns:
            (6, 3, 3) float32 tensor on GPU.
        """
        return self._intrinsics

    def get_batch_extrinsics(self) -> torch.Tensor:
        """Get batch extrinsics tensor (world-to-camera).

        Returns:
            (6, 4, 4) float32 tensor on GPU.
        """
        return self._extrinsics

    def get_left_intrinsic(self) -> torch.Tensor:
        """Get left camera rectified intrinsic (3, 3)."""
        return self._K_l

    def get_right_intrinsic(self) -> torch.Tensor:
        """Get right camera rectified intrinsic (3, 3)."""
        return self._K_r

    def get_left_extrinsic(self) -> torch.Tensor:
        """Get left camera extrinsic (4, 4) — identity."""
        return self._ext_l

    def get_right_extrinsic(self) -> torch.Tensor:
        """Get right camera extrinsic (4, 4)."""
        return self._ext_r

    @staticmethod
    def _load_calibration(path: str):
        """Load stereo calibration from OpenCV YAML file."""
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
        return K_l, D_l, K_r, D_r, R, T
