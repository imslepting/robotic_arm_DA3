"""
Feed-forward Gaussian Splatting Projector.

Converts depth maps + intrinsics into 3D Gaussian Splat parameters
WITHOUT optimization — pure feed-forward projection suitable for
real-time rendering on resource-constrained hardware.

Pipeline:
  depth + K → back-project → 3D means (μ)
  depth gradient → surface normal → quaternion rotations (q)
  local depth variation → scales (s)
  input RGB → colors
  confidence map → opacities
"""

import numpy as np
import torch


class GaussianProjector:
    """Feed-forward depth-to-Gaussian projector.

    Args:
        scale_multiplier: Base scale for Gaussians relative to depth.
        default_opacity: Default opacity for Gaussians.
        max_points: Maximum number of Gaussians to produce.
        device: CUDA device string.
    """

    def __init__(
        self,
        scale_multiplier: float = 0.01,
        default_opacity: float = 0.95,
        max_points: int = 500_000,
        device: str = "cuda",
    ):
        self.scale_multiplier = scale_multiplier
        self.default_opacity = default_opacity
        self.max_points = max_points
        self.device = torch.device(device)

    def project(
        self,
        depth: np.ndarray,
        color_image: np.ndarray,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
        confidence: np.ndarray | None = None,
        mask: np.ndarray | None = None,
    ) -> dict:
        """Project depth map to 3D Gaussian Splat parameters.

        Args:
            depth: (H, W) float32 metric depth map.
            color_image: (H, W, 3) uint8 BGR color image.
            intrinsic: (3, 3) float32 camera intrinsic matrix.
            extrinsic: (4, 4) float32 world-to-camera extrinsic matrix.
            confidence: (H, W) float32 confidence map (optional).
            mask: (H, W) bool mask of valid pixels (optional).

        Returns:
            Dictionary with:
                'means': (N, 3) float32 world-space Gaussian centers
                'scales': (N, 3) float32 Gaussian scales
                'rotations': (N, 4) float32 quaternions (wxyz)
                'colors': (N, 3) float32 RGB colors [0, 1]
                'opacities': (N,) float32 opacities
                'num_points': int
        """
        H, W = depth.shape

        # Resize color_image / confidence / mask to match depth dimensions
        # (DA3 internally resizes to process_res, so depth may be smaller)
        import cv2 as _cv2
        ch, cw = color_image.shape[:2]
        if (ch, cw) != (H, W):
            color_image = _cv2.resize(color_image, (W, H), interpolation=_cv2.INTER_LINEAR)
            if confidence is not None:
                confidence = _cv2.resize(confidence, (W, H), interpolation=_cv2.INTER_LINEAR)
            if mask is not None:
                mask = _cv2.resize(mask.astype(np.uint8), (W, H), interpolation=_cv2.INTER_NEAREST).astype(bool)

        # Convert to torch tensors on GPU
        depth_t = torch.from_numpy(depth).float().to(self.device)
        K = torch.from_numpy(intrinsic).float().to(self.device)
        ext = torch.from_numpy(extrinsic).float().to(self.device)

        # Build valid mask
        if mask is not None:
            valid = torch.from_numpy(mask).to(self.device)
        else:
            valid = depth_t > 0

        if confidence is not None:
            conf_t = torch.from_numpy(confidence).float().to(self.device)
        else:
            conf_t = torch.ones(H, W, device=self.device)

        valid = valid & (depth_t > 1e-3)

        # 1. Back-project to 3D points (camera space)
        points_cam = self._backproject(depth_t, K, H, W)  # (H, W, 3)

        # 2. Transform to world space
        c2w = torch.inverse(ext)  # camera-to-world
        points_world = self._transform_points(points_cam, c2w)  # (H, W, 3)

        # 3. Compute surface normals from depth gradient
        normals_world = self._compute_normals(points_world)  # (H, W, 3)

        # 4. Convert normals to quaternion rotations
        rotations = self._normals_to_quaternions(normals_world)  # (H, W, 4)

        # 5. Compute scales from local depth variation
        scales = self._compute_scales(depth_t, K)  # (H, W, 3)

        # 6. Extract colors
        color_rgb = torch.from_numpy(
            color_image[..., ::-1].copy()  # BGR → RGB
        ).float().to(self.device) / 255.0  # (H, W, 3)

        # 7. Opacities from confidence
        opacities = conf_t * self.default_opacity  # (H, W)

        # Apply mask and flatten
        valid_flat = valid.reshape(-1)
        means = points_world.reshape(-1, 3)[valid_flat]
        scales_out = scales.reshape(-1, 3)[valid_flat]
        rots_out = rotations.reshape(-1, 4)[valid_flat]
        colors_out = color_rgb.reshape(-1, 3)[valid_flat]
        opacities_out = opacities.reshape(-1)[valid_flat]

        # Subsample if too many points
        N = means.shape[0]
        if N > self.max_points:
            idx = torch.randperm(N, device=self.device)[:self.max_points]
            means = means[idx]
            scales_out = scales_out[idx]
            rots_out = rots_out[idx]
            colors_out = colors_out[idx]
            opacities_out = opacities_out[idx]
            N = self.max_points

        return {
            "means": means.cpu().numpy(),
            "scales": scales_out.cpu().numpy(),
            "rotations": rots_out.cpu().numpy(),
            "colors": colors_out.cpu().numpy(),
            "opacities": opacities_out.cpu().numpy(),
            "num_points": N,
        }

    def _backproject(
        self, depth: torch.Tensor, K: torch.Tensor, H: int, W: int
    ) -> torch.Tensor:
        """Back-project depth map to 3D points in camera space.

        Args:
            depth: (H, W) depth values.
            K: (3, 3) intrinsic matrix.

        Returns:
            (H, W, 3) camera-space 3D points.
        """
        # Create pixel grid
        u = torch.arange(W, device=self.device).float()
        v = torch.arange(H, device=self.device).float()
        u, v = torch.meshgrid(u, v, indexing="xy")  # (H, W)

        # Unproject
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        return torch.stack([x, y, z], dim=-1)  # (H, W, 3)

    def _transform_points(
        self, points: torch.Tensor, c2w: torch.Tensor
    ) -> torch.Tensor:
        """Transform points from camera to world space.

        Args:
            points: (H, W, 3) camera-space points.
            c2w: (4, 4) camera-to-world matrix.

        Returns:
            (H, W, 3) world-space points.
        """
        R = c2w[:3, :3]  # (3, 3)
        t = c2w[:3, 3]   # (3,)
        return torch.einsum("ij,...j->...i", R, points) + t

    def _compute_normals(self, points: torch.Tensor) -> torch.Tensor:
        """Compute surface normals from spatial point gradients.

        Args:
            points: (H, W, 3) world-space points.

        Returns:
            (H, W, 3) unit normals.
        """
        # Spatial gradients
        dpdx = torch.zeros_like(points)
        dpdy = torch.zeros_like(points)
        dpdx[:, 1:-1, :] = points[:, 2:, :] - points[:, :-2, :]
        dpdy[1:-1, :, :] = points[2:, :, :] - points[:-2, :, :]

        # Cross product
        normals = torch.cross(dpdx, dpdy, dim=-1)
        norm = normals.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        normals = normals / norm

        return normals

    def _normals_to_quaternions(self, normals: torch.Tensor) -> torch.Tensor:
        """Convert surface normals to quaternion rotations (wxyz).

        The quaternion represents the rotation from the z-axis [0,0,1]
        to the surface normal direction.

        Args:
            normals: (H, W, 3) unit normals.

        Returns:
            (H, W, 4) quaternions in wxyz order.
        """
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device)

        # Dot product with z-axis
        dot = normals[..., 2]  # equivalent to normals · [0,0,1]

        # Cross product with z-axis
        cross = torch.stack([
            -normals[..., 1],   # z_axis × normals (simplified)
            normals[..., 0],
            torch.zeros_like(dot),
        ], dim=-1)

        cross_norm = cross.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # Quaternion: w = 1 + dot, xyz = cross
        w = (1.0 + dot).unsqueeze(-1)
        xyz = cross

        quat = torch.cat([w, xyz], dim=-1)  # (H, W, 4) wxyz
        quat = quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        return quat

    def _compute_scales(
        self, depth: torch.Tensor, K: torch.Tensor
    ) -> torch.Tensor:
        """Compute Gaussian scales from local depth variation.

        Scale is proportional to depth / focal_length (approximate
        pixel footprint in world space).

        Args:
            depth: (H, W) depth map.
            K: (3, 3) intrinsic matrix.

        Returns:
            (H, W, 3) scale values for each Gaussian.
        """
        fx, fy = K[0, 0], K[1, 1]

        # Pixel footprint in world coordinates
        scale_x = depth * self.scale_multiplier / fx
        scale_y = depth * self.scale_multiplier / fy
        # Z-scale: thinner along the normal direction
        scale_z = scale_x * 0.5

        return torch.stack([scale_x, scale_y, scale_z], dim=-1)
