"""
Viser-based Real-time 3D Viewer for Gaussian Splatting.

Renders 3D Gaussian splats and point clouds using the Viser
web-based 3D visualization library.
"""

import threading
import time
from typing import Optional

import numpy as np

try:
    import viser
    import viser.transforms as tf
    HAS_VISER = True
except ImportError:
    HAS_VISER = False
    print("[ViserRenderer] viser 套件未安裝，3D 視覺化功能不可用")


class ViserRenderer:
    """Real-time 3D viewer using Viser.

    Creates a web server for browser-based 3D visualization.
    Supports rendering point clouds with per-point colors.

    Args:
        host: Server host address.
        port: Server port.
        point_size: Default point size.
        background_color: RGB background color, each [0, 1].
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        point_size: float = 0.005,
        background_color: tuple[float, ...] = (0.1, 0.1, 0.15),
    ):
        if not HAS_VISER:
            raise ImportError("viser 套件未安裝。請執行: pip install viser")

        self.host = host
        self.port = port
        self.point_size = point_size
        self.background_color = background_color

        # Initialize Viser server
        self._server = viser.ViserServer(host=host, port=port)

        # State tracking
        self._frame_count = 0
        self._fps = 0.0
        self._last_fps_time = time.time()
        self._fps_frame_count = 0
        self._lock = threading.Lock()

        # UI controls
        self._setup_ui()

        print(f"[ViserRenderer] 3D 檢視器已啟動: http://{host}:{port}")

    def _setup_ui(self):
        """Set up UI control panel."""
        with self._server.gui.add_folder("控制面板"):
            self._gui_conf_threshold = self._server.gui.add_slider(
                "信賴度閾值",
                min=0.0,
                max=1.0,
                step=0.05,
                initial_value=0.8,
            )
            self._gui_point_size = self._server.gui.add_slider(
                "點大小",
                min=0.001,
                max=0.05,
                step=0.001,
                initial_value=self.point_size,
            )
            self._gui_max_points = self._server.gui.add_slider(
                "最大點數 (千)",
                min=10,
                max=1000,
                step=10,
                initial_value=500,
            )

        with self._server.gui.add_folder("狀態資訊"):
            self._gui_fps = self._server.gui.add_text(
                "FPS", initial_value="0.0",
                disabled=True,
            )
            self._gui_points = self._server.gui.add_text(
                "點數", initial_value="0",
                disabled=True,
            )

    def update_point_cloud(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        name: str = "scene",
    ) -> None:
        """Update the displayed point cloud.

        Args:
            points: (N, 3) float32 world-space positions.
            colors: (N, 3) float32 or uint8 RGB colors.
                    If float32 and max > 1, assumed [0, 255].
                    If uint8, will be converted to [0, 255].
        """
        with self._lock:
            # Ensure colors are uint8 [0, 255]
            if colors.dtype == np.float32 or colors.dtype == np.float64:
                if colors.max() <= 1.0:
                    colors = (colors * 255).astype(np.uint8)
                else:
                    colors = colors.astype(np.uint8)

            # Apply max points from UI
            max_pts = int(self._gui_max_points.value * 1000)
            if len(points) > max_pts:
                idx = np.random.choice(len(points), max_pts, replace=False)
                points = points[idx]
                colors = colors[idx]

            # Update point cloud in scene
            point_size = self._gui_point_size.value
            self._server.scene.add_point_cloud(
                f"/{name}/points",
                points=points.astype(np.float32),
                colors=colors,
                point_size=point_size,
                point_shape="rounded",
            )

            # Update stats
            self._frame_count += 1
            self._fps_frame_count += 1

            now = time.time()
            dt = now - self._last_fps_time
            if dt >= 1.0:
                self._fps = self._fps_frame_count / dt
                self._fps_frame_count = 0
                self._last_fps_time = now
                self._gui_fps.value = f"{self._fps:.1f}"

            self._gui_points.value = f"{len(points):,}"

    def update_cameras(
        self,
        extrinsics: np.ndarray,
        intrinsics: np.ndarray,
        image_hw: tuple[int, int] = (480, 640),
        name: str = "cameras",
    ) -> None:
        """Visualize camera frustums in the scene.

        Args:
            extrinsics: (N, 4, 4) world-to-camera matrices.
            intrinsics: (N, 3, 3) intrinsic matrices.
            image_hw: (height, width) for frustum aspect ratio.
        """
        H, W = image_hw
        for i in range(len(extrinsics)):
            ext = extrinsics[i]
            ixt = intrinsics[i]

            w2c = ext
            c2w = np.linalg.inv(w2c)

            fx = ixt[0, 0]
            fov = 2 * np.arctan(W / (2 * fx))

            self._server.scene.add_camera_frustum(
                f"/{name}/cam_{i}",
                fov=fov,
                aspect=W / H,
                scale=0.05,
                wxyz=tf.SO3.from_matrix(c2w[:3, :3]).wxyz,
                position=c2w[:3, 3],
                color=(100, 200, 255) if i < 3 else (255, 200, 100),
            )

    def get_confidence_threshold(self) -> float:
        """Get current confidence threshold from UI slider."""
        return self._gui_conf_threshold.value

    @property
    def fps(self) -> float:
        """Current rendering FPS."""
        return self._fps

    @property
    def frame_count(self) -> int:
        """Total frames rendered."""
        return self._frame_count

    def add_depth_image(
        self,
        depth: np.ndarray,
        name: str = "depth_preview",
    ) -> None:
        """Add a depth visualization as a 2D image overlay.

        Args:
            depth: (H, W) float32 depth map.
        """
        # Normalize depth for visualization
        d_min = np.min(depth[depth > 0]) if np.any(depth > 0) else 0
        d_max = np.max(depth)
        if d_max - d_min > 1e-6:
            depth_norm = (depth - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(depth)

        # Apply colormap (turbo)
        import cv2
        depth_color = cv2.applyColorMap(
            (depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO
        )
        depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)

        # Currently viser doesn't have 2D image overlay in all versions
        # This is a placeholder for when the feature is available

    def shutdown(self):
        """Gracefully shutdown the Viser server."""
        print("[ViserRenderer] 正在關閉 3D 檢視器...")
        # Viser doesn't require explicit cleanup
