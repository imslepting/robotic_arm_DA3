"""
Async Inference Engine for DA3 depth estimation.

Supports two backends:
  1. TensorRT: Load compiled .engine file for FP16 inference
  2. PyTorch:  Direct model inference with autocast FP16

Uses torch.cuda.Stream() for non-blocking inference so the capture
thread can continue grabbing frames while inference runs.
"""

import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class InferenceEngine:
    """DA3 inference engine with TensorRT / PyTorch backends.

    Args:
        model_name: DA3 model name (e.g. 'da3-small').
        trt_engine_path: Path to TensorRT .engine file. None = PyTorch fallback.
        weights_path: Path to model.safetensors (for PyTorch backend).
        config_path: Path to model config.json (for PyTorch backend).
        use_fp16: Use FP16 precision.
        use_async: Use CUDA stream for non-blocking inference.
        process_res: DA3 internal processing resolution.
        device: CUDA device string.
    """

    # ImageNet normalization
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(
        self,
        model_name: str = "da3-small",
        trt_engine_path: Optional[str] = None,
        weights_path: Optional[str] = None,
        config_path: Optional[str] = None,
        use_fp16: bool = True,
        use_async: bool = True,
        process_res: int = 504,
        use_gaussian_head: bool = False,
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.use_fp16 = use_fp16
        self.use_async = use_async
        self.process_res = process_res
        self.model_name = model_name
        self.use_gaussian_head = use_gaussian_head

        # Async inference stream
        self._stream = torch.cuda.Stream(device=self.device) if use_async else None
        self._last_result = None
        self._inference_event = torch.cuda.Event(enable_timing=True)
        self._start_event = torch.cuda.Event(enable_timing=True)

        # Move normalization constants to GPU
        self.MEAN = self.MEAN.to(self.device)
        self.STD = self.STD.to(self.device)

        # Try TensorRT first, then fall back to PyTorch
        self._backend = None
        self._model = None
        self._trt_context = None

        if trt_engine_path and Path(trt_engine_path).exists():
            self._init_tensorrt(trt_engine_path)
        else:
            if trt_engine_path:
                print(f"[InferenceEngine] TensorRT engine 不存在: {trt_engine_path}")
                print("  使用 PyTorch 後備方案")
            self._init_pytorch(model_name, weights_path, config_path)

        print(f"[InferenceEngine] 後端: {self._backend}, FP16: {use_fp16}, 異步: {use_async}")

    def _init_tensorrt(self, engine_path: str):
        """Initialize TensorRT backend."""
        try:
            import tensorrt as trt

            logger = trt.Logger(trt.Logger.WARNING)
            with open(engine_path, "rb") as f:
                engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

            self._trt_context = engine.create_execution_context()
            self._backend = "tensorrt"
            print(f"[InferenceEngine] TensorRT engine 已載入: {engine_path}")
        except ImportError:
            print("[InferenceEngine] TensorRT Python 套件未安裝，使用 PyTorch 後備方案")
            self._init_pytorch(self.model_name, None, None)
        except Exception as e:
            print(f"[InferenceEngine] TensorRT 初始化失敗: {e}")
            self._init_pytorch(self.model_name, None, None)

    def _init_pytorch(
        self,
        model_name: str,
        weights_path: Optional[str],
        config_path: Optional[str],
    ):
        """Initialize PyTorch backend."""
        from depth_anything_3.api import DepthAnything3

        if config_path and Path(config_path).exists():
            # Load from explicit config + weights
            with open(config_path) as f:
                config = json.load(f)
            model = DepthAnything3(**config)
            if weights_path and Path(weights_path).exists():
                from safetensors.torch import load_file
                state_dict = load_file(weights_path)
                model.load_state_dict(state_dict, strict=False)
        else:
            # Load from HuggingFace Hub using model name
            model = DepthAnything3(model_name=model_name)

        model.eval()
        model = model.to(self.device)
        self._model = model
        self._backend = "pytorch"

    def preprocess_frames(self, frames: list[np.ndarray]) -> torch.Tensor:
        """Convert a list of BGR uint8 frames to a normalized GPU tensor.

        Args:
            frames: List of N frames, each (H, W, 3) uint8 BGR.

        Returns:
            Tensor of shape (1, N, 3, H, W) float32 on GPU, ImageNet-normalized.
        """
        tensors = []
        for frame in frames:
            # BGR → RGB, HWC → CHW, [0,255] → [0,1]
            img = frame[..., ::-1].copy()  # BGR to RGB
            t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            tensors.append(t)

        batch = torch.stack(tensors, dim=0).to(self.device)  # (N, 3, H, W)

        # Resize to process_res if needed
        _, _, h, w = batch.shape
        if max(h, w) != self.process_res:
            scale = self.process_res / max(h, w)
            new_h = int(h * scale) // 14 * 14  # Align to patch size
            new_w = int(w * scale) // 14 * 14
            batch = F.interpolate(batch, size=(new_h, new_w), mode="bilinear", align_corners=False)

        # ImageNet normalize
        batch = (batch - self.MEAN) / self.STD

        return batch.unsqueeze(0)  # (1, N, 3, H', W')

    @torch.inference_mode()
    def infer(
        self,
        frames: list[np.ndarray],
        extrinsics: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        infer_gs: bool = False,
    ) -> dict:
        """Run inference on a batch of frames.

        Args:
            frames: List of 6 frames [L_{t-2}, L_{t-1}, L_t, R_{t-2}, R_{t-1}, R_t],
                    each (H, W, 3) uint8 BGR.
            extrinsics: (6, 4, 4) camera extrinsics tensor on GPU.
            intrinsics: (6, 3, 3) camera intrinsics tensor on GPU.
                infer_gs: Enable Gaussian Splatting head inference for this call.

        Returns:
            Dictionary with:
                'depth': (N, H, W) numpy float32 metric depth maps
                'conf': (N, H, W) numpy float32 confidence maps
                'gaussians': Gaussians object (if use_gaussian_head=True), else None
                'time_ms': inference time in milliseconds
        """
        batch = self.preprocess_frames(frames)

        # Add batch dimension: (6,...) -> (1,6,...)
        ext = extrinsics.unsqueeze(0).float() if extrinsics is not None else None
        ixt = intrinsics.unsqueeze(0).float() if intrinsics is not None else None

        if self._backend == "tensorrt":
            return self._infer_tensorrt(batch)
        else:
            return self._infer_pytorch(
                batch,
                extrinsics=ext,
                intrinsics=ixt,
                infer_gs=infer_gs,
            )

    @torch.inference_mode()
    def infer_async(self, frames: list[np.ndarray]) -> None:
        """Submit inference asynchronously (non-blocking).

        Results are retrieved via `get_async_result()`.
        """
        if not self.use_async or self._stream is None:
            # Fallback to synchronous
            self._last_result = self.infer(frames)
            return

        batch = self.preprocess_frames(frames)

        with torch.cuda.stream(self._stream):
            self._start_event.record(self._stream)

            if self._backend == "tensorrt":
                self._last_result = self._infer_tensorrt(batch, is_async=True)
            else:
                self._last_result = self._infer_pytorch(batch, is_async=True)

            self._inference_event.record(self._stream)

    def get_async_result(self) -> Optional[dict]:
        """Retrieve the result of the last async inference.

        Returns:
            Result dict, or None if not ready.
        """
        if self._stream is None:
            return self._last_result

        if self._inference_event.query():
            result = self._last_result
            self._last_result = None
            return result
        return None

    def wait_async(self) -> dict:
        """Wait for async inference to complete and return the result."""
        if self._stream is not None:
            self._inference_event.synchronize()
        result = self._last_result
        self._last_result = None
        return result

    def _infer_pytorch(
        self,
        batch: torch.Tensor,
        extrinsics: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        is_async: bool = False,
        infer_gs: bool = False,
    ) -> dict:
        """Run inference using PyTorch backend."""
        autocast_dtype = torch.float16 if self.use_fp16 else torch.float32

        torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()

        # Keep backward compatibility: either per-call infer_gs or init-time use_gaussian_head.
        use_gs = bool(infer_gs or self.use_gaussian_head)

        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            output = self._model.model(
                batch,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                infer_gs=use_gs,
            )

        if not is_async:
            torch.cuda.synchronize(self.device)

        t1 = time.perf_counter()
        time_ms = (t1 - t0) * 1000

        # Extract results
        depth = output.depth.squeeze(0).float()  # (N, H, W)
        conf = output.depth_conf.squeeze(0).float() if hasattr(output, 'depth_conf') and output.depth_conf is not None else None

        result = {
            "depth": depth.cpu().numpy(),
            "time_ms": time_ms,
        }
        if conf is not None:
            result["conf"] = conf.cpu().numpy()
        
        # Extract Gaussians if enabled
        if use_gs and hasattr(output, 'gaussians') and output.gaussians is not None:
            result["gaussians"] = output.gaussians
        else:
            result["gaussians"] = None

        return result

    def _infer_tensorrt(self, batch: torch.Tensor, is_async: bool = False) -> dict:
        """Run inference using TensorRT backend."""
        # TensorRT inference placeholder —
        # actual implementation depends on specific TRT bindings
        raise NotImplementedError(
            "TensorRT inference not yet implemented. "
            "Use PyTorch fallback by setting trt_engine_path=null."
        )

    @property
    def backend_name(self) -> str:
        return self._backend or "none"
