"""Tests for DepthDecoder."""

import numpy as np
import pytest

from pipeline.depth_decoder import DepthDecoder


class TestDepthDecoder:
    """Unit tests for depth post-processing."""

    def _make_inference_result(self, H=48, W=64):
        """Create a synthetic inference result."""
        return {
            "depth": np.random.rand(6, H, W).astype(np.float32) * 10,
            "conf": np.random.rand(6, H, W).astype(np.float32),
            "time_ms": 42.0,
        }

    def test_decode_basic(self):
        decoder = DepthDecoder(confidence_threshold=0.5)
        result = self._make_inference_result()
        decoded = decoder.decode(result)

        assert "depth_left" in decoded
        assert "depth_right" in decoded
        assert "conf_left" in decoded
        assert "conf_right" in decoded
        assert "mask_left" in decoded
        assert "mask_right" in decoded
        assert decoded["time_ms"] == 42.0

    def test_correct_indices(self):
        """Left should be index 2, right should be index 5."""
        decoder = DepthDecoder()
        result = self._make_inference_result()
        # Set distinctive values
        result["depth"][2] = 42.0
        result["depth"][5] = 99.0

        decoded = decoder.decode(result)
        np.testing.assert_array_equal(decoded["depth_left"], 42.0)
        np.testing.assert_array_equal(decoded["depth_right"], 99.0)

    def test_confidence_mask(self):
        decoder = DepthDecoder(confidence_threshold=0.7)
        result = self._make_inference_result()
        result["conf"][2] = 0.5  # Below threshold
        result["conf"][5] = 0.9  # Above threshold

        decoded = decoder.decode(result)
        assert not decoded["mask_left"].any()  # All below 0.7
        assert decoded["mask_right"].all()     # All above 0.7

    def test_no_confidence(self):
        """When no confidence is provided, defaults to ones."""
        decoder = DepthDecoder()
        result = {
            "depth": np.ones((6, 48, 64), dtype=np.float32),
            "time_ms": 10.0,
        }
        decoded = decoder.decode(result)
        np.testing.assert_array_equal(decoded["conf_left"], 1.0)
        assert decoded["mask_left"].all()

    def test_depth_non_negative(self):
        decoder = DepthDecoder()
        result = self._make_inference_result()
        result["depth"][2] = -5.0  # Negative depth
        decoded = decoder.decode(result)
        assert (decoded["depth_left"] >= 0).all()

    def test_fuse_stereo(self):
        decoder = DepthDecoder()
        depth_l = np.full((10, 10), 5.0, dtype=np.float32)
        depth_r = np.full((10, 10), 3.0, dtype=np.float32)
        conf_l = np.full((10, 10), 0.8, dtype=np.float32)
        conf_r = np.full((10, 10), 0.2, dtype=np.float32)

        fused_d, fused_c = decoder.fuse_stereo_depth(depth_l, depth_r, conf_l, conf_r)
        # With conf_l=0.8, conf_r=0.2, fused should be closer to depth_l
        assert fused_d[0, 0] > 4.0  # Weighted toward depth_l=5.0
        assert fused_d[0, 0] < 5.0  # But not exactly 5.0
