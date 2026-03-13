"""Tests for ConfidenceFilter."""

import numpy as np
import pytest

from pipeline.confidence_filter import ConfidenceFilter


class TestConfidenceFilter:
    """Unit tests for the confidence-based filter."""

    def test_basic_mask(self):
        filt = ConfidenceFilter(threshold=0.5, use_morphology=False)
        conf = np.array([[0.3, 0.6], [0.8, 0.1]], dtype=np.float32)
        mask = filt.compute_mask(conf)

        assert mask[0, 0] == False
        assert mask[0, 1] == True
        assert mask[1, 0] == True
        assert mask[1, 1] == False

    def test_all_above_threshold(self):
        filt = ConfidenceFilter(threshold=0.5, use_morphology=False)
        conf = np.ones((10, 10), dtype=np.float32)
        mask = filt.compute_mask(conf)
        assert mask.all()

    def test_all_below_threshold(self):
        filt = ConfidenceFilter(threshold=0.5, use_morphology=False)
        conf = np.zeros((10, 10), dtype=np.float32)
        mask = filt.compute_mask(conf)
        assert not mask.any()

    def test_morphology_removes_thin_edges(self):
        filt = ConfidenceFilter(threshold=0.5, use_morphology=True, morph_kernel_size=3)
        # Create a confidence map with a thin 1-pixel high-confidence line
        conf = np.zeros((20, 20), dtype=np.float32)
        conf[10, :] = 1.0  # Thin horizontal line
        mask = filt.compute_mask(conf)
        # After erosion, the thin line should be removed
        assert not mask[10, :].all()

    def test_filter_points(self):
        filt = ConfidenceFilter(threshold=0.5)
        points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.float32)
        conf = np.array([0.2, 0.8, 0.6], dtype=np.float32)

        pts, cols, cfs = filt.filter_points(points, colors, conf)
        assert pts.shape[0] == 2  # Only indices 1 and 2 pass
        np.testing.assert_array_equal(pts[0], [4, 5, 6])
        np.testing.assert_array_equal(pts[1], [7, 8, 9])

    def test_filter_points_spatial_input(self):
        """Test with (H, W, 3) shaped inputs."""
        filt = ConfidenceFilter(threshold=0.5)
        points = np.random.rand(10, 10, 3).astype(np.float32)
        colors = np.random.randint(0, 255, (10, 10, 3)).astype(np.float32)
        conf = np.random.rand(10, 10).astype(np.float32)

        pts, cols, cfs = filt.filter_points(points, colors, conf)
        n_expected = (conf >= 0.5).sum()
        assert pts.shape[0] == n_expected

    def test_filter_gaussians(self):
        filt = ConfidenceFilter(threshold=0.7)
        N = 100
        means = np.random.rand(N, 3).astype(np.float32)
        scales = np.random.rand(N, 3).astype(np.float32)
        rotations = np.random.rand(N, 4).astype(np.float32)
        colors = np.random.rand(N, 3).astype(np.float32)
        opacities = np.random.rand(N).astype(np.float32)
        confidence = np.random.rand(N).astype(np.float32)

        m, s, r, c, o = filt.filter_gaussians(
            means, scales, rotations, colors, opacities, confidence
        )
        n_expected = (confidence >= 0.7).sum()
        assert m.shape[0] == n_expected
        assert s.shape[0] == n_expected
        assert r.shape[0] == n_expected
        assert c.shape[0] == n_expected
        assert o.shape[0] == n_expected
