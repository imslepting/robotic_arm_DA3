"""Tests for CircularFrameBuffer."""

import threading
import time

import numpy as np
import pytest

from pipeline.frame_buffer import CircularFrameBuffer


class TestCircularFrameBuffer:
    """Unit tests for the circular frame buffer."""

    def _make_frame(self, value: int = 0, h: int = 480, w: int = 640) -> np.ndarray:
        """Create a synthetic test frame."""
        return np.full((h, w, 3), value, dtype=np.uint8)

    def test_initial_state(self):
        buf = CircularFrameBuffer(capacity=3)
        assert len(buf) == 0
        assert not buf.is_ready()
        assert buf.get_temporal_batch() is None
        assert buf.get_latest_pair() is None

    def test_push_single(self):
        buf = CircularFrameBuffer(capacity=3)
        buf.push(self._make_frame(1), self._make_frame(2))
        assert len(buf) == 1
        assert not buf.is_ready()

    def test_push_until_ready(self):
        buf = CircularFrameBuffer(capacity=3)
        for i in range(3):
            buf.push(self._make_frame(i * 10), self._make_frame(i * 10 + 1))
        assert len(buf) == 3
        assert buf.is_ready()

    def test_circular_overflow(self):
        buf = CircularFrameBuffer(capacity=3)
        for i in range(5):
            buf.push(self._make_frame(i * 10), self._make_frame(i * 10 + 1))
        # Should still have exactly 3
        assert len(buf) == 3

    def test_temporal_batch_ordering(self):
        """Ensure batch is [L_{t-2}, L_{t-1}, L_t, R_{t-2}, R_{t-1}, R_t]."""
        buf = CircularFrameBuffer(capacity=3)
        for i in range(3):
            buf.push(self._make_frame(i), self._make_frame(100 + i))

        batch = buf.get_temporal_batch()
        assert batch is not None
        assert len(batch) == 6

        # First 3 = left frames (values 0, 1, 2)
        assert batch[0][0, 0, 0] == 0
        assert batch[1][0, 0, 0] == 1
        assert batch[2][0, 0, 0] == 2

        # Last 3 = right frames (values 100, 101, 102)
        assert batch[3][0, 0, 0] == 100
        assert batch[4][0, 0, 0] == 101
        assert batch[5][0, 0, 0] == 102

    def test_get_latest_pair(self):
        buf = CircularFrameBuffer(capacity=3)
        buf.push(self._make_frame(10), self._make_frame(20))
        buf.push(self._make_frame(30), self._make_frame(40))

        pair = buf.get_latest_pair()
        assert pair is not None
        assert pair[0][0, 0, 0] == 30
        assert pair[1][0, 0, 0] == 40

    def test_clear(self):
        buf = CircularFrameBuffer(capacity=3)
        for i in range(3):
            buf.push(self._make_frame(i), self._make_frame(i))
        buf.clear()
        assert len(buf) == 0
        assert not buf.is_ready()

    def test_push_copies_data(self):
        """Ensure push copies frames (not references)."""
        buf = CircularFrameBuffer(capacity=3)
        frame = self._make_frame(42)
        buf.push(frame, frame)
        frame[:] = 0  # Modify original
        pair = buf.get_latest_pair()
        assert pair[0][0, 0, 0] == 42  # Buffer's copy should be unchanged

    def test_thread_safety(self):
        """Concurrent read/write should not crash or corrupt data."""
        buf = CircularFrameBuffer(capacity=3)
        errors = []

        def writer():
            try:
                for i in range(100):
                    buf.push(self._make_frame(i % 255), self._make_frame(i % 255))
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    buf.get_temporal_batch()
                    buf.get_latest_pair()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
