"""
Circular Frame Buffer for temporal stereo frame storage.

Stores the most recent N rectified stereo frame pairs (left, right)
and provides batched retrieval for DA3 inference:
    {L_{t-2}, L_{t-1}, L_t, R_{t-2}, R_{t-1}, R_t}
"""

import threading
from collections import deque
from typing import Optional

import numpy as np


class CircularFrameBuffer:
    """Thread-safe circular buffer for temporal stereo frame storage.

    Stores up to `capacity` most recent stereo pairs (left, right).
    Provides batched retrieval suitable for DA3 batch-6 inference.

    Args:
        capacity: Maximum number of stereo pairs to store (default: 3).
    """

    def __init__(self, capacity: int = 3):
        self.capacity = capacity
        self._buffer: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def push(self, frame_left: np.ndarray, frame_right: np.ndarray) -> None:
        """Push a new stereo pair into the buffer.

        Args:
            frame_left: Left camera frame (H, W, 3), uint8 BGR.
            frame_right: Right camera frame (H, W, 3), uint8 BGR.
        """
        with self._lock:
            self._buffer.append((frame_left.copy(), frame_right.copy()))

    def is_ready(self) -> bool:
        """Check if the buffer has enough frames for a full temporal batch."""
        with self._lock:
            return len(self._buffer) == self.capacity

    def get_temporal_batch(self) -> Optional[list[np.ndarray]]:
        """Get temporal batch: [L_{t-2}, L_{t-1}, L_t, R_{t-2}, R_{t-1}, R_t].

        Returns:
            List of 6 frames in the order above, or None if not enough frames.
            Each frame is (H, W, 3) uint8 BGR.
        """
        with self._lock:
            if len(self._buffer) < self.capacity:
                return None

            left_frames = [pair[0] for pair in self._buffer]
            right_frames = [pair[1] for pair in self._buffer]
            return left_frames + right_frames

    def get_latest_pair(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Get the most recent stereo pair.

        Returns:
            Tuple of (left_frame, right_frame), or None if buffer is empty.
        """
        with self._lock:
            if len(self._buffer) == 0:
                return None
            return self._buffer[-1]

    def clear(self) -> None:
        """Clear all stored frames."""
        with self._lock:
            self._buffer.clear()

    @property
    def count(self) -> int:
        """Current number of stereo pairs in the buffer."""
        with self._lock:
            return len(self._buffer)

    def __len__(self) -> int:
        return self.count
