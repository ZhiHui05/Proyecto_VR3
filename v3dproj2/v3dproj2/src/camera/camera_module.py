"""
Camera acquisition module with threaded frame capture.

Captures frames from a webcam in a dedicated thread and places them
into a thread-safe queue for downstream processing.
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger


class CameraModule:
    """
    Manages webcam capture in a background thread.

    Attributes:
        width: Frame width.
        height: Frame height.
        fps: Target capture framerate.
        running: Whether the capture loop is active.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        queue_size: int = 8,
    ) -> None:
        self._logger = get_logger()
        self._camera_index = camera_index
        self._width = width
        self._height = height
        self._target_fps = fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._queue: list[Tuple[np.ndarray, float]] = []
        self._max_queue_size = queue_size
        self._frame_count = 0
        self._start_time = 0.0
        self._actual_fps = 0.0
        self._fps_log_timer = 0.0

    def start(self) -> bool:
        self._cap = cv2.VideoCapture(self._camera_index)
        if not self._cap.isOpened():
            self._logger.error(f"Failed to open camera index {self._camera_index}")
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._target_fps)

        actual_width = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self._width = int(actual_width)
        self._height = int(actual_height)

        self._logger.info(
            f"Camera opened: index={self._camera_index}, "
            f"resolution={self._width}x{self._height}"
        )

        self._running = True
        self._start_time = time.perf_counter()
        self._frame_count = 0
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def _capture_loop(self) -> None:
        while self._running:
            try:
                if self._cap is None or not self._cap.isOpened():
                    self._logger.warning("Camera disconnected")
                    time.sleep(0.1)
                    continue

                ret, frame = self._cap.read()
                if not ret:
                    self._logger.warning("Failed to read frame from camera")
                    time.sleep(0.01)
                    continue

                frame = cv2.flip(frame, 1).copy()

                self._frame_count += 1
                timestamp = time.perf_counter()

                elapsed = timestamp - self._start_time
                if elapsed >= 1.0:
                    self._actual_fps = self._frame_count / elapsed
                    self._frame_count = 0
                    self._start_time = timestamp
                    if timestamp - self._fps_log_timer >= 5.0:
                        self._logger.info(
                            f"Camera capture FPS: {self._actual_fps:.1f}"
                        )
                        self._fps_log_timer = timestamp

                with self._lock:
                    if len(self._queue) >= self._max_queue_size:
                        self._queue.pop(0)
                    self._queue.append((frame, timestamp))

                # No explicit sleep: drain the camera buffer as fast as possible
                # to keep latency low; get_latest_frame() always returns the newest.
            except Exception as e:
                self._logger.error(f"Camera capture error: {e}")
                time.sleep(0.05)

    def get_latest_frame(self) -> Optional[Tuple[np.ndarray, float]]:
        with self._lock:
            if self._queue:
                return self._queue[-1]
            return None

    def get_fps(self) -> float:
        return self._actual_fps

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
        self._logger.info("Camera module stopped")

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def is_running(self) -> bool:
        return self._running
