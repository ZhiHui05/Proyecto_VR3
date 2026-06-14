"""
Thread management module for parallel execution of the V3D pipeline.

Coordinates five threads:
  1. Camera acquisition  - polls CameraModule and feeds frame_queue
  2. MediaPipe processing - hand landmark extraction
  3. Gesture recognition  - rule-based gesture classification
  4. Drone simulation     - physics-based drone state update
  5. Visualization        - Open3D render loop

Uses thread-safe queues, locks, and a shared stop event for data exchange
and graceful shutdown.
"""

from __future__ import annotations

import threading
import time
from queue import Empty, Queue
from typing import Any, Optional, Tuple

import numpy as np

from src.camera.camera_module import CameraModule
from src.tracking.hand_tracker import HandTracker, TrackingResult
from src.gestures.gesture_classifier import GestureClassifier, GestureResult
from src.drone.drone_controller import DroneController, DroneTelemetry
from src.utils.logger import get_logger

try:
    from src.visualization.open3d_visualizer import Open3DVisualizer
    _O3D_IMPORT_OK = True
except ImportError:
    Open3DVisualizer = None  # type: ignore
    _O3D_IMPORT_OK = False


class SharedState:
    """Thread-safe shared state container for all pipeline data."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.stop_event = threading.Event()

        self.frame_queue: Queue[Tuple[np.ndarray, float]] = Queue(maxsize=4)
        self.tracking_queue: Queue[TrackingResult] = Queue(maxsize=4)
        self.gesture_queue: Queue[GestureResult] = Queue(maxsize=4)
        self.telemetry_queue: Queue[DroneTelemetry] = Queue(maxsize=4)

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_tracking: Optional[TrackingResult] = None
        self._latest_gesture: Optional[GestureResult] = None
        self._latest_telemetry: Optional[DroneTelemetry] = None
        self._component_errors: list[str] = []

    def report_error(self, component: str, error: str) -> None:
        with self._lock:
            self._component_errors.append(f"[{component}] {error}")

    def has_errors(self) -> bool:
        with self._lock:
            return len(self._component_errors) > 0

    def get_errors(self) -> list[str]:
        with self._lock:
            return list(self._component_errors)

    def set_latest_frame(self, frame: Optional[np.ndarray]) -> None:
        with self._lock:
            self._latest_frame = frame

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._latest_frame

    def set_latest_tracking(self, tracking: Optional[TrackingResult]) -> None:
        with self._lock:
            self._latest_tracking = tracking

    def get_latest_tracking(self) -> Optional[TrackingResult]:
        with self._lock:
            return self._latest_tracking

    def set_latest_gesture(self, gesture: Optional[GestureResult]) -> None:
        with self._lock:
            self._latest_gesture = gesture

    def get_latest_gesture(self) -> Optional[GestureResult]:
        with self._lock:
            return self._latest_gesture

    def set_latest_telemetry(self, telemetry: Optional[DroneTelemetry]) -> None:
        with self._lock:
            self._latest_telemetry = telemetry

    def get_latest_telemetry(self) -> Optional[DroneTelemetry]:
        with self._lock:
            return self._latest_telemetry


class ThreadManager:
    """
    Manages the lifecycle of all processing threads.

    Creates and coordinates the five pipeline threads:
    camera, tracking, gesture, drone, and visualization.
    """

    def __init__(self) -> None:
        self._logger = get_logger()
        self._state = SharedState()

        self._camera: Optional[CameraModule] = None
        self._tracker: Optional[HandTracker] = None
        self._classifier: Optional[GestureClassifier] = None
        self._drone: Optional[DroneController] = None
        self._visualizer: Optional[Any] = None

        self._threads: list[Tuple[str, threading.Thread]] = []

    def initialize(
        self,
        camera_index: int = 0,
        camera_width: int = 1280,
        camera_height: int = 720,
        camera_fps: int = 30,
    ) -> bool:
        """Initialize all pipeline components."""
        try:
            self._camera = CameraModule(
                camera_index=camera_index,
                width=camera_width,
                height=camera_height,
                fps=camera_fps,
            )
            self._tracker = HandTracker()
            self._classifier = GestureClassifier()
            self._drone = DroneController()

            if _O3D_IMPORT_OK:
                self._visualizer = Open3DVisualizer()
                if not self._visualizer.initialize():
                    self._logger.error(
                        "Open3D visualizer initialization failed"
                    )
                    return False
            else:
                self._logger.warning(
                    "Open3D not available - 3D visualization disabled. "
                    "Install with: pip install open3d"
                )

            self._logger.info("All components initialized")
            return True
        except Exception as e:
            self._logger.exception(f"Initialization failed: {e}")
            return False

    def start(self) -> None:
        """Start all five processing threads."""
        self._state.stop_event.clear()

        self._start_camera_thread()
        self._start_tracking_thread()
        self._start_gesture_thread()
        self._start_drone_thread()
        self._start_visualization_thread()

        self._logger.info(
            f"Started {len(self._threads)} processing threads"
        )

    def _start_camera_thread(self) -> None:
        """Thread 1: polls CameraModule and pushes frames into frame_queue."""
        if self._camera is None:
            self._state.report_error("camera", "Camera module not initialized")
            return
        if not self._camera.start():
            self._state.report_error("camera", "Failed to start camera")
            return

        def camera_loop() -> None:
            camera = self._camera
            if camera is None:
                return
            while not self._state.stop_event.is_set():
                try:
                    data = camera.get_latest_frame()
                    if data is None:
                        time.sleep(0.005)
                        continue
                    frame, timestamp = data
                    try:
                        self._state.frame_queue.put_nowait((frame, timestamp))
                    except Exception:
                        pass
                    self._state.set_latest_frame(frame)
                except Exception as e:
                    self._logger.error(f"Camera thread error: {e}")
                    self._state.report_error("camera", str(e))
                    time.sleep(0.05)

        t = threading.Thread(
            target=camera_loop, daemon=True, name="CameraAcquisition"
        )
        t.start()
        self._threads.append(("camera", t))
        self._logger.info("Camera acquisition thread started")

    def _start_tracking_thread(self) -> None:
        """Thread 2: MediaPipe hand tracking."""
        if self._tracker is None:
            self._state.report_error("tracking", "Tracker not initialized")
            return

        def tracking_loop() -> None:
            tracker = self._tracker
            while not self._state.stop_event.is_set():
                try:
                    frame, _timestamp = self._state.frame_queue.get(timeout=0.05)
                    result = tracker.process_frame(frame)
                    try:
                        self._state.tracking_queue.put_nowait(result)
                    except Exception:
                        pass
                    self._state.set_latest_tracking(result)
                except Empty:
                    continue
                except Exception as e:
                    self._logger.error(f"Tracking thread error: {e}")
                    self._state.report_error("tracking", str(e))

        t = threading.Thread(
            target=tracking_loop, daemon=True, name="HandTracking"
        )
        t.start()
        self._threads.append(("tracking", t))
        self._logger.info("Hand tracking thread started")

    def _start_gesture_thread(self) -> None:
        """Thread 3: gesture classification."""
        if self._classifier is None:
            self._state.report_error("gesture", "Classifier not initialized")
            return

        def gesture_loop() -> None:
            classifier = self._classifier
            while not self._state.stop_event.is_set():
                try:
                    tracking_result = self._state.tracking_queue.get(
                        timeout=0.05
                    )
                    gesture_result = classifier.classify(tracking_result)
                    try:
                        self._state.gesture_queue.put_nowait(gesture_result)
                    except Exception:
                        pass
                    self._state.set_latest_gesture(gesture_result)
                except Empty:
                    continue
                except Exception as e:
                    self._logger.error(f"Gesture thread error: {e}")
                    self._state.report_error("gesture", str(e))

        t = threading.Thread(
            target=gesture_loop, daemon=True, name="GestureRecognition"
        )
        t.start()
        self._threads.append(("gesture", t))
        self._logger.info("Gesture recognition thread started")

    def _start_drone_thread(self) -> None:
        """Thread 4: drone physics simulation."""
        if self._drone is None:
            self._state.report_error("drone", "Drone controller not initialized")
            return

        def drone_loop() -> None:
            drone = self._drone
            while not self._state.stop_event.is_set():
                try:
                    gesture_result = self._state.gesture_queue.get(
                        timeout=0.05
                    )
                    telemetry = drone.update(gesture_result)
                    try:
                        self._state.telemetry_queue.put_nowait(telemetry)
                    except Exception:
                        pass
                    self._state.set_latest_telemetry(telemetry)
                except Empty:
                    continue
                except Exception as e:
                    self._logger.error(f"Drone thread error: {e}")
                    self._state.report_error("drone", str(e))

        t = threading.Thread(
            target=drone_loop, daemon=True, name="DroneSimulation"
        )
        t.start()
        self._threads.append(("drone", t))
        self._logger.info("Drone simulation thread started")

    def _start_visualization_thread(self) -> None:
        """Thread 5: Open3D render loop."""
        if self._visualizer is None:
            self._logger.warning(
                "Visualization thread skipped - Open3D not available"
            )
            return

        def visualization_loop() -> None:
            vis = self._visualizer
            refresh_interval = 1.0 / 60.0
            while not self._state.stop_event.is_set() and vis.is_running:
                try:
                    telemetry = self._state.get_latest_telemetry()
                    gesture = self._state.get_latest_gesture()
                    tracking = self._state.get_latest_tracking()

                    ok = vis.update(
                        telemetry=telemetry,
                        gesture=gesture,
                        tracking=tracking,
                    )
                    if not ok:
                        self._logger.info("Visualization window closed")
                        self._state.stop_event.set()
                        break
                    time.sleep(refresh_interval)
                except Exception as e:
                    self._logger.error(f"Visualization thread error: {e}")
                    self._state.report_error("visualization", str(e))
                    break

        t = threading.Thread(
            target=visualization_loop, daemon=True, name="Visualization"
        )
        t.start()
        self._threads.append(("visualization", t))
        self._logger.info("Visualization thread started")

    def get_latest_frame(self) -> Optional[np.ndarray]:
        return self._state.get_latest_frame()

    def get_latest_tracking(self) -> Optional[TrackingResult]:
        return self._state.get_latest_tracking()

    def get_latest_gesture(self) -> Optional[GestureResult]:
        return self._state.get_latest_gesture()

    def get_latest_telemetry(self) -> Optional[DroneTelemetry]:
        return self._state.get_latest_telemetry()

    def stop(self, timeout: float = 3.0) -> None:
        """Gracefully stop all threads and release resources."""
        self._logger.info("Initiating shutdown...")
        self._state.stop_event.set()

        for name, thread in self._threads:
            thread.join(timeout=timeout)
            if thread.is_alive():
                self._logger.warning(f"Thread '{name}' did not stop in time")

        if self._camera:
            self._camera.stop()
        if self._tracker:
            self._tracker.release()
        if self._visualizer:
            self._visualizer.close()

        self._clear_queues()
        self._logger.info("All threads stopped")

    def _clear_queues(self) -> None:
        for q in (
            self._state.frame_queue,
            self._state.tracking_queue,
            self._state.gesture_queue,
            self._state.telemetry_queue,
        ):
            while not q.empty():
                try:
                    q.get_nowait()
                except Empty:
                    break

    @property
    def is_running(self) -> bool:
        return not self._state.stop_event.is_set()

    @property
    def state(self) -> SharedState:
        return self._state

    @property
    def camera(self) -> Optional[CameraModule]:
        return self._camera

    @property
    def visualizer(self) -> Optional[Any]:
        return self._visualizer

    @property
    def drone_controller(self) -> Optional[DroneController]:
        return self._drone
