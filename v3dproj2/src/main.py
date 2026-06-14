#!/usr/bin/env python3
"""
V3D Virtual Drone Control System - Main Entry Point

Launches the 3D hand-gesture-controlled virtual drone system using a
five-thread pipeline (camera, tracking, gesture, drone, visualization).
Uses MediaPipe for hand tracking, a rule-based gesture classifier for
drone commands, and Open3D for real-time 3D visualization.

Usage:
    python -m src.main
    python -m src.main --camera 0 --debug
    python -m src.main --config configs/ --headless
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from typing import Optional

import cv2
import numpy as np

from src.threading.thread_manager import ThreadManager
from src.tracking.hand_tracker import TrackingResult
from src.gestures.gesture_classifier import Gesture, GestureResult
from src.drone.drone_controller import DroneTelemetry, DroneState
from src.utils.config import ConfigLoader
from src.utils.logger import get_logger


class VirtualDroneApplication:
    """
    Main application orchestrating all modules of the V3D system.

    Uses ThreadManager to run the camera, tracking, gesture, drone, and
    visualization threads in parallel. The main thread displays the live
    camera feed with overlays and handles keyboard controls.
    """

    def __init__(
        self,
        config_dir: str = "configs",
        camera_index: int = 0,
        headless: bool = False,
    ) -> None:
        self._logger = get_logger()
        self._cfg = ConfigLoader(config_dir)
        self._headless = headless

        self._camera_index = (
            camera_index
            if camera_index is not None
            else self._cfg.get("camera.index", 0)
        )
        self._camera_width = self._cfg.get("camera.width", 1280)
        self._camera_height = self._cfg.get("camera.height", 720)
        self._camera_fps = self._cfg.get("camera.fps", 30)

        self._thread_manager = ThreadManager()
        if not self._thread_manager.initialize(
            camera_index=self._camera_index,
            camera_width=self._camera_width,
            camera_height=self._camera_height,
            camera_fps=self._camera_fps,
        ):
            self._logger.critical("Failed to initialize application components")
            sys.exit(1)

        self._running = False
        self._paused = False
        self._show_debug = self._cfg.get("application.debug", True)
        self._camera_frame_label = "V3D Drone Control - Camera Feed"

        self._camera_fps = 0.0
        self._vis_fps = 0.0
        self._loop_count = 0
        self._loop_timer = time.perf_counter()

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self._logger.info("VirtualDroneApplication initialized")

    def _signal_handler(self, signum: int, frame) -> None:
        self._logger.info(f"Received signal {signum}, shutting down...")
        self._running = False

    def _print_help(self) -> None:
        self._logger.info(
            "Controls: q/ESC=quit, SPACE=pause, d=debug, r=reset, "
            "t=takeoff, l=land, e=emergency, h=help"
        )

    def run(self) -> None:
        self._logger.info("=== V3D Virtual Drone Control System Starting ===")
        self._print_help()

        self._thread_manager.start()
        if not self._thread_manager.is_running:
            self._logger.critical("ThreadManager failed to start")
            self._shutdown()
            return

        camera = self._thread_manager.camera
        if camera is None:
            self._logger.critical("Camera not available")
            self._shutdown()
            return

        self._running = True

        try:
            while self._running:
                self._loop_count += 1
                now = time.perf_counter()
                if now - self._loop_timer >= 1.0:
                    self._camera_fps = self._loop_count / (now - self._loop_timer)
                    self._loop_count = 0
                    self._loop_timer = now

                if self._thread_manager.state.has_errors():
                    errors = self._thread_manager.state.get_errors()
                    if errors:
                        self._logger.warning(
                            f"Component errors: {errors[-3:]}"
                        )

                frame = self._thread_manager.get_latest_frame()
                tracking = self._thread_manager.get_latest_tracking()
                gesture = self._thread_manager.get_latest_gesture()
                telemetry = self._thread_manager.get_latest_telemetry()

                if frame is None:
                    time.sleep(0.005)
                    continue

                display_frame = frame.copy()

                if self._paused:
                    cv2.putText(
                        display_frame,
                        "PAUSED",
                        (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        3,
                    )
                elif self._show_debug:
                    if tracking is not None:
                        display_frame = self._draw_tracking_overlay(
                            display_frame, tracking
                        )
                    display_frame = self._draw_gesture_overlay(
                        display_frame, gesture
                    )
                    display_frame = self._draw_telemetry_overlay(
                        display_frame, telemetry
                    )
                    display_frame = self._draw_fps_overlay(display_frame)

                resized = cv2.resize(display_frame, (960, 540))
                if not self._headless:
                    cv2.imshow(self._camera_frame_label, resized)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    self._running = False
                elif key == ord(" "):
                    self._paused = not self._paused
                    self._logger.info(
                        f"Pipeline {'paused' if self._paused else 'resumed'}"
                    )
                elif key == ord("d"):
                    self._show_debug = not self._show_debug
                    self._logger.info(
                        f"Debug overlay: {'ON' if self._show_debug else 'OFF'}"
                    )
                elif key == ord("r"):
                    self._reset_system()
                elif key == ord("t"):
                    self._inject_gesture(Gesture.TAKEOFF)
                elif key == ord("l"):
                    self._inject_gesture(Gesture.LAND)
                elif key == ord("e"):
                    self._inject_gesture(Gesture.EMERGENCY_STOP)
                elif key == ord("h"):
                    self._print_help()

                if not self._headless:
                    try:
                        visible = cv2.getWindowProperty(
                            self._camera_frame_label, cv2.WND_PROP_VISIBLE
                        )
                        if visible < 1:
                            self._logger.info("Camera window closed by user")
                            self._running = False
                    except cv2.error:
                        pass

        except KeyboardInterrupt:
            self._logger.info("Keyboard interrupt received")
        except Exception as e:
            self._logger.exception(f"Runtime error: {e}")
        finally:
            self._shutdown()
            cv2.destroyAllWindows()

    def _inject_gesture(self, gesture: Gesture) -> None:
        """Inject a manual gesture command into the pipeline."""
        self._logger.info(f"Manual gesture injected: {gesture.name}")
        try:
            self._thread_manager.state.gesture_queue.put_nowait(
                GestureResult(gesture=gesture, confidence=0.9)
            )
        except Exception as e:
            self._logger.warning(f"Failed to inject gesture: {e}")

    def _reset_system(self) -> None:
        """Reset drone and classifier state."""
        drone = self._thread_manager.drone_controller
        if drone is not None:
            drone.reset()
        self._logger.info("System reset")

    def _shutdown(self) -> None:
        self._running = False
        self._logger.info("Shutting down...")
        self._thread_manager.stop()
        self._logger.info("Application shutdown complete")

    @staticmethod
    def _draw_tracking_overlay(
        frame: np.ndarray, tracking: TrackingResult
    ) -> np.ndarray:
        """Draw hand landmarks and tracking quality on the frame."""
        h, w = frame.shape[:2]
        y_offset = 30
        cv2.putText(
            frame,
            f"Tracking Quality: {tracking.tracking_quality:.2f}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        y_offset += 30

        for hand in (tracking.left_hand, tracking.right_hand):
            if hand is None:
                continue
            color = (255, 0, 0) if hand.handedness == "Left" else (0, 0, 255)
            for lm in hand.landmarks_3d:
                px = int(lm[0] * w)
                py = int(lm[1] * h)
                cv2.circle(frame, (px, py), 4, color, -1)

            wrist_x = int(hand.wrist[0] * w)
            wrist_y = int(hand.wrist[1] * h)
            cv2.circle(frame, (wrist_x, wrist_y), 8, (0, 255, 255), 3)
            cv2.putText(
                frame,
                f"{hand.handedness} conf={hand.confidence:.2f}",
                (wrist_x + 10, wrist_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        return frame

    @staticmethod
    def _draw_gesture_overlay(
        frame: np.ndarray, gesture: Optional[GestureResult]
    ) -> np.ndarray:
        """Draw gesture label and confidence on the frame."""
        h, w = frame.shape[:2]
        y_offset = h - 120

        overlay = frame.copy()
        cv2.rectangle(
            overlay, (5, y_offset - 5), (w - 10, h - 10), (0, 0, 0), -1
        )
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        gesture_name = gesture.gesture.name if gesture else "NONE"
        confidence = gesture.confidence if gesture else 0.0
        movement = (
            gesture.movement_vector if gesture else np.zeros(3)
        )
        rotation = gesture.rotation_angle if gesture else 0.0

        cv2.putText(
            frame,
            f"GESTURE: {gesture_name}",
            (15, y_offset + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Conf: {confidence:.2f}",
            (15, y_offset + 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            frame,
            f"Move: ({movement[0]:.2f}, {movement[1]:.2f}, {movement[2]:.2f})",
            (15, y_offset + 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            frame,
            f"Rot: {rotation:.2f}",
            (15, y_offset + 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        return frame

    @staticmethod
    def _draw_telemetry_overlay(
        frame: np.ndarray, telemetry: Optional[DroneTelemetry]
    ) -> np.ndarray:
        """Draw drone telemetry panel on the frame."""
        x_offset = frame.shape[1] - 290
        y_offset = 10

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x_offset - 5, 0),
            (frame.shape[1], 180),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(
            frame,
            "DRONE TELEMETRY",
            (x_offset, y_offset + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )
        y_offset += 25

        if telemetry is None:
            cv2.putText(
                frame,
                "Waiting for telemetry...",
                (x_offset, y_offset + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )
            return frame

        p = telemetry.position
        cv2.putText(
            frame,
            f"Pos: ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})",
            (x_offset, y_offset + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
        y_offset += 20

        o = telemetry.orientation
        cv2.putText(
            frame,
            f"Ori: ({np.degrees(o[0]):.1f}, {np.degrees(o[1]):.1f}, {np.degrees(o[2]):.1f})",
            (x_offset, y_offset + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
        y_offset += 20

        v = telemetry.velocity
        cv2.putText(
            frame,
            f"Vel: ({v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f})",
            (x_offset, y_offset + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
        y_offset += 20

        cv2.putText(
            frame,
            f"State: {telemetry.state.name}",
            (x_offset, y_offset + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
        )
        y_offset += 20

        cv2.putText(
            frame,
            f"Gesture: {telemetry.current_gesture}",
            (x_offset, y_offset + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
        )

        return frame

    def _draw_fps_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw camera and visualization FPS on the frame."""
        vis = self._thread_manager.visualizer
        vis_fps = vis.fps if vis is not None else 0.0

        cv2.putText(
            frame,
            f"Cam FPS: {self._camera_fps:.1f}",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
        )
        if vis_fps > 0:
            cv2.putText(
                frame,
                f"Vis FPS: {vis_fps:.1f}",
                (10, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
            )
        return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V3D Virtual Drone Control System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main
  python -m src.main --camera 0 --debug
  python -m src.main --config configs/ --headless
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs",
        help="Path to configuration directory (default: configs/)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without camera preview window",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug overlay by default",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default=None,
        help="Path to calibration file for undistortion",
    )
    args = parser.parse_args()

    app = VirtualDroneApplication(
        config_dir=args.config,
        camera_index=args.camera,
        headless=args.headless,
    )
    app._show_debug = args.debug or app._show_debug
    app.run()


if __name__ == "__main__":
    main()
