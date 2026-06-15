"""
Open3D-based 3D visualization module for the virtual drone environment.

Creates a complete 3D scene with ground plane, coordinate axes, grid,
obstacles, reference markers, and a real-time updating drone model.
Disposes of the drone as a wireframe LineSet for efficient per-frame
updates and renders telemetry labels in the 3D viewport.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import open3d as o3d
    _O3D_AVAILABLE = True
except ImportError:
    _O3D_AVAILABLE = False
    o3d = None  # type: ignore

from src.drone.drone_controller import DroneTelemetry, DroneState
from src.gestures.gesture_classifier import GestureResult
from src.tracking.hand_tracker import TrackingResult
from src.utils.logger import get_logger
from src.utils.config import get_config


class Open3DVisualizer:
    """
    Manages the Open3D 3D scene and drone model visualization.

    Creates and updates the virtual environment, drone model geometries,
    and text overlays in a separate Open3D window.
    """

    def __init__(self) -> None:
        self._logger = get_logger()
        self._cfg = get_config()

        self._world_bounds = (
            self._cfg.all().get("world", {}).get("dimensions", {})
        )
        self._ground_size = self._cfg.get("world.ground_size", 20.0)

        self._vis: Optional[Any] = None
        self._window_created = False
        self._running = False

        self._drone_lines: Optional[o3d.geometry.LineSet] = None
        self._drone_colors: Optional[np.ndarray] = None

        self._scene_geometries: List[o3d.geometry.Geometry3D] = []
        self._text_labels: Dict[str, Any] = {}
        self._labels_supported = False
        self._render_failures = 0

        self._fps_counter: float = 0.0
        self._frame_count: int = 0
        self._fps_timer: float = time.perf_counter()

        self._last_telemetry: Optional[DroneTelemetry] = None
        self._last_gesture: Optional[GestureResult] = None

    def initialize(self) -> bool:
        """
        Initialize the Open3D visualizer window and create all scene geometry.

        Returns:
            True if initialization succeeded, False otherwise.
        """
        if not _O3D_AVAILABLE or o3d is None:
            self._logger.error(
                "Open3D is not installed. Install with: pip install open3d"
            )
            return False

        try:
            self._vis = o3d.visualization.Visualizer()
            title = "V3D Virtual Drone Control - 3D World"
            self._vis.create_window(window_name=title, width=1024, height=768)
            self._window_created = True

            self._build_scene()
            self._build_drone_model()

            for geom in self._scene_geometries:
                self._vis.add_geometry(geom)
            if self._drone_lines is not None:
                self._vis.add_geometry(self._drone_lines)

            self._add_telemetry_labels()

            ctr = self._vis.get_view_control()
            ctr.set_front([-0.5, -1.0, 0.3])
            ctr.set_lookat([0.0, 0.0, 5.0])
            ctr.set_up([0.0, 0.0, 1.0])
            ctr.set_zoom(0.5)

            self._running = True
            self._logger.info("Open3D visualizer initialized")
            return True
        except Exception as e:
            self._logger.error(f"Failed to initialize Open3D: {e}")
            return False

    def _build_scene(self) -> None:
        """Create ground, grid, axes, obstacles, and reference markers."""
        scene = self._scene_geometries

        ground = o3d.geometry.TriangleMesh.create_box(
            width=self._ground_size,
            height=self._ground_size,
            depth=0.05,
        )
        ground.translate(
            np.array(
                [
                    -self._ground_size / 2.0,
                    -self._ground_size / 2.0,
                    -0.05,
                ]
            )
        )
        ground.paint_uniform_color([0.3, 0.5, 0.3])
        ground.compute_vertex_normals()
        scene.append(ground)

        if self._cfg.get("world.grid_lines", True):
            grid_spacing = self._cfg.get("world.grid_spacing", 1.0)
            half = int(self._ground_size / (2.0 * grid_spacing))
            for i in range(-half, half + 1):
                pos = i * grid_spacing

                line_x = o3d.geometry.TriangleMesh.create_cylinder(
                    radius=0.02, height=self._ground_size
                )
                line_x.rotate(
                    o3d.geometry.get_rotation_matrix_from_xyz(
                        [np.pi / 2.0, 0.0, 0.0]
                    )
                )
                line_x.translate(np.array([pos, 0.0, 0.001]))
                line_x.paint_uniform_color([0.4, 0.4, 0.4])
                scene.append(line_x)

                line_y = o3d.geometry.TriangleMesh.create_cylinder(
                    radius=0.02, height=self._ground_size
                )
                line_y.rotate(
                    o3d.geometry.get_rotation_matrix_from_xyz(
                        [np.pi / 2.0, 0.0, np.pi / 2.0]
                    )
                )
                line_y.translate(np.array([0.0, pos, 0.001]))
                line_y.paint_uniform_color([0.4, 0.4, 0.4])
                scene.append(line_y)

        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0, origin=[0.0, 0.0, 0.0]
        )
        scene.append(axes)

        obstacles: List[Dict[str, Any]] = [
            {"pos": [3.0, 4.0, 1.5], "size": [0.8, 0.8, 3.0]},
            {"pos": [-4.0, 3.0, 1.0], "size": [0.6, 0.6, 2.0]},
            {"pos": [5.0, -3.0, 1.0], "size": [0.7, 0.7, 2.0]},
            {"pos": [-5.0, -4.0, 1.5], "size": [0.5, 0.5, 3.0]},
            {"pos": [2.0, -5.0, 0.75], "size": [0.6, 0.6, 1.5]},
        ]
        for obs in obstacles:
            box = o3d.geometry.TriangleMesh.create_box(
                width=obs["size"][0],
                height=obs["size"][1],
                depth=obs["size"][2],
            )
            box.translate(
                np.array(
                    [
                        obs["pos"][0] - obs["size"][0] / 2.0,
                        obs["pos"][1] - obs["size"][1] / 2.0,
                        0.0,
                    ]
                )
            )
            box.paint_uniform_color([0.8, 0.3, 0.1])
            box.compute_vertex_normals()
            scene.append(box)

        markers = [
            {"pos": [0.0, 0.0, 5.0], "radius": 0.15},
            {"pos": [3.0, 3.0, 3.0], "radius": 0.1},
            {"pos": [-3.0, 3.0, 3.0], "radius": 0.1},
            {"pos": [3.0, -3.0, 3.0], "radius": 0.1},
            {"pos": [-3.0, -3.0, 3.0], "radius": 0.1},
        ]
        for marker in markers:
            sphere = o3d.geometry.TriangleMesh.create_sphere(
                radius=marker["radius"]
            )
            sphere.translate(np.array(marker["pos"]))
            sphere.paint_uniform_color([0.0, 1.0, 1.0])
            sphere.compute_vertex_normals()
            scene.append(sphere)

    def _build_drone_model(self) -> None:
        """Build the drone as a LineSet that can be updated per frame."""
        # Body frame (octahedron-ish box)
        points = [
            # body center cross
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [-0.25, 0.0, 0.0],
            [0.0, 0.25, 0.0],
            [0.0, -0.25, 0.0],
            [0.0, 0.0, 0.2],
            [0.0, 0.0, -0.1],
            # arm ends +X
            [0.6, 0.0, 0.0],
            # arm ends -X
            [-0.6, 0.0, 0.0],
            # arm ends +Y
            [0.0, 0.6, 0.0],
            # arm ends -Y
            [0.0, -0.6, 0.0],
            # propellers +X
            [0.6, 0.15, 0.05],
            [0.6, -0.15, 0.05],
            # propellers -X
            [-0.6, 0.15, 0.05],
            [-0.6, -0.15, 0.05],
            # propellers +Y
            [0.15, 0.6, 0.05],
            [-0.15, 0.6, 0.05],
            # propellers -Y
            [0.15, -0.6, 0.05],
            [-0.15, -0.6, 0.05],
        ]

        lines = [
            [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6],
            [1, 5], [2, 5], [3, 5], [4, 5],
            [1, 7], [7, 11], [7, 12],
            [2, 8], [8, 13], [8, 14],
            [3, 9], [9, 15], [9, 16],
            [4, 10], [10, 17], [10, 18],
        ]

        self._drone_lines = o3d.geometry.LineSet()
        self._drone_lines.points = o3d.utility.Vector3dVector(
            np.array(points, dtype=np.float64)
        )
        self._drone_lines.lines = o3d.utility.Vector2iVector(
            np.array(lines, dtype=np.int32)
        )

        colors = []
        body_color = [0.2, 0.6, 1.0]
        arm_color = [0.8, 0.8, 0.8]
        prop_color = [1.0, 0.8, 0.0]
        for i, line in enumerate(lines):
            if line[1] in (7, 8, 9, 10):
                colors.append(arm_color)
            elif line[1] >= 11:
                colors.append(prop_color)
            else:
                colors.append(body_color)
        self._drone_colors = np.array(colors, dtype=np.float64)
        self._drone_lines.colors = o3d.utility.Vector3dVector(self._drone_colors)

    def _add_telemetry_labels(self) -> None:
        """Add 3D text labels for telemetry overlay."""
        if self._vis is None:
            return
        try:
            self._text_labels["pos"] = self._vis.add_3d_label(
                [0.0, 0.0, 0.0], "POS: ---"
            )
            self._text_labels["ori"] = self._vis.add_3d_label(
                [0.0, 0.0, 0.0], "ORI: ---"
            )
            self._text_labels["gesture"] = self._vis.add_3d_label(
                [0.0, 0.0, 0.0], "GESTURE: ---"
            )
            self._text_labels["fps"] = self._vis.add_3d_label(
                [0.0, 0.0, 0.0], "FPS: ---"
            )
            self._labels_supported = True
        except Exception as e:
            self._labels_supported = False
            self._text_labels.clear()
            self._logger.debug(f"3D labels not supported: {e}")

    def _drone_base_points(self) -> np.ndarray:
        """Return the canonical drone points before transformation."""
        return np.asarray(self._drone_lines.points).copy()

    def _update_drone_pose(self, telemetry: DroneTelemetry) -> None:
        """Update the drone LineSet to the current position/orientation."""
        if self._drone_lines is None:
            return

        pos = telemetry.position
        yaw = telemetry.orientation[2]
        pitch = telemetry.orientation[1]
        roll = telemetry.orientation[0]

        R_yaw = o3d.geometry.get_rotation_matrix_from_xyz(
            [roll, pitch, yaw]
        )

        base_points = self._drone_base_points()
        rotated = (R_yaw @ base_points.T).T
        transformed = rotated + pos

        self._drone_lines.points = o3d.utility.Vector3dVector(transformed)
        try:
            self._vis.update_geometry(self._drone_lines)
        except Exception:
            try:
                self._vis.remove_geometry(
                    self._drone_lines, reset_bounding_box=False
                )
                self._vis.add_geometry(
                    self._drone_lines, reset_bounding_box=False
                )
            except Exception:
                pass

    def _update_telemetry_labels(self, telemetry: DroneTelemetry) -> None:
        """Update the 3D text labels with current telemetry."""
        if not self._labels_supported or not self._text_labels or self._vis is None:
            return

        try:
            pos = telemetry.position
            gesture = "NONE"
            if self._last_gesture is not None:
                gesture = self._last_gesture.gesture.name

            label_pos = np.array(
                [pos[0] - 1.0, pos[1] - 1.0, pos[2] + 1.0]
            )

            self._text_labels["pos"].position = label_pos
            self._text_labels["pos"].text = (
                f"POS: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
            )

            self._text_labels["ori"].position = label_pos - np.array(
                [0.0, 0.0, 0.4]
            )
            self._text_labels["ori"].text = (
                f"ORI: (R{np.degrees(telemetry.orientation[0]):.1f}, "
                f"P{np.degrees(telemetry.orientation[1]):.1f}, "
                f"Y{np.degrees(telemetry.orientation[2]):.1f})"
            )

            self._text_labels["gesture"].position = label_pos - np.array(
                [0.0, 0.0, 0.8]
            )
            self._text_labels["gesture"].text = f"GESTURE: {gesture}"

            self._text_labels["fps"].position = label_pos - np.array(
                [0.0, 0.0, 1.2]
            )
            self._text_labels["fps"].text = f"FPS: {self._fps_counter:.1f}"
        except Exception:
            pass

    def update(
        self,
        telemetry: Optional[DroneTelemetry] = None,
        gesture: Optional[GestureResult] = None,
        tracking: Optional[TrackingResult] = None,
    ) -> bool:
        """
        Update the 3D visualization with new telemetry data.

        Args:
            telemetry: Current drone state snapshot.
            gesture: Current gesture classification.
            tracking: Current hand tracking result.

        Returns:
            True if the window is still open, False if user closed it.
        """
        if self._vis is None:
            return False

        self._last_gesture = gesture

        self._frame_count += 1
        now = time.perf_counter()
        if now - self._fps_timer >= 1.0:
            self._fps_counter = self._frame_count / (now - self._fps_timer)
            self._frame_count = 0
            self._fps_timer = now

        try:
            if not self._vis.poll_events():
                return False

            if telemetry is not None:
                self._last_telemetry = telemetry
                self._update_drone_pose(telemetry)
                self._update_telemetry_labels(telemetry)

            self._vis.update_renderer()
        except Exception as e:
            self._render_failures += 1
            if self._render_failures <= 3:
                self._logger.error(f"Open3D render error: {e}")
            elif self._render_failures == 11:
                self._logger.error(
                    "Open3D renderer keeps failing; stopping visualization"
                )
            if self._render_failures > 10:
                return False
        return True

    def close(self) -> None:
        """Close the Open3D window and release resources."""
        self._running = False
        if self._vis is not None:
            try:
                self._vis.destroy_window()
            except Exception:
                pass
            self._vis = None
        self._logger.info("Open3D visualizer closed")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def fps(self) -> float:
        return self._fps_counter
