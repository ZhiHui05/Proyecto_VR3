"""
Unit tests for the DroneController module.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.drone.drone_controller import DroneController, DroneState
from src.gestures.gesture_classifier import Gesture, GestureResult


class TestDroneController:
    def setup_method(self) -> None:
        self._drone = DroneController()

    def _update(self, gesture: Gesture, count: int = 10) -> None:
        result = GestureResult(gesture=gesture, confidence=0.9)
        for _ in range(count):
            self._drone.update(result)
            time.sleep(0.01)

    def test_initial_state_is_landed(self) -> None:
        assert self._drone.state == DroneState.LANDED
        assert not self._drone.is_flying

    def test_takeoff_transitions_to_flying(self) -> None:
        self._update(Gesture.TAKEOFF, count=30)
        telemetry = self._drone.update(
            GestureResult(gesture=Gesture.HOVER, confidence=0.9)
        )
        assert telemetry.state == DroneState.FLYING
        assert self._drone.is_flying
        assert telemetry.position[2] > 0.5

    def test_land_transitions_to_landed(self) -> None:
        self._update(Gesture.TAKEOFF, count=30)
        self._update(Gesture.HOVER, count=5)
        for _ in range(120):
            self._drone.update(
                GestureResult(gesture=Gesture.LAND, confidence=0.9),
                dt=0.05,
            )
        telemetry = self._drone.update(
            GestureResult(gesture=Gesture.HOVER, confidence=0.9),
            dt=0.05,
        )
        assert telemetry.state == DroneState.LANDED
        assert not self._drone.is_flying
        assert telemetry.position[2] == 0.0

    def test_emergency_stop_halts_movement(self) -> None:
        self._update(Gesture.TAKEOFF, count=30)
        self._update(Gesture.MOVE_FORWARD, count=10)
        self._update(Gesture.EMERGENCY_STOP, count=5)
        telemetry = self._drone.update(
            GestureResult(gesture=Gesture.EMERGENCY_STOP, confidence=0.9)
        )
        assert telemetry.state == DroneState.EMERGENCY
        assert np.linalg.norm(telemetry.velocity) < 0.01

    def test_boundary_constraints_limit_position(self) -> None:
        self._update(Gesture.TAKEOFF, count=30)
        for _ in range(200):
            self._drone.update(
                GestureResult(gesture=Gesture.MOVE_RIGHT, confidence=0.9)
            )
            time.sleep(0.001)
        telemetry = self._drone.update(
            GestureResult(gesture=Gesture.HOVER, confidence=0.9)
        )
        x_max = self._drone._world_bounds.get("x_max", 10.0)
        margin = self._drone._boundary_margin
        assert telemetry.position[0] <= x_max - margin + 0.1

    def test_reset_restores_initial_state(self) -> None:
        self._update(Gesture.TAKEOFF, count=30)
        self._drone.reset()
        assert self._drone.state == DroneState.LANDED
        assert not self._drone.is_flying
        assert np.allclose(
            self._drone.position,
            [
                self._drone._cfg.get("drone.initial_position.x", 0.0),
                self._drone._cfg.get("drone.initial_position.y", 0.0),
                self._drone._cfg.get("drone.initial_position.z", 3.0),
            ],
        )

    def test_rotation_changes_yaw(self) -> None:
        self._update(Gesture.TAKEOFF, count=30)
        initial_yaw = self._drone.orientation[2]
        self._update(Gesture.ROTATE_RIGHT, count=20)
        final_yaw = self._drone.orientation[2]
        assert final_yaw != initial_yaw


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
