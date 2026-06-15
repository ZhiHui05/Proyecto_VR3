"""
Virtual drone controller with physics simulation.

Manages drone state (position, orientation, velocity) and applies
movement commands from gesture recognition. Supports smooth
translation, rotation, hover, and emergency stop with boundary constraints.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple

import numpy as np

from src.gestures.gesture_classifier import Gesture, GestureResult
from src.utils.logger import get_logger
from src.utils.config import get_config


class DroneState(Enum):
    """Operational states of the virtual drone."""

    IDLE = auto()
    TAKING_OFF = auto()
    FLYING = auto()
    LANDING = auto()
    LANDED = auto()
    EMERGENCY = auto()


@dataclass
class DroneTelemetry:
    """Complete drone state snapshot."""

    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    orientation: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: float = 0.0
    state: DroneState = DroneState.LANDED
    current_gesture: str = "NONE"
    timestamp: float = 0.0


class DroneController:
    """
    Simulates a virtual drone with physics-based movement.

    Translates gesture commands into smooth drone motion with
    velocity-based dynamics, boundary enforcement, and state management.
    """

    def __init__(self) -> None:
        self._logger = get_logger()
        self._cfg = get_config()

        self._pos = np.array(
            [
                self._cfg.get("drone.initial_position.x", 0.0),
                self._cfg.get("drone.initial_position.y", 0.0),
                self._cfg.get("drone.initial_position.z", 3.0),
            ],
            dtype=np.float64,
        )
        self._orientation = np.array(
            [
                self._cfg.get("drone.initial_orientation.roll", 0.0),
                self._cfg.get("drone.initial_orientation.pitch", 0.0),
                self._cfg.get("drone.initial_orientation.yaw", 0.0),
            ],
            dtype=np.float64,
        )
        self._velocity = np.zeros(3, dtype=np.float64)
        self._angular_velocity = 0.0

        self._translation_speed = self._cfg.get(
            "drone.physics.translation_speed", 3.0
        )
        self._rotation_speed = self._cfg.get("drone.physics.rotation_speed", 60.0)
        self._hover_damping = self._cfg.get("drone.physics.hover_damping", 0.95)
        self._max_velocity = self._cfg.get("drone.physics.max_velocity", 5.0)
        self._acceleration = self._cfg.get("drone.physics.acceleration", 1.5)
        self._angular_accel = self._cfg.get(
            "drone.physics.angular_acceleration", 30.0
        )

        self._min_altitude = self._cfg.get("drone.constraints.min_altitude", 0.5)
        self._max_altitude = self._cfg.get("drone.constraints.max_altitude", 14.0)
        self._boundary_margin = self._cfg.get(
            "drone.constraints.boundary_margin", 0.5
        )
        self._world_bounds = self._cfg.all().get("world", {}).get("dimensions", {})

        self._state = DroneState.LANDED
        self._current_gesture = Gesture.NONE
        self._last_update: float = time.perf_counter()
        self._has_taken_off = False
        self._emergency_stop_active = False

        self._logger.info("Drone controller initialized")

    def update(self, gesture_result: GestureResult, dt: Optional[float] = None) -> DroneTelemetry:
        """
        Update drone state based on gesture command.

        Args:
            gesture_result: Classified gesture with movement/rotation data.
            dt: Time delta in seconds. Computed automatically if None.

        Returns:
            DroneTelemetry with current state snapshot.
        """
        now = time.perf_counter()
        if dt is None:
            dt = now - self._last_update
            dt = max(dt, 0.001)
            dt = min(dt, 0.1)
        self._last_update = now

        gesture = gesture_result.gesture
        if gesture != self._current_gesture:
            self._logger.info(
                f"Drone received gesture: {gesture.name} "
                f"(confidence={gesture_result.confidence:.2f})"
            )
        self._current_gesture = gesture

        if gesture == Gesture.EMERGENCY_STOP:
            if self._state != DroneState.EMERGENCY:
                self._logger.warning("EMERGENCY STOP activated")
            self._state = DroneState.EMERGENCY
            self._emergency_stop_active = True
            self._velocity = np.zeros(3)
            self._angular_velocity = 0.0
            return self._get_telemetry(now)

        if self._state == DroneState.EMERGENCY:
            if gesture == Gesture.LAND:
                self._state = DroneState.LANDING
                self._emergency_stop_active = False
                self._logger.info("Exiting emergency stop - landing")
            else:
                return self._get_telemetry(now)

        target_velocity = np.zeros(3, dtype=np.float64)
        target_angular = 0.0

        match gesture:
            case Gesture.TAKEOFF:
                if self._state in (DroneState.LANDED, DroneState.IDLE):
                    if self._state != DroneState.TAKING_OFF:
                        self._logger.info("TAKEOFF initiated")
                    self._state = DroneState.TAKING_OFF
                    target_velocity = np.array([0.0, 0.0, 1.0])
                elif self._state == DroneState.FLYING:
                    target_velocity = np.array([0.0, 0.0, 0.5])

            case Gesture.LAND:
                if self._state in (DroneState.FLYING, DroneState.TAKING_OFF):
                    if self._state != DroneState.LANDING:
                        self._logger.info("LANDING initiated")
                    self._state = DroneState.LANDING
                    target_velocity = np.array([0.0, 0.0, -0.5])
                elif self._state == DroneState.LANDING:
                    target_velocity = np.array([0.0, 0.0, -0.5])
                else:
                    target_velocity = np.zeros(3)

            case Gesture.HOVER:
                if self._state == DroneState.TAKING_OFF:
                    if self._pos[2] >= 1.0:
                        if self._state != DroneState.FLYING:
                            self._logger.info("TAKEOFF complete - now FLYING")
                        self._state = DroneState.FLYING
                        self._has_taken_off = True
                elif self._state == DroneState.LANDING:
                    if self._pos[2] <= self._min_altitude:
                        if self._state != DroneState.LANDED:
                            self._logger.info("Drone landed")
                        self._state = DroneState.LANDED
                        self._has_taken_off = False
                        self._pos[2] = 0.0
                        self._velocity = np.zeros(3)
                        return self._get_telemetry(now)
                if self._has_taken_off:
                    self._state = DroneState.FLYING
                target_velocity = np.zeros(3)
                target_angular = 0.0

            case (Gesture.MOVE_FORWARD | Gesture.MOVE_BACKWARD |
                  Gesture.MOVE_LEFT | Gesture.MOVE_RIGHT |
                  Gesture.MOVE_UP | Gesture.MOVE_DOWN):
                if self._has_taken_off and gesture_result.confidence > 0.5:
                    self._state = DroneState.FLYING
                    target_velocity = gesture_result.movement_vector.copy()

            case Gesture.ROTATE_LEFT:
                if self._has_taken_off:
                    target_angular = -1.0

            case Gesture.ROTATE_RIGHT:
                if self._has_taken_off:
                    target_angular = 1.0

            case _:
                target_velocity = np.zeros(3)
                target_angular = 0.0

        if target_velocity[2] != 0.0 and not self._has_taken_off:
            self._has_taken_off = True
            if self._state != DroneState.TAKING_OFF:
                self._state = DroneState.FLYING

        self._accelerate_velocity(target_velocity * self._translation_speed, dt)

        if target_angular != 0.0:
            self._accelerate_angular(target_angular * self._rotation_speed, dt)
        else:
            self._angular_velocity *= (1.0 - dt * 3.0)
            if abs(self._angular_velocity) < 0.01:
                self._angular_velocity = 0.0

        self._pos += self._velocity * dt
        self._orientation[2] += self._angular_velocity * dt * (np.pi / 180.0)
        self._orientation[2] = self._orientation[2] % (2.0 * np.pi)

        self._enforce_boundaries()

        if self._pos[2] <= self._min_altitude and self._state == DroneState.LANDING:
            self._pos[2] = 0.0
            self._state = DroneState.LANDED
            self._has_taken_off = False
            self._velocity = np.zeros(3)
            self._logger.info("Drone landed")

        return self._get_telemetry(now)

    def _accelerate_velocity(self, target: np.ndarray, dt: float) -> None:
        for i in range(3):
            diff = target[i] - self._velocity[i]
            step = self._acceleration * dt
            if abs(diff) <= step:
                self._velocity[i] = target[i]
            else:
                self._velocity[i] += np.sign(diff) * step

        speed = float(np.linalg.norm(self._velocity))
        if speed > self._max_velocity:
            self._velocity = self._velocity / speed * self._max_velocity

    def _accelerate_angular(self, target: float, dt: float) -> None:
        diff = target - self._angular_velocity
        step = self._angular_accel * dt
        if abs(diff) <= step:
            self._angular_velocity = target
        else:
            self._angular_velocity += np.sign(diff) * step

    def _enforce_boundaries(self) -> None:
        """Enforce world boundaries and zero velocity on collision."""
        x_min = self._world_bounds.get("x_min", -10.0)
        x_max = self._world_bounds.get("x_max", 10.0)
        y_min = self._world_bounds.get("y_min", -10.0)
        y_max = self._world_bounds.get("y_max", 10.0)

        x_bounds_low = x_min + self._boundary_margin
        x_bounds_high = x_max - self._boundary_margin
        y_bounds_low = y_min + self._boundary_margin
        y_bounds_high = y_max - self._boundary_margin

        x_old = float(self._pos[0])
        y_old = float(self._pos[1])
        z_old = float(self._pos[2])

        x_new = float(np.clip(x_old, x_bounds_low, x_bounds_high))
        y_new = float(np.clip(y_old, y_bounds_low, y_bounds_high))

        # Allow landed drone to sit at z=0; otherwise clip to altitude envelope.
        if self._state == DroneState.LANDED:
            z_new = 0.0
        else:
            z_new = float(np.clip(z_old, self._min_altitude, self._max_altitude))

        if x_new != x_old:
            self._velocity[0] = 0.0
        if y_new != y_old:
            self._velocity[1] = 0.0
        if z_new != z_old:
            self._velocity[2] = 0.0

        self._pos[0] = x_new
        self._pos[1] = y_new
        self._pos[2] = z_new

    def _get_telemetry(self, timestamp: float) -> DroneTelemetry:
        return DroneTelemetry(
            position=self._pos.copy(),
            orientation=self._orientation.copy(),
            velocity=self._velocity.copy(),
            angular_velocity=self._angular_velocity,
            state=self._state,
            current_gesture=self._current_gesture.name,
            timestamp=timestamp,
        )

    def reset(self) -> None:
        self._pos = np.array(
            [
                self._cfg.get("drone.initial_position.x", 0.0),
                self._cfg.get("drone.initial_position.y", 0.0),
                self._cfg.get("drone.initial_position.z", 3.0),
            ],
            dtype=np.float64,
        )
        self._orientation = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self._velocity = np.zeros(3, dtype=np.float64)
        self._angular_velocity = 0.0
        self._state = DroneState.LANDED
        self._has_taken_off = False
        self._emergency_stop_active = False
        self._logger.info("Drone controller reset")

    @property
    def position(self) -> np.ndarray:
        return self._pos.copy()

    @property
    def orientation(self) -> np.ndarray:
        return self._orientation.copy()

    @property
    def state(self) -> DroneState:
        return self._state

    @property
    def is_flying(self) -> bool:
        return self._has_taken_off and self._state not in (
            DroneState.LANDED,
            DroneState.EMERGENCY,
        )
