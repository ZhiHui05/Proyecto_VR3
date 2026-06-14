"""
Rule-based gesture classifier for virtual drone control.

Classifies hand positions into drone control gestures:
TAKEOFF, LAND, MOVE_FORWARD, MOVE_BACKWARD, MOVE_LEFT, MOVE_RIGHT,
MOVE_UP, MOVE_DOWN, ROTATE_LEFT, ROTATE_RIGHT, HOVER, EMERGENCY_STOP.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.tracking.hand_tracker import HandData, TrackingResult
from src.utils.logger import get_logger
from src.utils.config import get_config


class Gesture(Enum):
    """Recognized gesture labels."""

    NONE = auto()
    TAKEOFF = auto()
    LAND = auto()
    MOVE_FORWARD = auto()
    MOVE_BACKWARD = auto()
    MOVE_LEFT = auto()
    MOVE_RIGHT = auto()
    MOVE_UP = auto()
    MOVE_DOWN = auto()
    ROTATE_LEFT = auto()
    ROTATE_RIGHT = auto()
    HOVER = auto()
    EMERGENCY_STOP = auto()


@dataclass
class GestureResult:
    """Output from gesture classification."""

    gesture: Gesture = Gesture.NONE
    confidence: float = 0.0
    movement_vector: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )
    rotation_angle: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


class GestureClassifier:
    """
    Rule-based classifier that maps hand positions to drone gestures.

    Uses relative wrist positions, distances between hands,
    hand orientation, and temporal consistency.
    """

    def __init__(self) -> None:
        self._logger = get_logger()
        self._cfg = get_config()

        self._prev_wrist_positions: Dict[str, Optional[np.ndarray]] = {
            "Left": None,
            "Right": None,
        }
        self._prev_timestamp: float = 0.0

        self._gesture_history: deque = deque(maxlen=30)
        self._current_gesture: Gesture = Gesture.NONE
        self._gesture_hold_count: int = 0
        self._last_gesture_change: float = 0.0

        self._debounce_enabled = self._cfg.get("gestures.debounce.enabled", True)
        self._debounce_cooldown = self._cfg.get(
            "gestures.debounce.gesture_change_cooldown", 0.3
        )

        self._hand_positions_history: Dict[str, deque] = {
            "Left": deque(maxlen=20),
            "Right": deque(maxlen=20),
        }

    def classify(self, tracking_result: TrackingResult) -> GestureResult:
        """
        Classify hand positions into a drone control gesture.

        Args:
            tracking_result: Current hand tracking result.

        Returns:
            GestureResult with gesture label, confidence, and movement parameters.
        """
        now = time.perf_counter()
        left = tracking_result.left_hand
        right = tracking_result.right_hand

        if left is None and right is None:
            self._gesture_hold_count = 0
            return GestureResult(
                gesture=Gesture.NONE,
                confidence=0.0,
                timestamp=now,
            )

        active_gesture = self._classify_internal(left, right, now)

        self._gesture_history.append(active_gesture.gesture)

        stabilized = self._stabilize_gesture(active_gesture.gesture, now)
        active_gesture.gesture = stabilized

        if left is not None:
            self._hand_positions_history["Left"].append(left.wrist.copy())
            self._prev_wrist_positions["Left"] = left.wrist.copy()
        if right is not None:
            self._hand_positions_history["Right"].append(right.wrist.copy())
            self._prev_wrist_positions["Right"] = right.wrist.copy()
        self._prev_timestamp = now

        return active_gesture

    def _classify_internal(
        self,
        left: Optional[HandData],
        right: Optional[HandData],
        timestamp: float,
    ) -> GestureResult:
        movement = np.zeros(3)
        rotation = 0.0
        metadata: Dict[str, Any] = {}

        if left is None or right is None:
            return GestureResult(
                gesture=Gesture.HOVER,
                confidence=0.4,
                movement_vector=movement,
                timestamp=timestamp,
                metadata=metadata,
            )

        lw = left.wrist
        rw = right.wrist
        lp = left.palm_center
        rp = right.palm_center

        hand_distance = float(np.linalg.norm(lw - rw))
        mid_point = (lw + rw) / 2.0

        wrist_vertical_diff = rw[1] - lw[1]
        wrist_horizontal_diff = rw[0] - lw[0]

        wrist_roll_angle = float(np.degrees(np.arctan2(wrist_vertical_diff, abs(wrist_horizontal_diff) + 0.001)))

        left_palm_z = lp[2]
        right_palm_z = rp[2]
        avg_palm_z = (left_palm_z + right_palm_z) / 2.0

        lw_vel = np.zeros(3)
        rw_vel = np.zeros(3)
        if self._prev_wrist_positions.get("Left") is not None and self._prev_timestamp > 0:
            dt = max(timestamp - self._prev_timestamp, 0.001)
            lw_vel = (lw - self._prev_wrist_positions["Left"]) / dt
        if self._prev_wrist_positions.get("Right") is not None and self._prev_timestamp > 0:
            dt = max(timestamp - self._prev_timestamp, 0.001)
            rw_vel = (rw - self._prev_wrist_positions["Right"]) / dt
        avg_vel = (lw_vel + rw_vel) / 2.0

        lateral_ratio = abs(wrist_horizontal_diff) / max(hand_distance, 0.001)
        vertical_ratio = abs(wrist_vertical_diff) / max(hand_distance, 0.001)

        gesture = Gesture.HOVER
        confidence = 0.5

        thresholds = self._cfg.all().get("thresholds", {})

        crossover_threshold = self._cfg.get(
            "gestures.gestures.EMERGENCY_STOP.conditions.wrists_crossed_threshold", 0.1
        )
        vertical_proximity = self._cfg.get(
            "gestures.gestures.EMERGENCY_STOP.conditions.vertical_proximity", 0.05
        )
        if (
            hand_distance < crossover_threshold
            and abs(wrist_vertical_diff) < vertical_proximity
        ):
            gesture = Gesture.EMERGENCY_STOP
            confidence = 0.85

        elif avg_vel[1] < -0.15 and mid_point[1] > 0.5:
            gesture = Gesture.TAKEOFF
            confidence = 0.8
            movement = np.array([0.0, 0.0, 1.0])

        elif avg_vel[1] > 0.15 and mid_point[1] < 0.3:
            gesture = Gesture.LAND
            confidence = 0.8
            movement = np.array([0.0, 0.0, -1.0])

        elif wrist_roll_angle < -15.0 and abs(wrist_vertical_diff) > 0.15:
            gesture = Gesture.ROTATE_LEFT
            confidence = 0.75
            rotation = -1.0

        elif wrist_roll_angle > 15.0 and abs(wrist_vertical_diff) > 0.15:
            gesture = Gesture.ROTATE_RIGHT
            confidence = 0.75
            rotation = 1.0

        elif abs(avg_palm_z) < 0.08 and avg_vel[2] < -0.05:
            gesture = Gesture.MOVE_FORWARD
            confidence = 0.75
            movement = np.array([0.0, 1.0, 0.0])

        elif avg_palm_z > 0.1 and avg_vel[2] > 0.05:
            gesture = Gesture.MOVE_BACKWARD
            confidence = 0.75
            movement = np.array([0.0, -1.0, 0.0])

        elif wrist_horizontal_diff < -0.1 and lateral_ratio > 0.4:
            gesture = Gesture.MOVE_LEFT
            confidence = 0.75
            movement = np.array([-1.0, 0.0, 0.0])

        elif wrist_horizontal_diff > 0.1 and lateral_ratio > 0.4:
            gesture = Gesture.MOVE_RIGHT
            confidence = 0.75
            movement = np.array([1.0, 0.0, 0.0])

        elif wrist_vertical_diff < -0.15 and vertical_ratio > 0.6:
            gesture = Gesture.MOVE_UP
            confidence = 0.75
            movement = np.array([0.0, 0.0, 1.0])

        elif wrist_vertical_diff > 0.15 and vertical_ratio > 0.6:
            gesture = Gesture.MOVE_DOWN
            confidence = 0.75
            movement = np.array([0.0, 0.0, -1.0])

        else:
            gesture = Gesture.HOVER
            confidence = 0.6
            if np.linalg.norm(avg_vel) < 0.05:
                confidence = 0.7

        if gesture == Gesture.EMERGENCY_STOP:
            movement = np.zeros(3)
            rotation = 0.0

        metadata = {
            "hand_distance": hand_distance,
            "mid_point": mid_point.tolist(),
            "wrist_roll_angle": wrist_roll_angle,
            "avg_palm_z": avg_palm_z,
            "avg_velocity_mag": float(np.linalg.norm(avg_vel)),
        }

        return GestureResult(
            gesture=gesture,
            confidence=confidence,
            movement_vector=movement,
            rotation_angle=rotation,
            metadata=metadata,
            timestamp=timestamp,
        )

    def _stabilize_gesture(self, detected: Gesture, timestamp: float) -> Gesture:
        if len(self._gesture_history) < 3:
            return detected

        recent = list(self._gesture_history)[-8:]
        mode_gesture = max(set(recent), key=recent.count)
        mode_count = recent.count(mode_gesture)

        if mode_count >= 5 and mode_gesture != self._current_gesture:
            if self._debounce_enabled:
                elapsed = timestamp - self._last_gesture_change
                if elapsed < self._debounce_cooldown and self._current_gesture != Gesture.NONE:
                    return self._current_gesture
            self._current_gesture = mode_gesture
            self._last_gesture_change = timestamp
            self._logger.info(f"Gesture changed: {mode_gesture.name} (confidence={mode_count}/8)")

        if mode_gesture == self._current_gesture:
            self._gesture_hold_count += 1
        elif mode_gesture == Gesture.NONE:
            self._gesture_hold_count = 0
        else:
            self._gesture_hold_count = max(self._gesture_hold_count - 1, 0)

        return self._current_gesture

    def get_current_gesture(self) -> Gesture:
        return self._current_gesture

    def draw_gesture_overlay(self, frame: np.ndarray, result: GestureResult) -> np.ndarray:
        """
        Draw gesture information overlay on the frame.

        Args:
            frame: BGR image to draw on.
            result: Current gesture result.

        Returns:
            Frame with gesture overlay.
        """
        h, w = frame.shape[:2]

        gesture_text = f"GESTURE: {result.gesture.name}"
        conf_text = f"Conf: {result.confidence:.2f}"
        mv_text = f"Move: ({result.movement_vector[0]:.2f}, {result.movement_vector[1]:.2f}, {result.movement_vector[2]:.2f})"
        rot_text = f"Rot: {result.rotation_angle:.2f}"

        y_offset = h - 120

        overlay = frame.copy()
        cv2.rectangle(overlay, (5, y_offset - 5), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(
            frame, gesture_text, (15, y_offset + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )
        cv2.putText(
            frame, conf_text, (15, y_offset + 45),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )
        cv2.putText(
            frame, mv_text, (15, y_offset + 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )
        cv2.putText(
            frame, rot_text, (15, y_offset + 95),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

        return frame

    def reset(self) -> None:
        self._gesture_history.clear()
        self._current_gesture = Gesture.NONE
        self._gesture_hold_count = 0
        self._prev_wrist_positions = {"Left": None, "Right": None}
        self._hand_positions_history["Left"].clear()
        self._hand_positions_history["Right"].clear()
