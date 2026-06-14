"""
Hand tracking module using MediaPipe Hands.

Tracks hand landmarks, extracts wrist and palm positions, and applies
temporal smoothing to reduce jitter.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import mediapipe as mp
import numpy as np

from src.utils.logger import get_logger
from src.utils.config import get_config


@dataclass
class HandData:
    """Container for tracked hand information."""

    handedness: str = ""
    confidence: float = 0.0
    wrist: np.ndarray = field(default_factory=lambda: np.zeros(3))
    palm_center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    index_tip: np.ndarray = field(default_factory=lambda: np.zeros(3))
    middle_tip: np.ndarray = field(default_factory=lambda: np.zeros(3))
    thumb_tip: np.ndarray = field(default_factory=lambda: np.zeros(3))
    pinky_tip: np.ndarray = field(default_factory=lambda: np.zeros(3))
    landmarks_3d: np.ndarray = field(default_factory=lambda: np.zeros((21, 3)))
    last_seen: float = 0.0
    palm_direction: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class TrackingResult:
    """Combined tracking result for both hands."""

    left_hand: Optional[HandData] = None
    right_hand: Optional[HandData] = None
    timestamp: float = 0.0
    tracking_quality: float = 0.0


class HandTracker:
    """
    Tracks hands using MediaPipe Hands with temporal smoothing.

    Extracts wrist, palm, and fingertip positions for both hands.
    Applies exponential moving average to smooth landmark positions.
    """

    WRIST_IDX = 0
    INDEX_MCP_IDX = 5
    MIDDLE_MCP_IDX = 9
    PINKY_MCP_IDX = 17
    INDEX_TIP_IDX = 8
    MIDDLE_TIP_IDX = 12
    THUMB_TIP_IDX = 4
    PINKY_TIP_IDX = 20

    def __init__(self) -> None:
        self._logger = get_logger()
        self._cfg = get_config()

        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles

        model_complexity = self._cfg.get("mediapipe.model_complexity", 1)
        min_detection_conf = self._cfg.get(
            "mediapipe.min_detection_confidence", 0.7
        )
        min_tracking_conf = self._cfg.get(
            "mediapipe.min_tracking_confidence", 0.5
        )
        max_hands = self._cfg.get("mediapipe.max_num_hands", 2)

        self._hands = self._mp_hands.Hands(
            static_image_mode=self._cfg.get("mediapipe.static_image_mode", False),
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf,
            max_num_hands=max_hands,
        )

        self._smoothing_enabled = self._cfg.get("tracking.smoothing.enabled", True)
        self._smoothing_alpha = self._cfg.get("tracking.smoothing.alpha", 0.3)
        self._loss_timeout = self._cfg.get("tracking.loss_timeout", 1.0)

        self._smooth_left: Optional[np.ndarray] = None
        self._smooth_right: Optional[np.ndarray] = None
        self._left_hand_lost: float = 0.0
        self._right_hand_lost: float = 0.0
        self._processed_frames = 0

        self._position_history: Dict[str, deque] = {
            "Left": deque(maxlen=10),
            "Right": deque(maxlen=10),
        }

    def process_frame(self, frame: np.ndarray) -> TrackingResult:
        """
        Process a BGR frame and return hand tracking results.

        Args:
            frame: BGR image array.

        Returns:
            TrackingResult with left/right hand data if detected.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)
        rgb.flags.writeable = True

        self._processed_frames += 1
        now = time.perf_counter()

        left_hand: Optional[HandData] = None
        right_hand: Optional[HandData] = None
        detected_count = 0
        total_confidence = 0.0

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handed_info in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                handedness_label = handed_info.classification[0].label
                confidence = handed_info.classification[0].score

                if confidence < self._cfg.get("tracking.confidence_threshold", 0.5):
                    continue

                hand_data = self._extract_hand_data(
                    hand_landmarks, handedness_label, confidence, now
                )

                detected_count += 1
                total_confidence += confidence

                if handedness_label == "Left":
                    left_hand = hand_data
                    self._left_hand_lost = now
                    self._position_history["Left"].append(hand_data.wrist.copy())
                else:
                    right_hand = hand_data
                    self._right_hand_lost = now
                    self._position_history["Right"].append(hand_data.wrist.copy())

        tracking_quality = (
            total_confidence / max(detected_count, 1) if detected_count > 0 else 0.0
        )
        if detected_count < 2:
            tracking_quality *= (detected_count / 2.0)

        result = TrackingResult(
            left_hand=left_hand,
            right_hand=right_hand,
            timestamp=now,
            tracking_quality=tracking_quality,
        )

        return result

    def _extract_hand_data(
        self,
        hand_landmarks: Any,
        handedness: str,
        confidence: float,
        timestamp: float,
    ) -> HandData:
        landmarks_3d = np.zeros((21, 3), dtype=np.float64)
        for i, lm in enumerate(hand_landmarks.landmark):
            landmarks_3d[i] = [lm.x, lm.y, lm.z]

        wrist = landmarks_3d[self.WRIST_IDX].copy()

        palm_center = (
            landmarks_3d[self.WRIST_IDX]
            + landmarks_3d[self.INDEX_MCP_IDX]
            + landmarks_3d[self.MIDDLE_MCP_IDX]
            + landmarks_3d[self.PINKY_MCP_IDX]
        ) / 4.0

        if self._smoothing_enabled:
            if handedness == "Left":
                if self._smooth_left is not None:
                    wrist = self._smoothing_alpha * wrist + (
                        1.0 - self._smoothing_alpha
                    ) * self._smooth_left
                    palm_center = self._smoothing_alpha * palm_center + (
                        1.0 - self._smoothing_alpha
                    ) * self._smooth_left
                self._smooth_left = wrist.copy()
            else:
                if self._smooth_right is not None:
                    wrist = self._smoothing_alpha * wrist + (
                        1.0 - self._smoothing_alpha
                    ) * self._smooth_right
                    palm_center = self._smoothing_alpha * palm_center + (
                        1.0 - self._smoothing_alpha
                    ) * self._smooth_right
                self._smooth_right = wrist.copy()

        palm_to_middle = landmarks_3d[self.MIDDLE_TIP_IDX] - landmarks_3d[self.MIDDLE_MCP_IDX]
        palm_direction = np.zeros(3)
        if np.linalg.norm(palm_to_middle) > 0.001:
            palm_direction = palm_to_middle / np.linalg.norm(palm_to_middle)

        hand_data = HandData(
            handedness=handedness,
            confidence=confidence,
            wrist=wrist,
            palm_center=palm_center,
            index_tip=landmarks_3d[self.INDEX_TIP_IDX].copy(),
            middle_tip=landmarks_3d[self.MIDDLE_TIP_IDX].copy(),
            thumb_tip=landmarks_3d[self.THUMB_TIP_IDX].copy(),
            pinky_tip=landmarks_3d[self.PINKY_TIP_IDX].copy(),
            landmarks_3d=landmarks_3d,
            last_seen=timestamp,
            palm_direction=palm_direction,
        )

        return hand_data

    def draw_landmarks(self, frame: np.ndarray, tracking_result: TrackingResult) -> np.ndarray:
        """
        Draw hand landmarks and connections on the frame.

        Args:
            frame: BGR image to draw on.
            tracking_result: Current tracking result.

        Returns:
            Frame with landmarks drawn.
        """
        h, w = frame.shape[:2]

        for hand_data in [tracking_result.left_hand, tracking_result.right_hand]:
            if hand_data is None:
                continue
            landmarks_3d = hand_data.landmarks_3d

            connections = self._mp_hands.HAND_CONNECTIONS
            for connection in connections:
                i1, i2 = connection
                if i1 < 21 and i2 < 21:
                    p1 = (int(landmarks_3d[i1][0] * w), int(landmarks_3d[i1][1] * h))
                    p2 = (int(landmarks_3d[i2][0] * w), int(landmarks_3d[i2][1] * h))
                    cv2.line(frame, p1, p2, (0, 255, 0), 2)

            color = (255, 0, 0) if hand_data.handedness == "Left" else (0, 0, 255)
            for lm in landmarks_3d:
                px = int(lm[0] * w)
                py = int(lm[1] * h)
                cv2.circle(frame, (px, py), 4, color, -1)

            wrist_x = int(hand_data.wrist[0] * w)
            wrist_y = int(hand_data.wrist[1] * h)
            cv2.circle(frame, (wrist_x, wrist_y), 8, (0, 255, 255), 3)

        return frame

    def draw_debug_overlay(self, frame: np.ndarray, tracking_result: TrackingResult) -> np.ndarray:
        """
        Draw debug information overlay on the frame.

        Args:
            frame: BGR image to draw on.
            tracking_result: Current tracking result.

        Returns:
            Frame with debug overlay.
        """
        h, w = frame.shape[:2]
        y_offset = 30

        cv2.putText(
            frame,
            f"Tracking Quality: {tracking_result.tracking_quality:.2f}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        y_offset += 30

        for hand_data in [tracking_result.left_hand, tracking_result.right_hand]:
            if hand_data is None:
                continue
            wrist_str = (
                f"{hand_data.handedness} Wrist: "
                f"({hand_data.wrist[0]:.3f}, {hand_data.wrist[1]:.3f}, {hand_data.wrist[2]:.3f})"
            )
            cv2.putText(
                frame,
                wrist_str,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            y_offset += 25

        return frame

    def release(self) -> None:
        self._hands.close()
        self._logger.info("Hand tracker released")

    @property
    def processed_frames(self) -> int:
        return self._processed_frames


def cv2_import_check() -> bool:
    try:
        import cv2
        return True
    except ImportError:
        return False
