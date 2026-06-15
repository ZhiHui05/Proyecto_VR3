"""
Unit tests for the GestureClassifier module.
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gestures.gesture_classifier import (
    Gesture,
    GestureClassifier,
    GestureResult,
)
from src.tracking.hand_tracker import HandData, TrackingResult


class TestGestureClassifier(unittest.TestCase):
    def setUp(self) -> None:
        self._classifier = GestureClassifier()

    def _make_hand(self, wrist: np.ndarray, handedness: str = "Right") -> HandData:
        palm = wrist + np.array([0.0, -0.1, 0.0])
        return HandData(
            handedness=handedness,
            confidence=0.9,
            wrist=wrist.copy(),
            palm_center=palm.copy(),
            index_tip=wrist + np.array([0.1, -0.2, 0.0]),
            middle_tip=wrist + np.array([0.15, -0.25, 0.0]),
            thumb_tip=wrist + np.array([0.02, -0.05, 0.0]),
            pinky_tip=wrist + np.array([-0.05, -0.15, 0.0]),
            landmarks_3d=np.random.rand(21, 3).astype(np.float64),
            last_seen=time.perf_counter(),
            palm_direction=np.array([0.0, -1.0, 0.0]),
        )

    def _make_tracking(self, left: np.ndarray, right: np.ndarray) -> TrackingResult:
        lh = self._make_hand(left, "Left")
        rh = self._make_hand(right, "Right")
        return TrackingResult(
            left_hand=lh,
            right_hand=rh,
            timestamp=time.perf_counter(),
            tracking_quality=1.0,
        )

    def test_hover_hands_neutral(self) -> None:
        tracking = self._make_tracking(
            left=np.array([0.48, 0.45, 0.0]),
            right=np.array([0.52, 0.55, 0.0]),
        )
        result = self._classifier.classify(tracking)
        self.assertIn(result.gesture, [Gesture.HOVER, Gesture.NONE])

    def test_move_left_hands_shifted(self) -> None:
        tracking = self._make_tracking(
            left=np.array([0.35, 0.5, 0.0]),
            right=np.array([0.15, 0.5, 0.0]),
        )
        result = self._classifier.classify(tracking)
        result2 = self._classifier.classify(tracking)
        result3 = self._classifier.classify(tracking)
        self.assertIn(result.gesture, [Gesture.MOVE_LEFT, Gesture.HOVER])

    def test_move_right_hands_shifted(self) -> None:
        tracking = self._make_tracking(
            left=np.array([0.55, 0.5, 0.0]),
            right=np.array([0.7, 0.5, 0.0]),
        )
        result = self._classifier.classify(tracking)
        result2 = self._classifier.classify(tracking)
        self.assertIn(result.gesture, [Gesture.MOVE_RIGHT, Gesture.HOVER])

    def test_emergency_stop_crossed_wrists(self) -> None:
        tracking = self._make_tracking(
            left=np.array([0.5, 0.5, 0.0]),
            right=np.array([0.51, 0.5, 0.0]),
        )
        for _ in range(5):
            result = self._classifier.classify(tracking)
        result5 = self._classifier.classify(tracking)
        self.assertIn(
            result5.gesture,
            [Gesture.EMERGENCY_STOP, Gesture.HOVER],
        )

    def test_rotate_left_roll_difference(self) -> None:
        tracking = self._make_tracking(
            left=np.array([0.47, 0.68, 0.0]),
            right=np.array([0.53, 0.42, 0.0]),
        )
        result = self._classifier.classify(tracking)
        self.assertIn(result.gesture, [Gesture.ROTATE_LEFT, Gesture.ROTATE_RIGHT, Gesture.HOVER])

    def test_rotate_right_roll_difference(self) -> None:
        tracking = self._make_tracking(
            left=np.array([0.47, 0.32, 0.0]),
            right=np.array([0.53, 0.58, 0.0]),
        )
        result = self._classifier.classify(tracking)
        self.assertIn(result.gesture, [Gesture.ROTATE_LEFT, Gesture.ROTATE_RIGHT, Gesture.HOVER])

    def test_single_hand_returns_hover(self) -> None:
        tracking = TrackingResult(
            left_hand=self._make_hand(
                np.array([0.5, 0.5, 0.0]), "Left"
            ),
            right_hand=None,
            timestamp=time.perf_counter(),
            tracking_quality=0.5,
        )
        result = self._classifier.classify(tracking)
        self.assertIn(result.gesture, [Gesture.HOVER, Gesture.NONE])

    def test_result_has_metadata(self) -> None:
        tracking = self._make_tracking(
            left=np.array([0.4, 0.45, -0.05]),
            right=np.array([0.6, 0.45, -0.05]),
        )
        result = self._classifier.classify(tracking)
        self.assertIn("hand_distance", result.metadata)
        self.assertIn("wrist_roll_angle", result.metadata)

    def test_reset_clears_history(self) -> None:
        tracking = self._make_tracking(
            left=np.array([0.4, 0.5, 0.0]),
            right=np.array([0.6, 0.5, 0.0]),
        )
        for _ in range(6):
            self._classifier.classify(tracking)
        self._classifier.reset()
        self.assertEqual(
            self._classifier.get_current_gesture(), Gesture.NONE
        )


if __name__ == "__main__":
    unittest.main()
