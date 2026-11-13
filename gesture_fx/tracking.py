"""Hand tracking. This is the only module that imports MediaPipe."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

# Pairs of landmark indices for drawing the hand skeleton.
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
               (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


@dataclass
class Hand:
    landmarks: np.ndarray        # (21, 2) float32, pixel coordinates
    label: str | None = None     # "Left" or "Right" as reported for the image given
    score: float = 1.0           # handedness confidence, 0..1
    z: np.ndarray | None = None  # (21,) relative depth in pixels (smaller = nearer the camera)


def pad_to_square(frame: np.ndarray):
    """Centre the frame on a black square. Returns (square image, x offset, y offset).

    MediaPipe warns that its landmark projection assumes a square image when it is
    not told the image size; padding makes that assumption true. Landmarks found on
    the square are shifted back by the offsets to land on the original frame.
    """
    h, w = frame.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    if top == 0 and left == 0:
        return frame, 0, 0
    sq = cv2.copyMakeBorder(frame, top, side - h - top, left, side - w - left,
                            cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return sq, left, top


class HandTracker:
    def __init__(self, model_path: Path, num_hands: int = 2, square: bool = True):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"{model_path} not found. Run: python scripts/download_model.py")
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path),
                                     delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1
        self.square = square

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[Hand]:
        timestamp_ms = max(timestamp_ms, self._last_ts + 1)  # must increase
        self._last_ts = timestamp_ms
        img, ox, oy = pad_to_square(frame_bgr) if self.square else (frame_bgr, 0, 0)
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        scale = np.array([w, h], dtype=np.float32)
        offset = np.array([ox, oy], dtype=np.float32)
        hands = []
        for i, lms in enumerate(result.hand_landmarks):
            cat = result.handedness[i][0] if i < len(result.handedness) else None
            hands.append(Hand(np.array([[p.x, p.y] for p in lms], dtype=np.float32) * scale - offset,
                              cat.category_name if cat else None,
                              float(cat.score) if cat else 1.0,
                              np.array([p.z for p in lms], dtype=np.float32) * w))
        return hands

    def close(self):
        self._landmarker.close()
