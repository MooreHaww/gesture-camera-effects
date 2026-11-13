"""Hand tracking. This is the only module that imports MediaPipe."""
import threading
import time
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


@dataclass
class TrackResult:
    seq: int            # number of the frame these hands were found in
    hands: list         # list[Hand]
    ms: float           # how long tracking that frame took


class TrackerThread:
    """Runs a tracker on its own thread, always on the newest frame handed to it.

    The main loop calls submit() with every camera frame and never has to wait: frames
    that arrive while the tracker is busy replace each other, so a slow tracker (the
    Raspberry Pi) skips frames instead of slowing the picture down. wait() returns the
    newest result, waiting at most `timeout` for the one of a given frame, which lets a
    fast tracker (a PC) keep every frame in step with its own hands.
    """

    def __init__(self, tracker):
        self.tracker = tracker
        self._cond = threading.Condition()
        self._job = None                     # (frame, timestamp ms, seq), newest only
        self._result: TrackResult | None = None
        self._error: BaseException | None = None
        self._running = True
        self.runs = 0                        # tracking results produced so far
        self._thread = threading.Thread(target=self._loop, name="tracker", daemon=True)
        self._thread.start()

    def submit(self, frame, timestamp_ms: int, seq: int):
        """Hand over a frame to track (the thread keeps its own reference: do not draw on it)."""
        with self._cond:
            self._job = (frame, timestamp_ms, seq)
            self._cond.notify_all()

    def wait(self, seq: int, timeout: float) -> TrackResult | None:
        """The newest result; waits up to `timeout` s for the result of frame `seq`."""
        with self._cond:
            if timeout > 0:
                self._cond.wait_for(lambda: self._error is not None or (
                    self._result is not None and self._result.seq >= seq), timeout)
            if self._error is not None:
                raise RuntimeError("hand tracking failed") from self._error
            return self._result

    def _loop(self):
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._job is not None or not self._running)
                if not self._running:
                    return
                frame, ts, seq = self._job
                self._job = None
            try:
                t0 = time.perf_counter()
                hands = self.tracker.detect(frame, ts)
                ms = (time.perf_counter() - t0) * 1000
            except BaseException as e:   # handed to the main loop, which logs it
                with self._cond:
                    self._error = e
                    self._cond.notify_all()
                return
            with self._cond:
                self._result = TrackResult(seq, hands, ms)
                self.runs += 1
                self._cond.notify_all()

    def close(self):
        with self._cond:
            self._running = False
            self._cond.notify_all()
        self._thread.join(timeout=2.0)
        self.tracker.close()
