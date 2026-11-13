"""Threaded webcam capture that always hands back the newest frame.

The capture thread never dies on an error: it logs it, and after a run of failed
reads it reopens the camera, so a USB hiccup does not end the app.
"""
import sys
import threading
import time

import cv2

from . import crashlog

REOPEN_AFTER = 60  # consecutive failed reads (about 2 s) before reopening the camera


class Camera:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480):
        self.index, self.width, self.height = index, width, height
        self.cap = self._open()
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {index}")
        self._frame = None
        self._seq = 0
        self._last_seq = 0
        self._running = True
        self._cond = threading.Condition()
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()

    def _open(self):
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    @property
    def size(self) -> tuple[int, int]:
        """Size of the frames really delivered (the requested size until the first one)."""
        if self._frame is not None:
            return self._frame.shape[1], self._frame.shape[0]
        return self.width, self.height

    def _loop(self):
        failures = 0
        while self._running:
            try:
                ok, frame = self.cap.read()
            except Exception:
                crashlog.write_once(("camera-read", self.index), "Camera read raised an error")
                ok, frame = False, None
            if not ok or frame is None:
                failures += 1
                if failures == REOPEN_AFTER:
                    crashlog.write(f"Camera gave no frames {failures} times in a row: reopening")
                    try:
                        self.cap.release()
                        self.cap = self._open()
                    except Exception:
                        crashlog.write("Camera reopen failed", sys.exc_info())
                    failures = 0
                time.sleep(0.03)
                continue
            failures = 0
            if self._frame is None:
                h, w = frame.shape[:2]
                note = "" if (w, h) == (self.width, self.height) else \
                    f" (asked for {self.width}x{self.height})"
                print(f"Camera delivers {w}x{h}{note}")
                crashlog.write(f"Camera delivers {w}x{h}{note}")
            with self._cond:
                self._frame = frame
                self._seq += 1
                self._cond.notify_all()

    def read(self, timeout: float = 5.0):
        """Block until a frame newer than the last one returned is available."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq != self._last_seq, timeout):
                return None
            self._last_seq = self._seq
            return self._frame.copy()

    def release(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()
