"""Threaded webcam capture that always hands back the newest frame.

The capture thread never dies on an error: it logs it, and after a run of failed
reads it reopens the camera, so a USB hiccup does not end the app.

On Linux (the Raspberry Pi) it asks for MJPG at 30 FPS and turns off the webcam's
dynamic frame rate with v4l2-ctl: many webcams (the Logitech C920s, for one) otherwise
drop to about 15 FPS in dim light to expose longer. The frame rate really delivered is
measured over the first seconds and printed.
"""
import shutil
import subprocess
import sys
import threading
import time

import cv2

from . import crashlog

REOPEN_AFTER = 60  # consecutive failed reads (about 2 s) before reopening the camera
FPS = 30
MEASURE_S = 2.0    # how long the delivered frame rate is measured at startup
# The same webcam control under its current and its older Linux name
NO_DYNAMIC_FPS = ("exposure_dynamic_framerate=0", "exposure_auto_priority=0")


def fixed_frame_rate(index: int) -> str | None:
    """Turn off the webcam's dynamic frame rate with v4l2-ctl (Linux, from v4l-utils).

    Returns the control that was set, or None when v4l2-ctl is not installed or the
    camera has no such control; both are fine and stay quiet.
    """
    exe = shutil.which("v4l2-ctl")
    if exe is None:
        return None
    for ctrl in NO_DYNAMIC_FPS:
        try:
            r = subprocess.run([exe, "-d", f"/dev/video{index}", f"--set-ctrl={ctrl}"],
                               capture_output=True, timeout=3)
        except (OSError, subprocess.SubprocessError):
            return None
        if r.returncode == 0:
            return ctrl
    return None


class CameraError(RuntimeError):
    """The camera could not be opened (missing, busy, or blocked by privacy settings)."""


class Camera:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480):
        self.index, self.width, self.height = index, width, height
        self.cap = self._open()
        if not self.cap.isOpened():
            self.cap.release()
            raise CameraError(f"Could not open camera index {index}")
        self._frame = None
        self.fps = None    # frame rate really delivered, measured over the first MEASURE_S
        self._seq = 0
        self._last_seq = 0
        self._running = True
        self._cond = threading.Condition()
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()

    def _open(self):
        linux = sys.platform.startswith("linux")
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2 if linux else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.index, backend)
        if not cap.isOpened() and backend != cv2.CAP_ANY:
            cap = cv2.VideoCapture(self.index, cv2.CAP_ANY)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if linux:
                cap.set(cv2.CAP_PROP_FPS, FPS)
                ctrl = fixed_frame_rate(self.index)
                if ctrl:
                    crashlog.write(f"Camera: set {ctrl} with v4l2-ctl")
        return cap

    @property
    def size(self) -> tuple[int, int]:
        """Size of the frames really delivered (the requested size until the first one)."""
        if self._frame is not None:
            return self._frame.shape[1], self._frame.shape[0]
        return self.width, self.height

    def _report(self, frame, fps):
        h, w = frame.shape[:2]
        note = "" if (w, h) == (self.width, self.height) else \
            f" (asked for {self.width}x{self.height})"
        msg = f"Camera delivers {w}x{h} at {fps:.1f} FPS{note}"
        print(msg)
        crashlog.write(msg)

    def _loop(self):
        failures = 0
        first_t, count = None, 0
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
            if self.fps is None:   # measure the frame rate really delivered
                now = time.perf_counter()
                if first_t is None:
                    first_t = now
                elif now - first_t >= MEASURE_S:
                    self.fps = count / (now - first_t)
                    self._report(frame, self.fps)
                count += 1
            with self._cond:
                self._frame = frame
                self._seq += 1
                self._cond.notify_all()

    def wait_first(self, timeout: float) -> bool:
        """True once the first frame has arrived. On Windows a camera that another app
        is using (or that camera privacy blocks) often opens fine but never delivers."""
        with self._cond:
            return self._cond.wait_for(lambda: self._seq > 0, timeout)

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
