"""Resizable window that scales the frame up with the correct aspect ratio."""
import sys
import time

import cv2
import numpy as np


FULLSCREEN_SETTLE_S = 1.0   # the window may blink invisible this long after a switch


def fit_transform(w: int, h: int, win_w: int, win_h: int):
    """(scale, x offset, y offset) that fits a w x h frame centred in the window."""
    if win_w <= 0 or win_h <= 0 or (win_w, win_h) == (w, h):
        return 1.0, 0, 0
    s = min(win_w / w, win_h / h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    return s, (win_w - nw) // 2, (win_h - nh) // 2


def fit_frame(frame: np.ndarray, win_w: int, win_h: int) -> np.ndarray:
    """Scale frame to fit inside win_w x win_h, keeping aspect, on a black canvas.

    Always returns a new array, so callers may draw on it freely.
    """
    h, w = frame.shape[:2]
    s, x, y = fit_transform(w, h, win_w, win_h)
    if s == 1.0 and (x, y) == (0, 0):
        return frame.copy()
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    interp = cv2.INTER_LINEAR if s > 1 else cv2.INTER_AREA
    canvas = np.zeros((win_h, win_w, 3), np.uint8)
    canvas[y:y + nh, x:x + nw] = cv2.resize(frame, (nw, nh), interpolation=interp)
    return canvas


def _hide_cursor():
    if sys.platform != "win32":
        return
    try:
        import ctypes
        for _ in range(10):  # ShowCursor keeps a counter; push it below zero
            if ctypes.windll.user32.ShowCursor(False) < 0:
                break
    except Exception:
        pass


class Display:
    def __init__(self, title: str, frame_size: tuple[int, int], kiosk: bool = False,
                 initial_scale: float = 1.5):
        self.title = title
        self.fullscreen = False
        self.clicks: list = []          # left clicks, in camera-frame pixel coordinates
        self._map = (1.0, 0, 0, frame_size)
        # FREERATIO: we already fit the frame with the right aspect ourselves. With
        # OpenCV's default keep-ratio it would scale our canvas a second time when its
        # idea of the window size differs from ours (e.g. Windows display scaling),
        # which shows up as an extra black bar on one side.
        cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
        cv2.setMouseCallback(title, self._on_mouse)
        cv2.resizeWindow(title, int(frame_size[0] * initial_scale),
                         int(frame_size[1] * initial_scale))
        if kiosk:
            self.toggle_fullscreen()
            _hide_cursor()

    def canvas_for(self, frame: np.ndarray) -> np.ndarray:
        """The frame scaled to the current window, ready for HUD text to be drawn on it."""
        try:
            _, _, ww, wh = cv2.getWindowImageRect(self.title)
        except cv2.error:
            ww = wh = 0
        h, w = frame.shape[:2]
        s, x, y = fit_transform(w, h, ww, wh)
        self._map = (s, x, y, (w, h))
        return fit_frame(frame, ww, wh)

    @property
    def frame_map(self):
        """(scale, x offset, y offset, (frame w, frame h)) from the last canvas_for call:
        how camera-frame pixels land on the window canvas."""
        return self._map

    def _on_mouse(self, event, x, y, flags, param):
        """Mouse position arrives in canvas pixels; undo the fit to get frame pixels."""
        if event == cv2.EVENT_LBUTTONDOWN:
            s, ox, oy, (w, h) = self._map
            fx, fy = (x - ox) / s, (y - oy) / s
            if 0 <= fx < w and 0 <= fy < h:
                self.clicks.append((fx, fy))

    def pop_clicks(self) -> list:
        clicks, self.clicks = self.clicks, []
        return clicks

    def present(self, canvas: np.ndarray):
        cv2.imshow(self.title, canvas)
        if not self._shown:   # startup time can be read off crash.log
            self._shown = True
            from . import crashlog
            crashlog.write("First picture in the window")

    def toggle_fullscreen(self):
        self._toggled_t = time.perf_counter()
        self.fullscreen = not self.fullscreen
        cv2.setWindowProperty(
            self.title, cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)

    def closed(self) -> bool:
        """True once the window has been closed.

        One check is enough: the next imshow would quietly open the window again, so
        the window is never "gone" twice in a row. Only within a second of switching
        fullscreen, when the window can blink invisible, 3 checks in a row are needed.
        """
        try:
            gone = cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            gone = True
        if not gone:
            self._gone = 0
            return False
        self._gone += 1
        settling = time.perf_counter() - self._toggled_t < FULLSCREEN_SETTLE_S
        return self._gone >= 3 or not settling

    _gone = 0
    _toggled_t = -1e9
    _shown = False
