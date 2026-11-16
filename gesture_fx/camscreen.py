"""Full-window screens shown before the effects run: the camera picker and the
"no camera / camera busy" messages. They draw into the normal app window (Display),
so they work in a window, fullscreen and kiosk mode alike, with mouse and keyboard."""
import math
import sys
import threading
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import cameras
from .cameras import CameraInfo
from .ui import ACCENT, ACCENT_RGB, font

W, H = 1280, 720
BG = (18, 16, 14)          # BGR
MUTED = (175, 175, 175)    # RGB
WHITE = (255, 255, 255)
PREVIEW_SIZE = (320, 240)
NO_PICTURE_AFTER = 3.0     # a preview without a frame after this long is called busy
RESCAN_EVERY = 2.0         # the picker re-reads the camera list this often (hot plug)

# cv2.waitKeyEx codes for the arrow keys on Windows
LEFT, UP, RIGHT, DOWN = 2424832, 2490368, 2555904, 2621440
ENTER, ESC, TAB = 13, 27, 9

PRIVACY = ("Check Windows Settings > Privacy & security > Camera: "
           "'Camera access' and 'Let desktop apps access your camera' must be On.")
BUSY_LINES = (
    "Close other apps that may be using the camera: Teams, Zoom, Skype, the Camera app, "
    "browser tabs with video calls, OBS.",
    PRIVACY,
    "Unplug the webcam and plug it in again, or try another USB port.",
)
NO_CAMERA_LINES = (
    "Plug in the webcam (try another USB port, and give Windows a few seconds).",
    PRIVACY,
    "If it still does not show up, check that it appears under Cameras in Device Manager.",
)


def _wrap(text: str, px: int, weight: str, max_w: float) -> list[str]:
    f, lines, cur = font(weight, px), [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if cur and f.getlength(trial) > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return lines + ([cur] if cur else [])


def _fit(text: str, px: int, weight: str, max_w: float) -> str:
    """text, cut short with an ellipsis when it is wider than max_w."""
    f = font(weight, px)
    if f.getlength(text) <= max_w:
        return text
    while text and f.getlength(text + "...") > max_w:
        text = text[:-1]
    return text.rstrip() + "..."


def _title_px(text: str, max_w: float, px: int = 44, smallest: int = 26) -> int:
    """Largest size from px down to smallest at which the title fits in one line."""
    while px > smallest and font("bold", px).getlength(text) > max_w:
        px -= 2
    return px


def _draw_texts(canvas: np.ndarray, items):
    """items: (x, y, text, px, weight, rgb, anchor). Drawn with Pillow (Inter font)."""
    img = Image.fromarray(canvas[..., ::-1])
    d = ImageDraw.Draw(img)
    for x, y, txt, px, weight, rgb, anchor in items:
        d.text((x, y), txt, font=font(weight, px), fill=rgb, anchor=anchor)
    canvas[:] = np.asarray(img)[..., ::-1]


def _show(display, img) -> int:
    canvas = display.canvas_for(img)
    display.present(canvas)
    return cv2.waitKeyEx(30)


def message_screen(display, title: str, lines, footer: str, poll=None, every: float = 2.0,
                   keys: str = "", duration: float | None = None):
    """Show a message until something happens. Returns one of:
    ("poll", value)  poll() returned something other than None (it runs every `every` s),
    ("key", ch)      a key in `keys` was pressed,
    ("quit", None)   q, Esc or the window was closed,
    ("timeout", None) after `duration` seconds or any other key, when duration is set.
    """
    t0 = time.perf_counter()
    next_poll = t0 + (0.6 if poll else math.inf)  # paint once before the first try
    dots = 0
    while True:
        img = np.full((H, W, 3), BG, np.uint8)
        tpx = _title_px(title, W - 160)
        items = [(W / 2, 150, _fit(title, tpx, "bold", W - 160), tpx, "bold", WHITE, "mm")]
        y = 240
        for para in lines:
            for ln in _wrap(para, 26, "regular", W - 240):
                items.append((120, y, ln, 26, "regular", WHITE, "lm"))
                y += 38
            y += 18
        if poll:
            footer_txt = footer + "   " + "." * (dots % 4)
        else:
            footer_txt = footer
        items.append((W / 2, H - 70, footer_txt, 22, "regular", MUTED, "mm"))
        cv2.rectangle(img, (W // 2 - 60, 196), (W // 2 + 60, 200), ACCENT, -1)
        _draw_texts(img, items)
        key = _show(display, img)
        if display.closed():
            return "quit", None
        if key in (ESC, ord("q"), ord("Q")):
            return "quit", None
        if 0 <= key < 0x110000 and chr(key).lower() in keys:
            return "key", chr(key).lower()
        now = time.perf_counter()
        if duration is not None and (key != -1 or now - t0 >= duration):
            return "timeout", None
        if now >= next_poll:
            dots += 1
            value = poll()
            if value is not None:
                return "poll", value
            next_poll = time.perf_counter() + every


class Preview:
    """A small live picture of one camera, read on its own thread."""

    def __init__(self, info: CameraInfo):
        self.info = info
        self.frame = None
        self.t0 = time.perf_counter()
        self._running = True
        self._cap = None
        self._thread = threading.Thread(target=self._loop, daemon=True, name="preview")
        self._thread.start()

    def _loop(self):
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.info.index, backend)
        self._cap = cap
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, PREVIEW_SIZE[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PREVIEW_SIZE[1])
        while self._running and cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                self.frame = frame
            else:
                time.sleep(0.05)
        cap.release()

    @property
    def busy(self) -> bool:
        return self.frame is None and time.perf_counter() - self.t0 > NO_PICTURE_AFTER

    def close(self):
        self._running = False

    def join(self):
        self._thread.join(timeout=2.0)


def _layout(n: int):
    """Top-left corners and size of n preview tiles (4:3) plus room for a label."""
    cols = n if n <= 3 else 2 if n == 4 else 3
    rows = math.ceil(n / cols)
    gap, label, top, bottom, side = 30, 46, 120, 100, 60
    tw = min((W - 2 * side - (cols - 1) * gap) / cols,
             ((H - top - bottom - (rows - 1) * gap) / rows - label) * 4 / 3)
    th = tw * 3 / 4
    total_w = cols * tw + (cols - 1) * gap
    total_h = rows * (th + label) + (rows - 1) * gap
    x0, y0 = (W - total_w) / 2, top + (H - top - bottom - total_h) / 2
    rects = []
    for i in range(n):
        r, c = divmod(i, cols)
        rects.append((int(x0 + c * (tw + gap)), int(y0 + r * (th + label + gap)),
                      int(tw), int(th)))
    return rects, cols


def pick_camera(display, cams: list[CameraInfo], highlight: int = 0,
                reason: str | None = None) -> CameraInfo | None:
    """Live preview of every camera; pick with a click, its number, or arrows + Enter.
    Returns the chosen camera, or None when the user quits."""
    previews = [Preview(c) for c in cams]
    sel = max(0, min(highlight, len(cams) - 1))
    next_scan = time.perf_counter() + RESCAN_EVERY
    chosen = None
    try:
        while True:
            now = time.perf_counter()
            if now >= next_scan:     # a webcam plugged in or out: rebuild the tiles
                fresh = cameras.list_cameras()
                if fresh and fresh != [p.info for p in previews]:
                    keep = previews[sel].info if previews else None
                    for p in previews:
                        p.close()
                    for p in previews:
                        p.join()
                    previews = [Preview(c) for c in fresh]
                    sel = next((i for i, c in enumerate(fresh) if c == keep), 0)
                next_scan = now + RESCAN_EVERY
            if not previews:
                return None

            rects, cols = _layout(len(previews))
            img = np.full((H, W, 3), BG, np.uint8)
            items = [(W / 2, 52, "Choose a camera", 40, "bold", WHITE, "mm")]
            if reason:
                items.append((W / 2, 92, reason, 22, "regular", MUTED, "mm"))
            for i, (p, (x, y, tw, th)) in enumerate(zip(previews, rects)):
                frame = p.frame
                if frame is not None:
                    img[y:y + th, x:x + tw] = cv2.resize(frame, (tw, th),
                                                         interpolation=cv2.INTER_AREA)
                else:
                    cv2.rectangle(img, (x, y), (x + tw, y + th), (40, 36, 32), -1)
                    msg = "Busy or no picture" if p.busy else "Starting..."
                    items.append((x + tw / 2, y + th / 2, msg, 24, "regular", MUTED, "mm"))
                if i == sel:
                    cv2.rectangle(img, (x - 4, y - 4), (x + tw + 4, y + th + 4), ACCENT, 4)
                label = f"{i + 1}   {p.info.name}" if i < 9 else p.info.name
                weight = "bold" if i == sel else "regular"
                items.append((x + tw / 2, y + th + 24, _fit(label, 24, weight, tw), 24, weight,
                              ACCENT_RGB if i == sel else WHITE, "mm"))
            items.append((W / 2, H - 40, "Click a camera or press its number.   Arrows and "
                          "Enter also work.   Q quits.", 22, "regular", MUTED, "mm"))
            _draw_texts(img, items)
            key = _show(display, img)

            if display.closed() or key in (ESC, ord("q"), ord("Q")):
                return None
            for (cx, cy) in display.pop_clicks():
                for i, (x, y, tw, th) in enumerate(rects):
                    if x <= cx <= x + tw and y <= cy <= y + th + 46:
                        chosen = previews[i].info
            if ord("1") <= key <= ord("9") and key - ord("1") < len(previews):
                chosen = previews[key - ord("1")].info
            elif key in (RIGHT, TAB, ord("c")):
                sel = (sel + 1) % len(previews)
            elif key == LEFT:
                sel = (sel - 1) % len(previews)
            elif key == DOWN:
                sel = min(sel + cols, len(previews) - 1)
            elif key == UP:
                sel = max(sel - cols, 0)
            elif key in (ENTER, ord(" ")):
                chosen = previews[sel].info
            if chosen is not None:
                return chosen
    finally:
        for p in previews:   # stop all first, then wait: closing cameras takes a moment
            p.close()
        for p in previews:
            p.join()
