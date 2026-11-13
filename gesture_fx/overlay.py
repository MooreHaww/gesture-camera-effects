"""Hand drawing on the camera frame: coloured skeleton and pinch line.

All on-screen UI (mode strip, buttons, info card, photo countdown ...) is in ui.py.
"""
import cv2

from .fxkit import col
from .tracking import CONNECTIONS

WHITE = (255, 255, 255)


def draw_hands(frame, infos, skeleton: bool = True):
    """The skeleton of each hand in that player's colour (key l turns it off)."""
    if not skeleton:
        return
    thick = max(1, round(frame.shape[0] / 300))
    for h in infos:
        c = col(h.color)
        pts = h.lm.astype(int)
        for a, b in CONNECTIONS:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), c, thick, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, tuple(p), 2, WHITE, -1, cv2.LINE_AA)


def draw_pinch(frame, info):
    """Line between thumb and index tip in the hand's colour."""
    a, b = info.lm[4].astype(int), info.lm[8].astype(int)
    c = col(info.color)
    cv2.line(frame, tuple(a), tuple(b), c, 2, cv2.LINE_AA)
    cv2.circle(frame, tuple(a), 5, c, -1, cv2.LINE_AA)
    cv2.circle(frame, tuple(b), 5, c, -1, cv2.LINE_AA)
