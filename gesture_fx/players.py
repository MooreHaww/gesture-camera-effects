"""Turns raw tracked hands into per-hand state: stable id, colour, openness, speed."""
from dataclasses import dataclass, replace

import numpy as np

from . import gestures
from .tracking import Hand

# BGR neon colours, one per player slot
PALETTE = [(255, 210, 40), (200, 60, 255), (80, 255, 110), (30, 170, 255),
           (255, 90, 150), (60, 240, 255), (90, 90, 255), (255, 255, 120)]


@dataclass
class HandInfo:
    id: int
    lm: np.ndarray        # (21, 2) pixel landmarks
    mid: np.ndarray       # thumb/index pinch midpoint
    palm: np.ndarray      # palm centre
    palm_r: float         # palm size in pixels
    open: float           # smoothed pinch openness, 0 pinched .. 1 open
    ratio: float          # smoothed pinch ratio
    fist: bool
    color: tuple
    vel: np.ndarray       # pinch point velocity, pixels per frame (smoothed)
    tips: np.ndarray      # (5, 2) fingertip positions
    label: str | None = None  # "Left" or "Right" (the user's real hand), if known
    spread: float = 0.0   # smoothed finger spread, 0 together .. 1 wide
    normal: np.ndarray | None = None  # unit vector the palm faces (x, y, z away from camera)


def closest_pair(infos: list[HandInfo]):
    """The two hands whose palms are closest together, or None with fewer than two."""
    best, best_d = None, float("inf")
    for i in range(len(infos)):
        for j in range(i + 1, len(infos)):
            d = float(np.hypot(*(infos[i].palm - infos[j].palm)))
            if d < best_d:
                best, best_d = (infos[i], infos[j]), d
    return best


def dedupe(hands: list[Hand]) -> list[Hand]:
    """Drop duplicate detections of one hand (two hands cannot share the same palm)."""
    kept: list[Hand] = []
    for hd in sorted(hands, key=lambda h: -h.score):
        palm = (hd.landmarks[gestures.WRIST] + hd.landmarks[gestures.MIDDLE_MCP]) / 2
        size = gestures.palm_size(hd.landmarks)
        if all(np.hypot(*(palm - (k.landmarks[gestures.WRIST] + k.landmarks[gestures.MIDDLE_MCP]) / 2))
               > 0.8 * max(size, gestures.palm_size(k.landmarks)) for k in kept):
            kept.append(hd)
    return kept


class Players:
    def __init__(self, cfg):
        self.cfg = cfg
        # MediaPipe assumes a mirrored (selfie) image. We mirror unless told not to, so
        # its labels are the user's real hands; swap them if the image is not mirrored.
        self.swap_labels = (not cfg.mirror) != cfg.swap_hands
        self.identity = gestures.HandIdentity()
        self._smooth, self._click, self._prev_mid, self._vel = {}, {}, {}, {}
        self._spread = {}
        self._seen: set = set()
        self.clicks: list[int] = []  # ids whose pinch was released this frame
        self.pinched: dict = {}      # id -> (x, y, palm size) for pinches closed now
        self.released: dict = {}     # id -> (x, y, palm size) for pinches released this frame

    def update(self, hands: list[Hand], frame_shape) -> list[HandInfo]:
        h, w = frame_shape[:2]
        hands = dedupe(hands)
        palms = [(hd.landmarks[gestures.WRIST] + hd.landmarks[gestures.MIDDLE_MCP]) / 2
                 for hd in hands]
        labels = [self._label(hd) for hd in hands]
        ids = self.identity.update(palms, labels, max_dist=0.25 * float(np.hypot(w, h)))
        infos, self.clicks = [], []
        self.pinched, self.released = {}, {}
        for hd, pid, palm, label in zip(hands, ids, palms, labels):
            lm = hd.landmarks
            if pid not in self._seen:  # new or returning hand: drop stale state
                self._smooth[pid] = gestures.Smoother(self.cfg.smooth_alpha)
                self._click[pid] = gestures.PinchClick(self.cfg.click_close, self.cfg.click_release)
                self._prev_mid.pop(pid, None)
                self._vel[pid] = np.zeros(2)
                self._spread[pid] = gestures.Smoother(0.4)
            ratio = self._smooth[pid].update(gestures.pinch_ratio(lm))
            mid = gestures.pinch_midpoint(lm)
            if pid in self._prev_mid:
                self._vel[pid] = 0.5 * self._vel[pid] + 0.5 * (mid - self._prev_mid[pid])
            self._prev_mid[pid] = mid
            # A pinch made with the other fingers curled can pass the fist test, so a
            # hand whose thumb and index touch counts as pinching, not as a fist.
            fist = gestures.is_fist(lm) and ratio >= self.cfg.click_close
            size = max(gestures.palm_size(lm), 1.0)
            if fist:
                self._click[pid].closed = False
            elif self._click[pid].update(ratio):
                self.clicks.append(pid)
                self.released[pid] = (mid[0], mid[1], size)
            if self._click[pid].closed:
                self.pinched[pid] = (mid[0], mid[1], size)
            infos.append(HandInfo(
                id=pid, lm=lm, mid=mid, palm=palm,
                palm_r=max(gestures.palm_size(lm), 1.0),
                open=gestures.openness(ratio, self.cfg.pinch_closed, self.cfg.pinch_open),
                ratio=ratio, fist=fist, color=PALETTE[pid % len(PALETTE)],
                vel=self._vel[pid].copy(),
                tips=lm[[4, 8, 12, 16, 20]], label=label,
                spread=self._spread[pid].update(gestures.finger_spread(
                    lm, self.cfg.spread_closed, self.cfg.spread_open)),
                normal=gestures.palm_normal(lm, hd.z, hd.label)))
        self._seen = set(ids)
        return infos

    def _label(self, hand: Hand):
        if hand.label not in ("Left", "Right"):
            return None
        if self.swap_labels:
            return "Left" if hand.label == "Right" else "Right"
        return hand.label


class HandPredictor:
    """Moves hands between tracker results so effects stay smooth on a slow tracker.

    Time is counted in camera frames. Each hand keeps its landmark velocity (pixels per
    frame) from its last two results and is carried forward along it on frames with no
    new result, for at most MAX_AHEAD frames, so a stalled tracker makes the hand stop
    rather than fly off. A new result is eased in over a couple of frames (BLEND) instead
    of snapping, which hides the small jump when the guess was off.

    Only positions move. Pinch, fist and openness come from the tracker result, and so
    do pinch clicks, which the main loop takes once per result. When every frame has its
    own result (a PC), hands pass through untouched.
    """

    MAX_AHEAD = 4         # frames a hand may be carried forward (about 130 ms at 30 fps)
    MAX_SPEED = 0.6       # palm sizes per frame; faster is a tracking jump, not motion
    STILL = 0.015         # palm sizes per frame; slower counts as still (hides jitter)
    BLEND = 0.5           # share of the gap to the new estimate closed each frame

    def __init__(self):
        self.infos: list[HandInfo] = []
        self.seq = None
        self.gap = 1                      # frames between the last two results
        self._last: dict = {}             # id -> (seq, landmarks)
        self._v: dict = {}                # id -> (21, 2) landmark velocity, px per frame
        self._shown: dict = {}            # id -> (seq, landmarks drawn on that frame)

    def observe(self, infos: list[HandInfo], seq: int):
        """A new tracker result, for camera frame number `seq`."""
        vel = {}
        for h in infos:
            last = self._last.get(h.id)
            if last is None or not 0 < seq - last[0] <= 3 * self.MAX_AHEAD:
                continue
            v = (h.lm - last[1]) / (seq - last[0])
            speed = float(np.mean(np.hypot(v[:, 0], v[:, 1]))) / h.palm_r
            if self.STILL <= speed <= self.MAX_SPEED:
                vel[h.id] = v
        self._v = vel
        self._last = {h.id: (seq, h.lm.copy()) for h in infos}
        self.gap = 1 if self.seq is None else max(1, seq - self.seq)
        self.infos, self.seq = infos, seq

    def predict(self, seq: int) -> list[HandInfo]:
        """Hands as they should be at camera frame `seq`."""
        if self.seq is None:
            return []
        ahead = min(seq - self.seq, self.MAX_AHEAD)
        out, shown = [], {}
        for h in self.infos:
            # HandInfo.vel is per tracker result; effects expect it per camera frame
            vel = h.vel if self.gap == 1 else h.vel / self.gap
            if ahead <= 0 and self.gap == 1:      # this frame's own result: as tracked
                out.append(h)
                shown[h.id] = (seq, h.lm)
                continue
            v = self._v.get(h.id)
            lm = h.lm + v * max(ahead, 0) if v is not None else h.lm
            prev = self._shown.get(h.id)
            if prev is not None and prev[0] == seq - 1:   # ease from last frame's hand
                expect = prev[1] + (v if v is not None else 0)
                lm = expect + self.BLEND * (lm - expect)
            lm = lm.astype(h.lm.dtype)
            shown[h.id] = (seq, lm)
            out.append(replace(h, lm=lm, mid=gestures.pinch_midpoint(lm),
                               palm=(lm[gestures.WRIST] + lm[gestures.MIDDLE_MCP]) / 2,
                               tips=lm[[4, 8, 12, 16, 20]], vel=vel))
        self._shown = shown
        return out
