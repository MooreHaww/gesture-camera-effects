"""Pure gesture math on 21-point hand landmarks. No OpenCV or MediaPipe."""
import numpy as np

WRIST, THUMB_TIP, INDEX_TIP = 0, 4, 8
MIDDLE_MCP = 9
INDEX_MCP, PINKY_MCP = 5, 17
FINGER_TIPS = (8, 12, 16, 20)
FINGER_PIPS = (6, 10, 14, 18)


def _dist(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


def palm_size(lm) -> float:
    """Wrist to middle-finger knuckle: a scale reference that tracks hand size."""
    return _dist(lm[WRIST], lm[MIDDLE_MCP])


def pinch_distance(lm) -> float:
    return _dist(lm[THUMB_TIP], lm[INDEX_TIP])


def pinch_ratio(lm) -> float:
    """Thumb-index distance in palm sizes, so it ignores distance to the camera."""
    size = palm_size(lm)
    return pinch_distance(lm) / size if size > 1e-6 else 0.0


def pinch_midpoint(lm) -> np.ndarray:
    return (np.asarray(lm[THUMB_TIP], float) + np.asarray(lm[INDEX_TIP], float)) / 2


def openness(ratio: float, closed: float = 0.2, open_: float = 1.1) -> float:
    """Map a pinch ratio to 0 (pinched) .. 1 (fully open), clamped."""
    return float(np.clip((ratio - closed) / (open_ - closed), 0.0, 1.0))


def is_fist(lm) -> bool:
    """True when all four fingertips are closer to the wrist than their middle joints."""
    wrist = lm[WRIST]
    return all(_dist(lm[t], wrist) < _dist(lm[p], wrist)
               for t, p in zip(FINGER_TIPS, FINGER_PIPS))


def polygon_area(pts) -> float:
    """Absolute area by the shoelace formula. A self-crossing bow-tie scores low."""
    p = np.asarray(pts, float)
    x, y = p[:, 0], p[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2


def portal_quad(a_thumb, a_index, b_thumb, b_index) -> np.ndarray:
    """Order the four fingertips (thumb and index of two hands) into a quadrilateral.

    Hand A's tips are adjacent corners and so are hand B's. B's two tips can be
    joined in either order; the order with the shorter perimeter avoids a
    self-crossing bow-tie, so the quad stays valid however the hands are tilted.
    """
    at, ai = np.asarray(a_thumb, float), np.asarray(a_index, float)
    bt, bi = np.asarray(b_thumb, float), np.asarray(b_index, float)
    candidates = (np.array([at, ai, bi, bt]), np.array([at, ai, bt, bi]))

    def perimeter(q):
        return sum(_dist(q[i], q[(i + 1) % 4]) for i in range(4))

    return min(candidates, key=perimeter)


def finger_spread(lm, closed: float = 0.8, open_: float = 1.8) -> float:
    """How far the fingers are spread, 0 (together) .. 1 (wide), from thumb-to-pinky span."""
    size = palm_size(lm)
    if size < 1e-6:
        return 0.0
    span = _dist(lm[THUMB_TIP], lm[20]) / size
    return float(np.clip((span - closed) / (open_ - closed), 0.0, 1.0))


def palm_normal(lm, z, label):
    """Unit vector the palm faces, (x right, y down, z away from the camera), or None.

    Built from the wrist, index knuckle and little-finger knuckle with MediaPipe's
    relative depth. label is MediaPipe's own handedness for the image it saw; it sets
    the sign, since the same three points wind the other way on the other hand.
    Checked on photos with palms to the camera: the result points at the camera (z < 0).
    """
    if z is None or label not in ("Left", "Right"):
        return None
    p = np.column_stack([np.asarray(lm, float), np.asarray(z, float)])
    n = np.cross(p[INDEX_MCP] - p[WRIST], p[PINKY_MCP] - p[WRIST])
    size = float(np.linalg.norm(n))
    if size < 1e-6:
        return None
    return n / size * (1.0 if label == "Right" else -1.0)


def is_peace(lm) -> bool:
    """Index and middle fingers up and apart, ring and little fingers curled (a V sign)."""
    wrist = lm[WRIST]

    def up(tip, pip):
        return _dist(lm[tip], wrist) > 1.05 * _dist(lm[pip], wrist)

    def curled(tip, pip):
        return _dist(lm[tip], wrist) < _dist(lm[pip], wrist)

    v_apart = _dist(lm[8], lm[12]) > 0.3 * palm_size(lm)
    return up(8, 6) and up(12, 10) and curled(16, 14) and curled(20, 18) and v_apart


class BothPinch:
    """Portal style change: two hands pinched at the same moment, then both let go.

    update(a, b) takes whether each of the two hands is pinched right now and
    returns True once, when both are open again after having been pinched together.
    """

    def __init__(self):
        self.armed = False

    def reset(self):
        self.armed = False

    def update(self, a: bool, b: bool) -> bool:
        if a and b:
            self.armed = True
        elif self.armed and not a and not b:
            self.armed = False
            return True
        return False


class Dwell:
    """Fires once when a condition (a fingertip inside a button) has held for hold_s.

    Leaving cancels it. Dropouts shorter than grace_s are treated as tracking flicker
    and do not cancel it. After firing it re-arms only once the condition has been
    off for grace_s (the fingertip left the button), so one dwell = one press.
    """

    def __init__(self, hold_s: float = 1.5, grace_s: float = 0.15):
        self.hold_s, self.grace_s = hold_s, grace_s
        self.armed = True
        self._since = None
        self._off_since = None

    def progress(self, now: float) -> float:
        return 0.0 if self._since is None else min(1.0, (now - self._since) / self.hold_s)

    def update(self, active: bool, now: float) -> bool:
        if active:
            self._off_since = None
            if not self.armed:
                return False
            if self._since is None:
                self._since = now
            if now - self._since >= self.hold_s:
                self._since, self.armed = None, False
                return True
            return False
        if self._off_since is None:
            self._off_since = now
        if now - self._off_since >= self.grace_s:
            self._since, self.armed = None, True
        return False


class HandIdentity:
    """Gives each hand an id that stays the same across frames, so it keeps its colour.

    Matching uses where the hand is predicted to be (last position plus velocity),
    the left/right label when known, and a memory of hands that went missing, so a
    hand that leaves and returns gets its colour back even if it reappears far away.
    """

    def __init__(self, max_dist: float = 100.0, grace: int = 90):
        self.max_dist, self.grace = max_dist, grace
        self._tracks: dict[int, dict] = {}

    def update(self, points, labels=None, max_dist: float | None = None) -> list[int]:
        limit = self.max_dist if max_dist is None else max_dist
        pts = [np.asarray(p, float) for p in points]
        labels = list(labels) if labels is not None else [None] * len(pts)
        ids: list = [None] * len(pts)
        used: set = set()

        cands = []
        for tid, t in self._tracks.items():
            pred = t["pos"] + t["vel"] * (t["missed"] + 1)
            reach = limit * (1 + 0.5 * min(t["missed"], 4))
            for i, p in enumerate(pts):
                d = float(np.linalg.norm(p - pred))
                if d <= reach:
                    wrong = t["label"] and labels[i] and t["label"] != labels[i]
                    cands.append((d + (0.5 * limit if wrong else 0.0), tid, i))
        for _, tid, i in sorted(cands):
            if tid not in used and ids[i] is None:
                ids[i] = tid
                used.add(tid)

        # A hand that jumped too far for position matching still returns to the colour
        # of a missing hand with the same label (the nearest one if there are several).
        for i, p in enumerate(pts):
            if ids[i] is not None or labels[i] is None:
                continue
            free = [(float(np.linalg.norm(p - t["pos"])), tid) for tid, t in self._tracks.items()
                    if tid not in used and t["label"] == labels[i]]
            if free:
                ids[i] = min(free)[1]
                used.add(ids[i])

        for tid in list(self._tracks):
            if tid not in used:
                self._tracks[tid]["missed"] += 1
                if self._tracks[tid]["missed"] > self.grace:
                    del self._tracks[tid]
        for i, p in enumerate(pts):
            if ids[i] is None:
                ids[i] = next(n for n in range(len(pts) + len(self._tracks) + 1)
                              if n not in self._tracks)
                self._tracks[ids[i]] = {"pos": p, "vel": np.zeros(2), "missed": 0,
                                        "label": labels[i]}
            else:
                t = self._tracks[ids[i]]
                vel = (p - t["pos"]) / (t["missed"] + 1)
                t["vel"] = 0.5 * t["vel"] + 0.5 * vel
                t["pos"], t["missed"] = p, 0
                t["label"] = labels[i] or t["label"]
        return ids


class Smoother:
    """Exponential smoothing. alpha=1 means no smoothing."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.value = None

    def update(self, x: float) -> float:
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value
        return self.value

    def reset(self):
        self.value = None


class PinchClick:
    """Fires once per close-and-release, with hysteresis between the thresholds."""

    def __init__(self, close: float = 0.35, release: float = 0.6):
        assert close < release
        self.close, self.release = close, release
        self.closed = False

    def update(self, ratio: float) -> bool:
        if not self.closed and ratio < self.close:
            self.closed = True
        elif self.closed and ratio > self.release:
            self.closed = False
            return True
        return False


class HoldDetector:
    """Fires once after a condition has held for hold_s, then waits out a cooldown."""

    def __init__(self, hold_s: float = 1.0, cooldown_s: float = 1.0):
        self.hold_s, self.cooldown_s = hold_s, cooldown_s
        self._since = None
        self._blocked_until = 0.0

    def progress(self, now: float) -> float:
        """0..1 progress toward firing, for drawing a progress bar."""
        return 0.0 if self._since is None else min(1.0, (now - self._since) / self.hold_s)

    def update(self, active: bool, now: float) -> bool:
        if now < self._blocked_until or not active:
            self._since = None
            return False
        if self._since is None:
            self._since = now
        if now - self._since >= self.hold_s:
            self._since = None
            self._blocked_until = now + self.cooldown_s
            return True
        return False
