"""Small shared drawing kit: cheap glow, point plotting, particles, lightning."""
import cv2
import numpy as np

_K3 = np.ones((3, 3), np.uint8)


def col(c) -> tuple:
    """Colour as a tuple of plain ints, which is what OpenCV drawing wants."""
    return tuple(int(min(255, max(0, v))) for v in c)


def scale_col(c, f: float) -> tuple:
    return col(np.asarray(c, float) * f)


def mix_col(a, b) -> tuple:
    return col((np.asarray(a, float) + np.asarray(b, float)) / 2)


def composite_glow(frame, layer, dim: float = 0.85, gain: float = 2.0):
    """Add a bright layer on top of the frame plus a soft glow around it.

    The glow is made at quarter size (blur is cheap there) and scaled back up.
    dim darkens the camera picture first so the light stands out.
    """
    h, w = frame.shape[:2]
    base = frame if dim >= 0.999 else cv2.convertScaleAbs(frame, alpha=dim)
    small = cv2.resize(layer, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    small = cv2.blur(small, (5, 5))
    glow = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if gain != 1.0:
        glow = cv2.convertScaleAbs(glow, alpha=gain)
    return cv2.add(cv2.add(base, layer), glow)


def plot_points(layer, pos, colors):
    """Plot many coloured points in one go (vectorised), each made 3x3 pixels."""
    if len(pos) == 0:
        return
    h, w = layer.shape[:2]
    xi, yi = pos[:, 0].astype(int), pos[:, 1].astype(int)
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    if not ok.any():
        return
    pl = np.zeros_like(layer)
    pl[yi[ok], xi[ok]] = np.clip(colors[ok], 0, 255).astype(np.uint8)
    np.maximum(layer, cv2.dilate(pl, _K3), out=layer)


class ParticleSystem:
    """Capped particle pool stored in NumPy arrays and drawn in one pass."""

    def __init__(self, cap: int = 500):
        self.cap = cap
        self.rng = np.random.default_rng()
        self.reset()

    def reset(self):
        self.pos = np.zeros((0, 2), np.float32)
        self.vel = np.zeros((0, 2), np.float32)
        self.life = np.zeros(0, np.float32)
        self.decay = np.zeros(0, np.float32)
        self.color = np.zeros((0, 3), np.float32)

    def emit(self, origin, n, color, speed=(1.5, 7.0), decay=(0.02, 0.045), up=0.0, scale=1.0,
             ring=False, radius=0.0, jitter=0.0, whiten=0.5):
        """Add n particles. ring=True gives them all the same speed (a clean expanding
        ring); radius starts them on a circle around origin instead of at one point."""
        n = int(n)
        if n <= 0:
            return
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False) if ring else self.rng.uniform(0, 2 * np.pi, n)
        spd = (np.full(n, np.mean(speed)) if ring else self.rng.uniform(*speed, n)) * scale
        vel = np.stack([np.cos(ang) * spd, np.sin(ang) * spd - up * scale], axis=1)
        white = self.rng.uniform(0, whiten, (n, 1))
        c = np.asarray(color, np.float32) * (1 - white) + 255 * white
        origin = np.asarray(origin, np.float32)
        start = origin + radius * np.stack([np.cos(ang), np.sin(ang)], axis=1)
        if jitter:
            start = start + self.rng.normal(0, jitter, (n, 2))
        cat = np.concatenate
        self.pos = cat([self.pos, start.astype(np.float32)])[-self.cap:]
        self.vel = cat([self.vel, vel.astype(np.float32)])[-self.cap:]
        self.life = cat([self.life, np.ones(n, np.float32)])[-self.cap:]
        self.decay = cat([self.decay, self.rng.uniform(*decay, n).astype(np.float32)])[-self.cap:]
        self.color = cat([self.color, c.astype(np.float32)])[-self.cap:]

    def step(self, gravity: float = 0.0, drag: float = 0.0):
        if len(self.life) == 0:
            return
        self.vel[:, 1] += gravity
        if drag:
            self.vel *= 1 - drag
        self.pos += self.vel
        self.life -= self.decay
        keep = self.life > 0
        if not keep.all():
            self.pos, self.vel, self.life = self.pos[keep], self.vel[keep], self.life[keep]
            self.decay, self.color = self.decay[keep], self.color[keep]

    def draw(self, layer):
        plot_points(layer, self.pos, self.color * self.life[:, None])


def lightning(layer, p1, p2, color, rng, k: float, segs: int = 8, thick: int = 2):
    """Jagged crackling line between two points with a white core."""
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    d = p2 - p1
    length = float(np.hypot(*d))
    if length < 3:
        return
    nrm = np.array([-d[1], d[0]]) / length
    t = np.linspace(0, 1, segs + 1)
    off = rng.normal(0, min(0.07 * length, 16 * k), segs + 1)
    off[0] = off[-1] = 0
    pts = (p1[None, :] + t[:, None] * d[None, :] + off[:, None] * nrm[None, :]).astype(np.int32)
    cv2.polylines(layer, [pts], False, col(color), max(1, thick), cv2.LINE_AA)
    cv2.polylines(layer, [pts], False, (255, 255, 255), 1, cv2.LINE_AA)
