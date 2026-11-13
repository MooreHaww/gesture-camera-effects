"""Picture effects: styles, lens and the quad portal. CPU only."""
from functools import lru_cache

import cv2
import numpy as np

from .fxkit import col, composite_glow, mix_col

STYLES = ["cartoon", "sketch", "comic", "oil paint", "neon glow", "edges", "thermal",
          "night vision", "vintage", "duotone", "glitch", "pixelate"]

# Sepia as a BGR -> BGR colour matrix (rows: out B, G, R; columns: in B, G, R)
_SEPIA = np.array([[0.131, 0.534, 0.272], [0.168, 0.686, 0.349], [0.189, 0.769, 0.393]], np.float32)
_DUO_DARK, _DUO_LIGHT = np.array([96, 24, 44], float), np.array([140, 190, 255], float)  # BGR
_DUOTONE_LUT = np.clip(_DUO_DARK + (_DUO_LIGHT - _DUO_DARK) * (np.arange(256)[:, None] / 255) ** 0.9,
                       0, 255).astype(np.uint8).reshape(1, 256, 3)
_EMBOSS = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]], np.float32)
_NOISE = np.random.default_rng(7).integers(0, 48, (4, 1088, 1920), dtype=np.uint8)


@lru_cache(maxsize=8)
def _vignette(h: int, w: int, strength: float) -> np.ndarray:
    """3-channel uint8 mask, 255 in the middle, darker toward the corners."""
    ys = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
    xs = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
    v = np.clip(1 - strength * (xs ** 2 + ys ** 2) / 2, 0, 1)
    return cv2.merge([(v * 255).astype(np.uint8)] * 3)


def _vig(img, strength):
    h, w = img.shape[:2]
    return cv2.multiply(img, _vignette(h, w, strength), scale=1 / 255)


def _noise(h, w, t):
    """Grain the size of the picture, a slice of a pre-made texture that moves every frame."""
    n = _NOISE[int(t * 30) % 4]
    if h > n.shape[0] or w > n.shape[1]:
        return cv2.resize(n, (w, h), interpolation=cv2.INTER_NEAREST)
    y = int(t * 977) % (n.shape[0] - h + 1)
    x = int(t * 1531) % (n.shape[1] - w + 1)
    return n[y:y + h, x:x + w]


@lru_cache(maxsize=8)
def _hue_base(h: int, w: int) -> np.ndarray:
    xs = np.arange(w, dtype=np.float32)[None, :]
    ys = np.arange(h, dtype=np.float32)[:, None]
    return ((xs + ys) / (w + h) * 255).astype(np.uint8)


@lru_cache(maxsize=8)
def _dot_tile(h: int, w: int, cell: int) -> np.ndarray:
    """Distance of every pixel to the centre of its halftone cell."""
    c = (np.arange(cell, dtype=np.float32) + 0.5) - cell / 2
    d = np.hypot(c[:, None], c[None, :])
    return np.tile(d, (h // cell + 1, w // cell + 1))[:h, :w]


def _cells_up(a, cell, h, w):
    """Blow a per-cell image up to full size, each cell a solid block."""
    up = cv2.resize(a, (a.shape[1] * cell, a.shape[0] * cell), interpolation=cv2.INTER_NEAREST)
    up = cv2.copyMakeBorder(up, 0, max(0, h - up.shape[0]), 0, max(0, w - up.shape[1]),
                            cv2.BORDER_REPLICATE)
    return up[:h, :w]


# The styles that cost over 1 ms at 640x480 on a PC; lite mode (the Pi) runs them at
# half size, which makes them 2 to 3 times cheaper.
HEAVY_STYLES = {"cartoon", "comic", "oil paint", "neon glow"}


def apply_style(frame: np.ndarray, name: str, t: float = 0.0, lite: bool = False) -> np.ndarray:
    """Restyle a picture. Every style is a few cheap OpenCV passes."""
    h, w = frame.shape[:2]
    if lite and name in HEAVY_STYLES and min(h, w) >= 64:
        small = cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        return cv2.resize(apply_style(small, name, t), (w, h), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if name == "cartoon":
        quant = (frame // 48) * 48 + 24
        blur = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 7)
        return cv2.bitwise_and(quant, quant, mask=edges)
    if name == "sketch":   # pencil: colour dodge of the picture with its blurred negative
        blur = cv2.GaussianBlur(255 - gray, (0, 0), 6)
        lines = cv2.divide(gray, 255 - blur, scale=256)
        lines = cv2.multiply(lines, lines, scale=1 / 255)   # darker strokes
        return cv2.cvtColor(lines, cv2.COLOR_GRAY2BGR)
    if name == "comic":    # halftone dots on paper, with ink outlines
        cell = 7
        ch, cw = max(1, h // cell), max(1, w // cell)
        small = cv2.resize(frame, (cw, ch), interpolation=cv2.INTER_AREA)
        ink = cv2.convertScaleAbs(small, alpha=1.4, beta=-40) // 86 * 86   # bold flat inks
        lum = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        radius = (1 - lum / 255) * cell * 0.78 + 0.6
        dots = cv2.compare(_dot_tile(h, w, cell), _cells_up(radius, cell, h, w), cv2.CMP_LT)
        out = np.empty_like(frame)
        out[:] = (225, 245, 252)
        cv2.copyTo(_cells_up(ink, cell, h, w), dots, out)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 70, 150)
        out[edges > 0] = (30, 20, 20)
        return out
    if name == "oil paint":   # big smudged strokes of bold colour with canvas relief
        small = cv2.resize(frame, (max(1, w // 3), max(1, h // 3)), interpolation=cv2.INTER_AREA)
        small = cv2.convertScaleAbs(cv2.medianBlur(small, 5), alpha=1.25, beta=-25)
        paint = cv2.resize(small & np.uint8(0xE0) | np.uint8(16), (w, h), interpolation=cv2.INTER_LINEAR)
        relief = cv2.filter2D(cv2.cvtColor(paint, cv2.COLOR_BGR2GRAY), -1, _EMBOSS, delta=128)
        return cv2.addWeighted(paint, 1.0, cv2.cvtColor(relief, cv2.COLOR_GRAY2BGR), 0.5, -64)
    if name == "neon glow":   # glowing rainbow outlines on a dark picture
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 120)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
        hue = cv2.applyColorMap(_hue_base(h, w) + np.uint8(int(t * 40) % 256), cv2.COLORMAP_HSV)
        lines = cv2.bitwise_and(hue, hue, mask=edges)
        glow = cv2.GaussianBlur(lines, (0, 0), 4)
        return cv2.add(cv2.convertScaleAbs(frame, alpha=0.18),
                       cv2.addWeighted(lines, 1.0, glow, 2.2, 0))
    if name == "edges":
        e = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 140)
        out = np.zeros_like(frame)
        out[e > 0] = (255, 220, 0)
        return out
    if name == "thermal":
        return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    if name == "night vision":   # bright green, grainy, scan lines, dark edges
        g = cv2.add(cv2.convertScaleAbs(gray, alpha=1.7, beta=18), _noise(h, w, t))
        out = cv2.merge([cv2.convertScaleAbs(g, alpha=0.22), g, cv2.convertScaleAbs(g, alpha=0.4)])
        out[::3] = out[::3] // 4 * 3
        return _vig(out, 1.3)
    if name == "vintage":   # sepia, faded blacks, grain, soft vignette
        out = cv2.transform(frame, _SEPIA)
        out = cv2.convertScaleAbs(out, alpha=0.82, beta=30)
        out = cv2.subtract(out, cv2.merge([_noise(h, w, t) // 3] * 3))
        return _vig(out, 0.9)
    if name == "duotone":
        return cv2.LUT(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), _DUOTONE_LUT)
    if name == "glitch":    # split colour channels, torn rows, scan lines; changes 12 times a second
        rng = np.random.default_rng(int(t * 12))
        shift = int(rng.integers(3, 12))
        b, g, r = cv2.split(frame)
        out = cv2.merge([np.roll(b, shift, axis=1), g, np.roll(r, -shift, axis=1)])
        for _ in range(int(rng.integers(2, 6))):
            y0 = int(rng.integers(0, max(1, h - 4)))
            y1 = min(h, y0 + int(rng.integers(4, max(5, h // 12))))
            out[y0:y1] = np.roll(out[y0:y1], int(rng.integers(-w // 12 - 1, w // 12 + 1)), axis=1)
        out[1::2] = out[1::2] // 4 * 3
        return out
    if name == "pixelate":
        small = cv2.resize(frame, (max(1, w // 14), max(1, h // 14)),
                           interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    raise ValueError(f"unknown style: {name}")


def _roi_grid(frame, center, radius):
    """Bounding box around a circle plus coordinate grids relative to the centre."""
    h, w = frame.shape[:2]
    cx, cy = float(center[0]), float(center[1])
    x0, x1 = max(0, int(cx - radius)), min(w, int(cx + radius))
    y0, y1 = max(0, int(cy - radius)), min(h, int(cy + radius))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    xs, ys = np.meshgrid(np.arange(x0, x1, dtype=np.float32),
                         np.arange(y0, y1, dtype=np.float32))
    return x0, x1, y0, y1, cx, cy, xs - cx, ys - cy


def _remap_roi(frame, out, grid, map_x, map_y):
    x0, x1, y0, y1 = grid[:4]
    out[y0:y1, x0:x1] = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT)


def lens(frame: np.ndarray, center, radius: float, colour=(255, 255, 255),
         magnification: float = 2.2) -> np.ndarray:
    """Magnifying lens: the middle is enlarged and the edge blends into the picture."""
    grid = _roi_grid(frame, center, radius)
    if grid is None or radius < 4:
        return frame
    _, _, _, _, cx, cy, dx, dy = grid
    t = np.clip(np.hypot(dx, dy) / radius, 0, 1) ** 2
    factor = 1.0 / magnification + (1.0 - 1.0 / magnification) * t  # 1 at the rim
    out = frame.copy()
    _remap_roi(frame, out, grid, (cx + dx * factor).astype(np.float32),
               (cy + dy * factor).astype(np.float32))
    cv2.circle(out, (int(cx), int(cy)), int(radius), col(colour), 3, cv2.LINE_AA)
    cv2.circle(out, (int(cx), int(cy)), int(radius) + 2, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def portal(frame: np.ndarray, quad, style: str, colours=((255, 255, 255),) * 2,
           t: float = 0.0, lite: bool = False) -> np.ndarray:
    """Styled picture inside a quadrilateral through four fingertips, with glowing edges.

    quad is ordered [A thumb, A index, B index, B thumb]. A polygon mask is much
    cheaper than a perspective warp and enough here, since it shows the same
    picture restyled.
    """
    h, w = frame.shape[:2]
    k = h / 480.0
    pts = np.round(np.asarray(quad)).astype(np.int32)
    x0, y0 = max(0, pts[:, 0].min()), max(0, pts[:, 1].min())
    x1, y1 = min(w, pts[:, 0].max() + 1), min(h, pts[:, 1].max() + 1)
    out = frame.copy()
    if x1 - x0 > 3 and y1 - y0 > 3:
        roi = frame[y0:y1, x0:x1]
        mask = np.zeros(roi.shape[:2], np.uint8)
        cv2.fillPoly(mask, [pts - (x0, y0)], 255)
        np.copyto(out[y0:y1, x0:x1], apply_style(roi, style, t, lite), where=mask[..., None] > 0)

    ca, cb = colours
    edge = [ca, mix_col(ca, cb), cb, mix_col(ca, cb)]
    corner = [ca, ca, cb, cb]
    layer = np.zeros_like(out)
    for e in range(4):
        p, q = pts[e], pts[(e + 1) % 4]
        cv2.line(layer, tuple(int(v) for v in p), tuple(int(v) for v in q),
                 col(edge[e]), max(2, int(3 * k)), cv2.LINE_AA)
        s = (t * 0.6 + e * 0.25) % 1.0   # a light that runs around the frame
        dot = p + (q - p) * s
        cv2.circle(layer, (int(dot[0]), int(dot[1])), max(2, int(4 * k)), (255, 255, 255), -1,
                   cv2.LINE_AA)
        cv2.circle(layer, tuple(int(v) for v in p), max(3, int(6 * k)), col(corner[e]), -1,
                   cv2.LINE_AA)
    return composite_glow(out, layer, dim=1.0, gain=1.6)
