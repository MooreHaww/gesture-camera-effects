"""On-screen UI in one glass design language.

Everything here is drawn on the display canvas (window or fullscreen size), so it
stays crisp at any resolution. Things that live in the camera picture (buttons,
hands) are given in camera-frame pixels and mapped onto the canvas.

Cost: every widget is rendered once with Pillow (2x supersampled for smooth
edges) and cached as a small sprite. Per frame we only alpha-blend those sprites
and, under glass panels, blur the small area behind them at quarter size. Nothing
blurs a large area every frame.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Palette (RGB for Pillow sprites, BGR for OpenCV drawing)
ACCENT_RGB = (76, 201, 240)
ACCENT = ACCENT_RGB[::-1]
INK_RGB = (8, 20, 30)                  # dark text on the accent colour
PANEL_RGBA = (14, 16, 24, 140)
PANEL_HOT_RGBA = (14, 16, 24, 185)
BORDER_RGBA = (255, 255, 255, 52)
WHITE_RGB = (255, 255, 255)
MUTED_RGBA = (255, 255, 255, 150)
SS = 2                                 # supersampling for sprite edges

FONT_DIR = Path(__file__).resolve().parent / "assets" / "fonts"
FONT_FILES = {
    "bold": [FONT_DIR / "Inter-Bold.ttf", FONT_DIR / "Inter-SemiBold.ttf",
             Path("C:/Windows/Fonts/segoeuib.ttf"),
             Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
             Path("/Library/Fonts/Arial Bold.ttf")],
    "regular": [FONT_DIR / "Inter-Regular.ttf", FONT_DIR / "Inter-Medium.ttf",
                Path("C:/Windows/Fonts/segoeui.ttf"),
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                Path("/Library/Fonts/Arial.ttf")],
}

# Short help per mode, shown in the info card (two lines each)
INFO = [
    ("Thumb and index of both hands make a frame.", "Pinch both and let go to change the look."),
    ("Right hand pinch: next look.", "Left hand pinch: previous look."),
    ("Your pinch is a magnifying glass.", "Open your fingers to make it bigger."),
    ("Touch thumb and index to draw in the air.", "Make a fist to clear."),
    ("Pinch and let go to launch.", "Squeeze hard for sparks."),
    ("Pop the bubbles in your colour.", "Another player's colour resets your score."),
    ("Hold your hands close to charge.", "Thrust them the way you want to fire."),
    ("Swipe through the storm.", "Open hands push harder."),
    ("Spread your fingers for bigger flames.", "Flick a hand to throw a fireball."),
    ("Move and your ghosts follow you.", "Pinch and let go to freeze a ghost."),
]
INFO_FOOTER = "Two peace signs: photo"


@lru_cache(maxsize=None)
def font(weight: str, px: int):
    for path in FONT_FILES[weight]:
        if path.exists():
            return ImageFont.truetype(str(path), px)
    return ImageFont.load_default(px)


# ---------------------------------------------------------------------------
# Sprites
# ---------------------------------------------------------------------------
@dataclass
class Sprite:
    rgb: np.ndarray            # (h, w, 3) float32, BGR, premultiplied by alpha
    alpha: np.ndarray          # (h, w, 1) float32 0..1
    glass: np.ndarray | None   # (h, w, 1) float32 mask of the frosted area, or None

    @property
    def size(self):
        return self.alpha.shape[1], self.alpha.shape[0]


def _sprite(w: int, h: int, paint, glass_shape=None) -> Sprite:
    """Render with Pillow at SS x size, then scale down for anti-aliased edges.
    paint(draw, s) draws in supersampled pixels (s = SS); glass_shape(draw, s)
    draws the frosted-glass mask in white."""
    w, h = max(1, int(w)), max(1, int(h))
    img = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
    paint(ImageDraw.Draw(img), SS, img)
    img = img.convert("RGBa").resize((w, h), Image.LANCZOS)      # premultiplied resize
    arr = np.asarray(img).astype(np.float32) / 255.0
    rgb, alpha = arr[..., 2::-1].copy(), arr[..., 3:4].copy()
    mask = None
    if glass_shape is not None:
        m = Image.new("L", (w * SS, h * SS), 0)
        glass_shape(ImageDraw.Draw(m), SS)
        mask = (np.asarray(m.resize((w, h), Image.LANCZOS)).astype(np.float32) / 255.0)[..., None]
    return Sprite(rgb * 255.0, alpha, mask)


def _round_panel(d, box, r, fill=PANEL_RGBA, border=True, s=SS):
    d.rounded_rectangle(box, r, fill=fill)
    if border:
        d.rounded_rectangle(box, r, outline=BORDER_RGBA, width=max(1, int(1.5 * s)))


def _text_w(txt, weight, px):
    return font(weight, px).getlength(txt)


@lru_cache(maxsize=512)
def pill(txt: str, px: int, pad: int, hot: bool = False, dot: tuple | None = None,
         accent: bool = False) -> Sprite:
    """Rounded glass pill with text, an optional colour dot on the left, or filled with
    the accent colour (accent=True, for prompts)."""
    f_px = px * SS
    tw = _text_w(txt, "bold", f_px) / SS
    dot_w = px * 0.9 if dot else 0
    h = px * 1.75
    w = tw + 2 * pad + dot_w
    r = h / 2

    def paint(d, s, img):
        box = (0, 0, w * s - 1, h * s - 1)
        if accent:
            d.rounded_rectangle(box, r * s, fill=ACCENT_RGB + (235,))
        else:
            _round_panel(d, box, r * s, PANEL_HOT_RGBA if hot else PANEL_RGBA)
        x = pad * s
        if dot:
            dr = px * 0.28 * s
            cy = h * s / 2
            d.ellipse((x, cy - dr, x + 2 * dr, cy + dr), fill=tuple(int(c) for c in dot[::-1]))
            x += dot_w * s
        d.text((x, h * s / 2), txt, font=font("bold", f_px),
               fill=INK_RGB if accent else WHITE_RGB, anchor="lm")

    glass = None if accent else (lambda d, s: d.rounded_rectangle((0, 0, w * s - 1, h * s - 1), r * s, 255))
    return _sprite(w, h, paint, glass)


@lru_cache(maxsize=64)
def mode_strip(cur: int, names: tuple, u100: int) -> Sprite:
    """The 10 modes as numbers 1 to 10; the current one is an accent pill with its name."""
    u = u100 / 100
    keys = [str(i + 1) for i in range(len(names))]
    h, dot, gap, pad = 88 * u, 60 * u, 8 * u, 14 * u
    name_px = int(36 * u)
    name_w = _text_w(names[cur], "bold", name_px * SS) / SS + 40 * u
    w = (len(names) - 1) * (dot + gap) + dot + name_w + 2 * pad

    def paint(d, s, img):
        _round_panel(d, (0, 0, w * s - 1, h * s - 1), h * s / 2)
        x, cy = pad, h / 2
        for i in range(len(names)):
            if i == cur:
                wp = dot + name_w
                d.rounded_rectangle((x * s, (cy - dot / 2) * s, (x + wp) * s, (cy + dot / 2) * s),
                                    dot * s / 2, fill=ACCENT_RGB)
                d.text(((x + dot / 2) * s, cy * s), keys[i], font=font("bold", int(30 * u * s)),
                       fill=INK_RGB, anchor="mm")
                d.text(((x + dot - 2 * u) * s, cy * s), names[i], font=font("bold", name_px * s),
                       fill=INK_RGB, anchor="lm")
                x += wp + gap
            else:
                d.text(((x + dot / 2) * s, cy * s), keys[i], font=font("regular", int(30 * u * s)),
                       fill=MUTED_RGBA, anchor="mm")
                x += dot + gap

    return _sprite(w, h, paint, lambda d, s: d.rounded_rectangle((0, 0, w * s - 1, h * s - 1),
                                                                 h * s / 2, 255))


@lru_cache(maxsize=32)
def round_button(r: int, glyph: str, hot: bool) -> Sprite:
    """Glass disc with a glyph: 'next', 'prev', 'info', 'camera' or 'rec'."""
    size = 2 * r + 2

    def paint(d, s, img):
        c = size * s / 2
        R = r * s
        box = (c - R, c - R, c + R, c + R)
        d.ellipse(box, fill=PANEL_HOT_RGBA if hot else PANEL_RGBA)
        d.ellipse(box, outline=BORDER_RGBA if not hot else ACCENT_RGB + (200,), width=max(2, int(1.5 * s)))
        if glyph in ("next", "prev"):
            k = 1 if glyph == "next" else -1
            a = R * 0.36
            wdt = max(3, int(R * 0.13))
            pts = [(c - k * a * 0.55, c - a), (c + k * a * 0.55, c), (c - k * a * 0.55, c + a)]
            d.line(pts, fill=WHITE_RGB, width=wdt, joint="curve")
            for p in (pts[0], pts[2]):
                d.ellipse((p[0] - wdt / 2, p[1] - wdt / 2, p[0] + wdt / 2, p[1] + wdt / 2), fill=WHITE_RGB)
        elif glyph == "info":
            d.text((c, c), "?", font=font("bold", int(R * 1.15)), fill=WHITE_RGB, anchor="mm")
        elif glyph == "camera":
            bw, bh = R * 0.9, R * 0.62
            d.rounded_rectangle((c - bw / 2, c - bh / 2 + R * 0.06, c + bw / 2, c + bh / 2 + R * 0.06),
                                R * 0.12, fill=WHITE_RGB)
            d.rounded_rectangle((c - bw * 0.2, c - bh / 2 - R * 0.08, c + bw * 0.2, c - bh / 2 + R * 0.1),
                                R * 0.05, fill=WHITE_RGB)
            lr = R * 0.2
            d.ellipse((c - lr, c - lr + R * 0.06, c + lr, c + lr + R * 0.06), fill=PANEL_RGBA[:3] + (255,))
        elif glyph == "rec":
            lr = R * 0.38
            d.ellipse((c - lr, c - lr, c + lr, c + lr), fill=(255, 64, 64))

    return _sprite(size, size, paint,
                   lambda d, s: d.ellipse((s, s, (size - 1) * s, (size - 1) * s), 255))


@lru_cache(maxsize=16)
def info_card(title: str, lines: tuple, footer: str, u100: int) -> Sprite:
    u = u100 / 100
    w = 720 * u
    h = (112 + 52 * len(lines) + 56) * u
    r = 26 * u

    def paint(d, s, img):
        _round_panel(d, (0, 0, w * s - 1, h * s - 1), r * s)
        x = 34 * u * s
        d.text((x, 54 * u * s), title, font=font("bold", int(42 * u * s)), fill=WHITE_RGB, anchor="lm")
        for i, ln in enumerate(lines):
            d.text((x, (112 + 52 * i) * u * s), ln, font=font("regular", int(32 * u * s)),
                   fill=(255, 255, 255, 215), anchor="lm")
        y = (112 + 52 * len(lines) + 14) * u * s
        d.line((x, y - 22 * u * s, w * s - x, y - 22 * u * s), fill=(255, 255, 255, 40), width=max(1, s))
        d.text((x, y + 12 * u * s), footer, font=font("regular", int(28 * u * s)), fill=MUTED_RGBA,
               anchor="lm")

    return _sprite(w, h, paint, lambda d, s: d.rounded_rectangle((0, 0, w * s - 1, h * s - 1), r * s, 255))


@lru_cache(maxsize=8)
def brand_badge(name: str, logo: str | None, u100: int) -> Sprite:
    """The name in a glass pill, with the --logo image in front of it when one is given
    (and readable). Without a logo the pill holds the name only."""
    u = u100 / 100
    h = 76 * u
    px = int(36 * u)
    logo_img = None
    if logo:
        raw = cv2.imread(logo, cv2.IMREAD_UNCHANGED)
        if raw is not None:
            if raw.ndim == 2:
                raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGRA)
            elif raw.shape[2] == 3:
                raw = cv2.cvtColor(raw, cv2.COLOR_BGR2BGRA)
            logo_img = Image.fromarray(cv2.cvtColor(raw, cv2.COLOR_BGRA2RGBA))
    text_x = (h + 8 * u) if logo_img is not None else 30 * u
    w = text_x + _text_w(name, "bold", px * SS) / SS + 30 * u

    def paint(d, s, img):
        _round_panel(d, (0, 0, w * s - 1, h * s - 1), h * s / 2)
        if logo_img is not None:
            c = ((h / 2 + 4 * u) * s, h / 2 * s)
            side = int(2 * h * 0.34 * s)
            lg = logo_img.copy()
            lg.thumbnail((side, side), Image.LANCZOS)
            img.alpha_composite(lg, (int(c[0] - lg.width / 2), int(c[1] - lg.height / 2)))
        d.text((text_x * s, h / 2 * s), name, font=font("bold", px * s), fill=WHITE_RGB, anchor="lm")

    return _sprite(w, h, paint, lambda d, s: d.rounded_rectangle((0, 0, w * s - 1, h * s - 1),
                                                                 h * s / 2, 255))


@lru_cache(maxsize=8)
def countdown_disc(n: int, r: int) -> Sprite:
    size = 2 * r + 2

    def paint(d, s, img):
        c, R = size * s / 2, r * s
        d.ellipse((c - R, c - R, c + R, c + R), fill=PANEL_RGBA)
        d.ellipse((c - R, c - R, c + R, c + R), outline=BORDER_RGBA, width=max(2, int(1.5 * s)))
        d.text((c, c - R * 0.05), str(n), font=font("bold", int(R * 1.25)), fill=WHITE_RGB, anchor="mm")

    return _sprite(size, size, paint, lambda d, s: d.ellipse((s, s, (size - 1) * s, (size - 1) * s), 255))


@lru_cache(maxsize=48)
def waving_hand(px: int, angle10: int) -> Sprite:
    """A simple flat hand icon, rotated about the wrist (angle in tenths of a degree)."""
    def paint(d, s, img):
        S = px * s
        hand = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        hd = ImageDraw.Draw(hand)
        wcol = WHITE_RGB + (255,)
        hd.rounded_rectangle((S * 0.30, S * 0.45, S * 0.70, S * 0.85), S * 0.12, wcol)
        for fx, top in ((0.33, 0.20), (0.43, 0.12), (0.53, 0.14), (0.63, 0.24)):
            hd.rounded_rectangle((S * fx - S * 0.045, S * top, S * fx + S * 0.045, S * 0.55), S * 0.045, wcol)
        hd.rounded_rectangle((S * 0.16, S * 0.46, S * 0.26, S * 0.70), S * 0.05, wcol)
        hand = hand.rotate(angle10 / 10, resample=Image.BICUBIC, center=(S / 2, S * 0.85))
        img.alpha_composite(hand)

    return _sprite(px, px, paint)


@lru_cache(maxsize=8)
def big_text(txt: str, px: int) -> Sprite:
    f = font("bold", px * SS)
    w = f.getlength(txt) / SS + px * 0.4
    h = px * 1.4

    def paint(d, s, img):
        d.text((w * s / 2, h * s / 2), txt, font=font("bold", px * s), fill=WHITE_RGB, anchor="mm")

    return _sprite(w, h, paint)


# ---------------------------------------------------------------------------
# Blending
# ---------------------------------------------------------------------------
def blit(canvas: np.ndarray, sp: Sprite, x: float, y: float, alpha: float = 1.0, glass: bool = True):
    """Blend sprite with its top-left at (x, y). Under the glass mask the picture is
    softened first (quarter-size blur of just this small area)."""
    if alpha <= 0.01:
        return
    H, W = canvas.shape[:2]
    w, h = sp.size
    x, y = int(round(x)), int(round(y))
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x0 >= x1 or y0 >= y1:
        return
    sx, sy = x0 - x, y0 - y
    roi = canvas[y0:y1, x0:x1]
    base = roi.astype(np.float32)
    a = sp.alpha[sy:sy + y1 - y0, sx:sx + x1 - x0] * alpha
    if glass and sp.glass is not None and min(roi.shape[:2]) >= 8:
        rh, rw = roi.shape[:2]
        small = cv2.resize(roi, (max(1, rw // 4), max(1, rh // 4)), interpolation=cv2.INTER_AREA)
        soft = cv2.resize(cv2.blur(small, (3, 3)), (rw, rh), interpolation=cv2.INTER_LINEAR)
        g = sp.glass[sy:sy + y1 - y0, sx:sx + x1 - x0] * alpha
        base += (soft.astype(np.float32) - base) * g
    rgb = sp.rgb[sy:sy + y1 - y0, sx:sx + x1 - x0]
    roi[:] = np.clip(base * (1 - a) + rgb * alpha, 0, 255).astype(np.uint8)


def blit_centered(canvas, sp, cx, cy, alpha=1.0, glass=True):
    w, h = sp.size
    blit(canvas, sp, cx - w / 2, cy - h / 2, alpha, glass)


def ring(canvas, c, r, frac, thick, colour=ACCENT, track=True):
    """Progress ring with rounded ends, clockwise from the top."""
    cx, cy = int(round(c[0])), int(round(c[1]))
    r, thick = int(round(r)), max(2, int(round(thick)))
    if track:
        x0, y0 = max(0, cx - r - thick), max(0, cy - r - thick)
        x1, y1 = min(canvas.shape[1], cx + r + thick + 1), min(canvas.shape[0], cy + r + thick + 1)
        if x1 > x0 and y1 > y0:
            roi = canvas[y0:y1, x0:x1]
            lay = roi.copy()
            cv2.circle(lay, (cx - x0, cy - y0), r, (255, 255, 255), thick, cv2.LINE_AA)
            cv2.addWeighted(lay, 0.22, roi, 0.78, 0, dst=roi)
    if frac <= 0:
        return
    end = -90 + 360 * min(1.0, frac)
    cv2.ellipse(canvas, (cx, cy), (r, r), 0, -90, end, colour, thick, cv2.LINE_AA)
    for ang in (-90, end):
        px = cx + r * math.cos(math.radians(ang))
        py = cy + r * math.sin(math.radians(ang))
        cv2.circle(canvas, (int(round(px)), int(round(py))), thick // 2, colour, -1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# State and the UI itself
# ---------------------------------------------------------------------------
@dataclass
class NavView:
    """A dwell button as the UI needs it, in camera-frame pixels."""
    centre: tuple
    radius: float
    step: int             # +1 next, -1 previous, 0 info
    progress: float = 0.0
    hot: bool = False


@dataclass
class UIState:
    mode: int = 0
    names: tuple = ()
    t: float = 0.0
    kiosk: bool = False
    toast: str | None = None
    toast_t0: float = -99.0
    stats: str | None = None              # None = hidden
    info_open: bool = False
    buttons: list = field(default_factory=list)       # list[NavView]
    countdown: int | None = None
    countdown_frac: float = 1.0
    flash: float = 0.0
    saved: bool = False
    attract: bool = False
    recording: bool = False
    counters: list = field(default_factory=list)      # (frame point, BGR colour, number)
    prompt: tuple | None = None                       # (frame point, text)
    photo_hold: tuple | None = None                   # (frame point, progress)
    cues: list = field(default_factory=list)          # (frame points, age 0..1)
    brand: str | None = None
    logo: str | None = None


class UI:
    TOAST_IN, TOAST_HOLD, TOAST_OUT = 0.15, 1.1, 0.35
    FADE_S = 0.2

    def __init__(self):
        self._fade: dict = {}
        self._last_t = None
        self._vignette = {}

    def _fader(self, key, visible: bool, dt: float) -> float:
        a = self._fade.get(key, 0.0)
        a = min(1.0, a + dt / self.FADE_S) if visible else max(0.0, a - dt / self.FADE_S)
        self._fade[key] = a
        return a * a * (3 - 2 * a)   # smoothstep

    def draw(self, canvas: np.ndarray, st: UIState, fmap=None):
        """Draw the whole UI onto canvas in place. fmap = (scale, x offset, y offset)
        maps camera-frame pixels to canvas pixels (None = canvas is the frame)."""
        s, ox, oy = fmap[:3] if fmap else (1.0, 0, 0)
        H, W = canvas.shape[:2]
        if fmap and len(fmap) > 3:
            fw, fh = fmap[3]
            pw, ph = fw * s, fh * s
        else:
            pw, ph = W, H
        u = ph / 1080 * (1.15 if st.kiosk else 1.0)
        u100 = max(20, int(round(u * 100)))
        u = u100 / 100

        def P(p):
            return ox + p[0] * s, oy + p[1] * s

        dt = 1 / 30 if self._last_t is None else float(np.clip(st.t - self._last_t, 0, 0.2))
        self._last_t = st.t
        m = 30 * u

        # attract: dim + vignette first, everything else sits on top
        a_attract = self._fader("attract", st.attract, dt)
        if a_attract > 0:
            self._draw_attract(canvas, st, u, ox, oy, pw, ph, a_attract)

        if a_attract < 1:
            vis = 1 - a_attract
            # buttons (next / previous / info), mapped from frame pixels
            for b in st.buttons:
                c = P(b.centre)
                r = int(round(b.radius * s))
                glyph = {1: "next", -1: "prev", 0: "info"}[b.step]
                if b.step == 0 and st.info_open:
                    glyph = "info"
                blit_centered(canvas, round_button(r, glyph, b.hot or (b.step == 0 and st.info_open)),
                              c[0], c[1], vis)
                if b.progress > 0:
                    ring(canvas, c, r + 12 * u, b.progress, 8 * u)

            # mode strip, bottom centre of the picture
            strip = mode_strip(st.mode, tuple(st.names), u100)
            blit(canvas, strip, ox + pw / 2 - strip.size[0] / 2, oy + ph - 28 * u - strip.size[1], vis)

            # bubble counters, prompts above hands
            for p, colour, n in st.counters:
                c = P(p)
                blit_centered(canvas, pill(str(n), int(52 * u), int(26 * u), dot=tuple(int(v) for v in colour)),
                              c[0], c[1], vis)
            if st.prompt:
                c = P(st.prompt[0])
                blit_centered(canvas, pill(st.prompt[1], int(44 * u), int(30 * u), accent=True), c[0], c[1], vis)

            # portal style change: rings bursting from both pinches
            for pts, age in st.cues:
                fade = 1 - age
                for p in pts:
                    c = P(p)
                    cv2.circle(canvas, (int(c[0]), int(c[1])), int((18 + 80 * age) * u * 1.5),
                               tuple(int(v * fade + 255 * (1 - fade) * 0.2) for v in ACCENT),
                               max(1, int(6 * u * fade)), cv2.LINE_AA)

            # photo hold (two peace signs)
            if st.photo_hold:
                p, prog = st.photo_hold
                c = P(p)
                r = int(46 * u)
                blit_centered(canvas, round_button(r, "camera", True), c[0], c[1])
                ring(canvas, c, r + 12 * u, prog, 8 * u)

        # brand, top left; stats under it
        top = oy + m
        name = (st.brand or "").strip()
        if name:   # no name, no badge (a --logo on its own shows nothing)
            badge = brand_badge(name, st.logo, u100)
            blit(canvas, badge, ox + m, top)
            top += badge.size[1] + 14 * u
        a_stats = self._fader("stats", st.stats is not None, dt)
        if a_stats > 0 and st.stats:
            blit(canvas, pill(st.stats, int(26 * u), int(22 * u)), ox + m, top, a_stats)

        # recording dot, top right next to the info button
        if st.recording:
            info = next((b for b in st.buttons if b.step == 0), None)
            r = int(30 * u)
            cx = (P(info.centre)[0] - info.radius * s - 16 * u - r) if info else ox + pw - m - r
            cy = P(info.centre)[1] if info else oy + m + r
            blit_centered(canvas, round_button(r, "rec", False), cx, cy)

        # info card under the info button
        a_info = self._fader("info", st.info_open and not st.attract, dt)
        if a_info > 0:
            title = st.names[st.mode] if st.names else ""
            card = info_card(title, INFO[st.mode % len(INFO)], INFO_FOOTER, u100)
            info = next((b for b in st.buttons if b.step == 0), None)
            right = (P(info.centre)[0] + info.radius * s) if info else ox + pw - m
            y = (P(info.centre)[1] + info.radius * s + 16 * u) if info else oy + m
            blit(canvas, card, right - card.size[0], y - 12 * u * (1 - a_info), a_info)

        # toast: mode or look name, top centre, fades and slides in
        if st.toast:
            age = st.t - st.toast_t0
            if 0 <= age < self.TOAST_IN + self.TOAST_HOLD + self.TOAST_OUT:
                if age < self.TOAST_IN:
                    a = age / self.TOAST_IN
                elif age < self.TOAST_IN + self.TOAST_HOLD:
                    a = 1.0
                else:
                    a = 1 - (age - self.TOAST_IN - self.TOAST_HOLD) / self.TOAST_OUT
                a = a * a * (3 - 2 * a)
                sp = pill(st.toast, int(84 * u), int(55 * u))
                blit_centered(canvas, sp, ox + pw / 2, oy + ph * 0.2 + 16 * u * (1 - a), a)

        # photo countdown: big numeral in an emptying ring, above head height
        if st.countdown is not None:
            r = int(150 * u)
            cx, cy = ox + pw / 2, oy + ph * 0.22
            blit_centered(canvas, countdown_disc(st.countdown, r), cx, cy)
            ring(canvas, (cx, cy), r - 16 * u, st.countdown_frac, 12 * u)

        # saved confirmation, above the strip
        a_saved = self._fader("saved", st.saved, dt)
        if a_saved > 0:
            sp = pill("Photo saved", int(46 * u), int(36 * u), dot=ACCENT)
            blit_centered(canvas, sp, ox + pw / 2, oy + ph - 190 * u + 10 * u * (1 - a_saved), a_saved)

        if st.flash > 0:
            cv2.addWeighted(canvas, 1 - st.flash, np.full_like(canvas, 255), st.flash, 0, dst=canvas)

    def _draw_attract(self, canvas, st, u, ox, oy, pw, ph, a):
        H, W = canvas.shape[:2]
        key = (W, H)
        if key not in self._vignette:   # precomputed once per canvas size
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            v = 1 - 0.55 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) ** 0.8
            m = (np.clip(0.38 * np.clip(v, 0.2, 1), 0, 1) * 255).astype(np.uint8)
            self._vignette[key] = cv2.merge([m, m, m])
        dimmed = cv2.multiply(canvas, self._vignette[key], scale=1 / 255)
        if a >= 0.999:
            canvas[:] = dimmed
        else:  # fading in or out
            cv2.addWeighted(dimmed, a, canvas, 1 - a, 0, dst=canvas)
        cx, cy = ox + pw / 2, oy + ph * 0.42
        # rings pulse outward and fade
        for k in range(3):
            ph_ = (st.t * 0.6 + k / 3) % 1.0
            r = (150 + 140 * ph_) * u
            col_ = tuple(int(c * (1 - ph_) * a) for c in ACCENT)
            cv2.circle(canvas, (int(cx), int(cy)), int(r), col_, max(1, int(3 * u)), cv2.LINE_AA)
        angle = 16 * math.sin(st.t * 5.0)
        hand = waving_hand(int(330 * u), int(round(angle / 2)) * 20)   # 2 degree steps, cached
        blit_centered(canvas, hand, cx, cy, a, glass=False)
        blit_centered(canvas, big_text("Wave a hand", int(120 * u)), cx, oy + ph * 0.76, a, glass=False)
