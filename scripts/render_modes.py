"""Render a contact sheet of all ten modes with simulated hands. No camera, no people.

Every picture is generated: an empty drawn room as the background and grey mannequin
hands moved by a script, so the sheet is safe to commit and share.

    python scripts/render_modes.py [--out docs/modes.png]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from gesture_fx import modes, overlay  # noqa: E402
from gesture_fx.config import Config  # noqa: E402
from gesture_fx.players import Players  # noqa: E402
from gesture_fx.tracking import CONNECTIONS, Hand  # noqa: E402

W, H = 640, 480
FONTS = ROOT / "gesture_fx" / "assets" / "fonts"
NOTE = "Simulated hands, not a real recording"

# A right hand pointing up, palm to the camera, in palm sizes from the wrist.
_SHAPE = np.array([
    (0, 0), (-0.35, -0.25), (-0.6, -0.5), (-0.75, -0.75), (0, 0),          # wrist, thumb
    (-0.3, -0.95), (-0.35, -1.35), (-0.38, -1.6), (0, 0),                  # index
    (0, -1.0), (0, -1.45), (0, -1.75), (0, -2.0),                          # middle
    (0.25, -0.95), (0.3, -1.35), (0.33, -1.6), (0.35, -1.82),              # ring
    (0.45, -0.82), (0.55, -1.1), (0.6, -1.3), (0.65, -1.48)], np.float32)  # pinky
_PINCH_AT = np.array([-0.62, -1.25], np.float32)
_PINCH_DIR = np.array([0.5, -0.87], np.float32)


def sim_hand(x, y, size, gap, left=False):
    """Simulated hand with the wrist at (x, y). gap: thumb-to-index distance in palm sizes."""
    p = _SHAPE.copy()
    p[8] = _PINCH_AT + gap / 2 * _PINCH_DIR      # index tip
    p[4] = _PINCH_AT - gap / 2 * _PINCH_DIR      # thumb tip
    p[7] = (p[6] + p[8]) / 2
    p[3] = (p[2] + p[4]) / 2
    if left:
        p[:, 0] *= -1
    return Hand(p * size + (x, y), label="Left" if left else "Right")


def room():
    """An empty, neutral room drawn from shapes: wall, window, shelf, poster, floor."""
    yy = np.linspace(0, 1, H)[:, None, None]
    img = ((1 - yy) * np.array([150, 158, 165]) + yy * np.array([118, 124, 132])) * np.ones((1, W, 1))
    img = img.astype(np.uint8)
    cv2.rectangle(img, (0, 360), (W, H), (78, 96, 120), -1)                      # floor
    for x in range(-200, W + 200, 70):
        cv2.line(img, (x, 360), (x + (x - W // 2) // 2, H), (66, 82, 104), 2)
    cv2.rectangle(img, (40, 50), (210, 250), (95, 85, 80), -1)                   # window
    cv2.rectangle(img, (52, 62), (198, 238), (235, 222, 196), -1)
    cv2.line(img, (125, 62), (125, 238), (95, 85, 80), 6)
    cv2.line(img, (52, 150), (198, 150), (95, 85, 80), 6)
    cv2.rectangle(img, (420, 175), (620, 185), (70, 80, 95), -1)                 # shelf
    rng = np.random.default_rng(3)
    x = 428
    while x < 605:
        bw, bh = int(rng.integers(10, 20)), int(rng.integers(45, 75))
        c = tuple(int(v) for v in rng.integers(50, 210, 3))
        cv2.rectangle(img, (x, 175 - bh), (x + bw, 175), c, -1)
        x += bw + 2
    cv2.rectangle(img, (270, 60), (380, 170), (60, 60, 70), -1)                  # poster
    for r, c in ((45, (90, 140, 210)), (30, (200, 170, 80)), (15, (80, 70, 190))):
        cv2.circle(img, (325, 115), r, c, -1)
    cv2.rectangle(img, (230, 260), (400, 360), (60, 70, 85), -1)                 # low cabinet
    cv2.line(img, (315, 265), (315, 355), (48, 56, 70), 3)
    noise = rng.normal(0, 3, img.shape)
    return np.clip(cv2.GaussianBlur(img, (0, 0), 1.2) + noise, 0, 255).astype(np.uint8)


def draw_mannequin(frame, hands, size):
    """Grey glove-like hands, so modes that look at the picture (echo, lens) see them."""
    for hd in hands:
        pts = hd.landmarks.astype(int)
        cv2.fillConvexPoly(frame, cv2.convexHull(pts[[0, 1, 5, 9, 13, 17]]), (196, 198, 204))
        for a, b in CONNECTIONS:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (196, 198, 204), int(size * 0.24),
                     cv2.LINE_AA)


# Each scene: (mode, style, frames, function step -> (hands, clicked hand indices), size)
def _pair(gap_l, gap_r, cx=320, cy=330, apart=240, size=60, dy=0):
    return [sim_hand(cx - apart / 2 + 0.6 * size, cy + dy, size, gap_l, left=True),
            sim_hand(cx + apart / 2 - 0.6 * size, cy - dy, size, gap_r)]


def scene_portal(f):
    return _pair(1.9, 1.9, cx=300, cy=330, apart=360, size=70), []


def scene_styles(f):
    return [sim_hand(470, 420, 60, 1.2)], []


def scene_lens(f):
    return [sim_hand(560 + 0 * f, 340, 55, 1.4)], []


def scene_neon(f):
    a = f / 55 * 2 * np.pi
    x, y = 320 + 150 * np.sin(a), 330 + 60 * np.sin(2 * a)   # a figure eight
    return [sim_hand(x + 37, y + 75, 60, 0.05)], []


def scene_fireworks(f):
    hands = _pair(1.1, 1.1, cy=420, apart=380)
    return hands, [i for i, at in ((0, 18), (1, 26), (0, 34), (1, 40)) if f == at]


def scene_bubbles(f):
    return _pair(1.0, 1.0, cy=440 + 10 * np.sin(f / 9), apart=420, size=55), []


def scene_energy(f):
    if f < 50:                                   # charge with the hands close
        return _pair(1.2, 1.2, cy=400, apart=170, size=60), []
    g = f - 50                                   # then thrust up and to the right
    return _pair(1.2, 1.2, cx=320 + 35 * g, cy=400 - 28 * g, apart=170, size=60), []


def scene_storm(f):
    return _pair(1.2, 1.2, cx=320 + 150 * np.sin(f / 6), cy=360 - 40 * np.cos(f / 6), apart=280), []


def scene_fire(f):
    return _pair(1.5, 1.5, cy=430, apart=330, size=62), []


def scene_echo(f):
    x = 60 + 9 * f
    return [sim_hand(x, 330 + 30 * np.sin(f / 5), 60, 1.2)], [0] if f == 30 else []


SCENES = [(modes.PORTAL, "comic", 20, scene_portal),
          (modes.STYLES, "thermal", 5, scene_styles),
          (modes.LENS, "thermal", 5, scene_lens),
          (modes.NEON, "thermal", 52, scene_neon),
          (modes.FIREWORKS, "thermal", 46, scene_fireworks),
          (modes.BUBBLES, "thermal", 320, scene_bubbles),
          (modes.ENERGY, "thermal", 54, scene_energy),
          (modes.STORM, "thermal", 40, scene_storm),
          (modes.FIRE, "thermal", 30, scene_fire),
          (modes.ECHO, "thermal", 52, scene_echo)]


def render_scene(mode, style, frames, scene, base):
    players, renderer = Players(Config()), modes.ModeRenderer()
    for i, obj in enumerate(renderer._stateful.values()):   # same picture every run
        for j, part in enumerate([obj] + list(vars(obj).values())):
            if hasattr(part, "rng"):
                part.rng = np.random.default_rng(100 * i + j)
    out = base
    for f in range(frames):
        hands, clicked = scene(f)
        frame = base.copy()
        draw_mannequin(frame, hands, 60)
        infos = players.update(hands, frame.shape)
        clicks = [infos[i].id for i in clicked if i < len(infos)]
        out = renderer.render(mode, frame, infos, clicks, False, style, f / 30)
        out = out.copy() if out is frame else out
        overlay.draw_hands(out, infos)
    return out


def sheet(tiles, cols=5, tw=384, th=288):
    bold = ImageFont.truetype(str(FONTS / "Inter-Bold.ttf"), 30)
    reg = ImageFont.truetype(str(FONTS / "Inter-Regular.ttf"), 20)
    small = ImageFont.truetype(str(FONTS / "Inter-Bold.ttf"), 13)
    gap, top, cap = 12, 104, 38
    rows = (len(tiles) + cols - 1) // cols
    sw = cols * tw + (cols + 1) * gap
    sh = top + rows * (th + cap + gap) + 44
    img = Image.new("RGB", (sw, sh), (24, 26, 32))
    d = ImageDraw.Draw(img)
    d.text((gap, 18), "Gesture Camera Effects: the ten modes", font=bold, fill=(240, 240, 245))
    d.rounded_rectangle((gap, 60, gap + 470, 92), 8, fill=(250, 196, 60))
    d.text((gap + 12, 64), NOTE, font=reg, fill=(20, 20, 24))
    for n, (name, tile) in enumerate(tiles):
        r, c = divmod(n, cols)
        x, y = gap + c * (tw + gap), top + r * (th + cap + gap)
        rgb = cv2.cvtColor(cv2.resize(tile, (tw, th), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2RGB)
        img.paste(Image.fromarray(rgb), (x, y))
        d.rounded_rectangle((x + 6, y + 6, x + 92, y + 26), 5, fill=(250, 196, 60))
        d.text((x + 12, y + 8), "SIMULATED", font=small, fill=(20, 20, 24))
        d.text((x + 2, y + th + 6), f"{n + 1}  {name}", font=reg, fill=(225, 225, 232))
    d.text((gap, sh - 34), "Empty drawn room and grey mannequin hands, moved by "
           "scripts/render_modes.py. No camera or photo was used.", font=reg,
           fill=(160, 162, 172))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "modes.png"))
    a = ap.parse_args()
    base = room()
    tiles = []
    for mode, style, frames, scene in SCENES:
        name = modes.NAMES[mode]
        tiles.append((name[:1].upper() + name[1:], render_scene(mode, style, frames, scene, base)))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet(tiles).save(out, optimize=True)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
