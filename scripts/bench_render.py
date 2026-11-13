"""Time every mode with fake moving hands. No camera or MediaPipe needed.

Measures the cost of the player logic, the effect and the HUD per frame, so you
can see how effects scale with the number of hands. It does not include camera
capture or hand tracking; use `python -m gesture_fx --benchmark 30` for those.

    python scripts/bench_render.py [--frames 90] [--hands 1 2 4 6] [--lite]

--lite times the Raspberry Pi profile (fewer particles, lighter heavy styles).
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from gesture_fx import modes, overlay  # noqa: E402
from gesture_fx.config import Config  # noqa: E402
from gesture_fx.ui import UI, UIState  # noqa: E402
from gesture_fx.players import Players  # noqa: E402
from gesture_fx.tracking import Hand  # noqa: E402


def fake_hand(cx, cy, size, open_amount) -> Hand:
    """A plausible 21-point hand: palm upward, thumb and index spread by open_amount."""
    lm = np.tile(np.array([cx, cy - 0.6 * size], np.float32), (21, 1))  # unset joints: palm
    lm[0] = (cx, cy)
    lm[9] = (cx, cy - size)
    lm[4] = (cx - 0.5 * size, cy - 1.1 * size)
    lm[8] = lm[4] + (size * (0.1 + 1.0 * open_amount), -0.2 * size)
    for i, (pip, tip) in enumerate(zip((10, 14, 18), (12, 16, 20))):
        x = cx + (i - 1) * 0.35 * size
        lm[pip] = (x, cy - 1.3 * size)
        lm[tip] = (x, cy - 1.9 * size)
    lm[6] = (cx + 0.2 * size, cy - 1.3 * size)
    return Hand(lm)


def hands_at(step, n, w, h):
    out = []
    for i in range(n):
        ph = i * 1.7
        cx = w * (0.15 + 0.7 * (i + 0.5) / n) + 40 * np.sin(step * 0.15 + ph)
        cy = h * 0.65 + 50 * np.cos(step * 0.11 + ph)
        out.append(fake_hand(cx, cy, 55, 0.5 + 0.5 * np.sin(step * 0.2 + ph)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--hands", type=int, nargs="+", default=[1, 2, 4, 6])
    ap.add_argument("--size", type=int, nargs=2, default=[640, 480], metavar=("W", "H"))
    ap.add_argument("--lite", action="store_true", help="the Raspberry Pi profile")
    a = ap.parse_args()
    w, h = a.size
    rng = np.random.default_rng(0)
    base = cv2.resize((rng.random((12, 16, 3)) * 200 + 30).astype("uint8"), (w, h))

    print(f"Render cost in ms per frame at {w}x{h}{', Pi profile' if a.lite else ''} (lower is better)")
    print(f"{'mode':<13}" + "".join(f"{n:>8} hands" for n in a.hands))
    for mode, name in enumerate(modes.NAMES):
        row = f"{name:<13}"
        for n in a.hands:
            players, renderer, ui = Players(Config()), modes.ModeRenderer(a.lite), UI()
            total = 0.0
            for step in range(a.frames + 10):
                t0 = time.perf_counter()
                infos = players.update(hands_at(step, n, w, h), base.shape)
                # every few frames pretend a pinch was released, so click effects run
                clicks = [i.id for i in infos] if step % 25 == 0 else []
                out = renderer.render(mode, base, infos, clicks, False, "thermal", step / 30)
                out = out.copy() if out is base else out
                overlay.draw_hands(out, infos)
                ui.draw(out, UIState(mode=mode, names=tuple(modes.NAMES), t=step / 30))
                if step >= 10:  # skip warm-up
                    total += time.perf_counter() - t0
            row += f"{total / a.frames * 1000:>14.1f}"
        print(row)


if __name__ == "__main__":
    main()
