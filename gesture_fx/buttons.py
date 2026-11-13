"""On-screen buttons pressed by holding an index fingertip on them: NEXT, PREVIOUS
and the info button. Geometry and hit tests only; ui.py draws them and the dwell
timing lives in gestures.Dwell.
"""
from dataclasses import dataclass, field

import numpy as np

from . import gestures

INDEX_TIP = gestures.INDEX_TIP


@dataclass
class NavButton:
    step: int                 # +1 next effect, -1 previous effect, 0 info
    centre: np.ndarray        # frame pixels
    radius: float
    dwell: gestures.Dwell = field(default_factory=gestures.Dwell)

    def contains(self, p, margin: float = 1.0) -> bool:
        return float(np.hypot(*(np.asarray(p, float) - self.centre))) <= self.radius * margin


def layout(w: int, h: int, hold_s: float) -> list[NavButton]:
    """NEXT on the right edge, a smaller PREVIOUS on the left edge, both mid-height, and a
    small info button in the top right corner. Sizes follow the frame height, so they
    scale with the picture in every window size."""
    r_next, r_prev, r_info = 0.078 * h, 0.06 * h, 0.042 * h
    m = 0.04 * h    # room for the progress ring inside the frame
    mi = 0.028 * h
    return [NavButton(+1, np.array([w - m - r_next, h * 0.5]), r_next, gestures.Dwell(hold_s)),
            NavButton(-1, np.array([m + r_prev, h * 0.5]), r_prev, gestures.Dwell(hold_s)),
            NavButton(0, np.array([w - mi - r_info, mi + r_info]), r_info, gestures.Dwell(hold_s))]


def tips_inside(button: NavButton, infos) -> list:
    """Hands whose index fingertip is inside the button."""
    return [h for h in infos if button.contains(h.lm[INDEX_TIP])]


def blocked(infos, buttons, margin: float = 1.25) -> set:
    """Ids of hands pointing at a button (index tip inside it, with a small margin).
    Effects ignore these hands, so pressing a button does not pop bubbles etc."""
    return {h.id for h in infos for b in buttons if b.contains(h.lm[INDEX_TIP], margin)}
