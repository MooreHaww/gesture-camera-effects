"""The ten modes and the code that renders each one (their help text is in ui.INFO)."""
import cv2
import numpy as np

from . import effects, gestures, playful
from .players import closest_pair

PORTAL, STYLES, LENS, NEON, FIREWORKS, BUBBLES, ENERGY, STORM, FIRE, ECHO = range(10)

NAMES = ["portal", "styles", "lens", "neon draw", "fireworks", "bubbles", "energy ball",
         "storm", "hands on fire", "time echo"]
KEYS = "1234567890"  # keyboard key that selects each mode (key 0 = mode 10)
STYLE_FADE_S = 0.4


def frame_quad(infos):
    """(quad, hand A, hand B) for the two closest hands, or None."""
    pair = closest_pair(infos)
    if pair is None:
        return None
    a, b = pair
    quad = gestures.portal_quad(a.lm[gestures.THUMB_TIP], a.lm[gestures.INDEX_TIP],
                                b.lm[gestures.THUMB_TIP], b.lm[gestures.INDEX_TIP])
    return quad, a, b


class ModeRenderer:
    """Applies the selected mode to a frame and owns the state of stateful modes."""

    def __init__(self):
        self.neon = playful.NeonDraw()
        self.fireworks = playful.Fireworks()
        self.bubbles = playful.Bubbles()
        self.energy = playful.EnergyBall()
        self.storm = playful.Storm()
        self.fire = playful.FireHands()
        self.echo = playful.TimeEcho()
        self._stateful = {ECHO: self.echo, NEON: self.neon, FIREWORKS: self.fireworks,
                          BUBBLES: self.bubbles, ENERGY: self.energy, STORM: self.storm,
                          FIRE: self.fire}
        self._last_mode = None
        self._style = None
        self._prev_style = None
        self._style_t = -99.0

    def reset_mode(self, mode: int):
        """Throw away a mode's state, e.g. after it raised an error."""
        obj = self._stateful.get(mode)
        if obj is not None:
            obj.reset()

    def render(self, mode: int, frame, infos, clicks, clear: bool, style: str, t: float):
        """infos: list[HandInfo]; clicks: ids that released a pinch; clear: wipe neon drawing."""
        if mode != self._last_mode:  # leaving a mode clears its leftovers
            for m, obj in self._stateful.items():
                if m != mode:
                    obj.reset()
            self._last_mode = mode
        k = frame.shape[0] / 480.0

        if mode == PORTAL:
            fq = frame_quad(infos)
            if fq is None:
                return frame
            quad, a, b = fq
            return effects.portal(frame, quad, style, (a.color, b.color), t)
        if mode == STYLES:
            return self._styled(frame, style, t)
        if mode == LENS:
            out = frame
            for h in infos:
                radius = (30 + 100 * h.open) * k
                out = effects.lens(out, h.mid, radius, h.color)
            return out
        if mode == NEON:
            if clear:
                self.neon.clear()
            return self.neon.render(frame, infos, k, t)
        if mode == FIREWORKS:
            return self.fireworks.render(frame, infos, clicks, t, k)
        if mode == BUBBLES:
            return self.bubbles.render(frame, infos, t, k)
        if mode == ENERGY:
            return self.energy.render(frame, infos, clicks, t, k)
        if mode == STORM:
            return self.storm.render(frame, infos, k)
        if mode == FIRE:
            return self.fire.render(frame, infos, t, k)
        if mode == ECHO:
            return self.echo.render(frame, infos, clicks, t, k)
        raise ValueError(mode)

    def _styled(self, frame, style, t):
        """Apply a style, cross-fading from the previous one for a moment after a change."""
        if self._style is None:
            self._style = style
        if style != self._style:
            self._prev_style, self._style, self._style_t = self._style, style, t
        cur = effects.apply_style(frame, style, t)
        a = (t - self._style_t) / STYLE_FADE_S
        if self._prev_style is not None and 0 <= a < 1:
            prev = effects.apply_style(frame, self._prev_style, t)
            return cv2.addWeighted(prev, 1 - a, cur, a, 0)
        return cur
