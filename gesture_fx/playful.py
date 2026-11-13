"""Playful stateful modes: each reacts to hand motion, pinches and group size."""
from collections import deque

import cv2
import numpy as np

from .fxkit import (ParticleSystem, col, composite_glow, lightning, mix_col, plot_points,
                    scale_col)
from .players import closest_pair

def _mid(a, b):
    return (np.asarray(a, float) + np.asarray(b, float)) / 2


class Stroke:
    """One continuous line: every point it was given, with the time it was added."""

    __slots__ = ("pts", "ts", "colour", "level", "tail")

    def __init__(self, colour):
        self.pts: list = []
        self.ts: list = []
        self.colour = colour
        self.level = np.zeros(0, int)
        self.tail = None   # latest pen position while the stroke is being drawn

    def add(self, p, t):
        self.pts.append(np.asarray(p, float))
        self.ts.append(t)


class NeonDraw:
    """Pinch to draw glowing lines in the air. Lines fade by age.

    Each stroke is kept as a list of points and the whole stroke is redrawn every
    frame as one connected polyline, so nothing depends on what earlier frames
    drew and no frame can leave a gap. A stroke only ends when the pen is clearly
    lifted: a short flicker in the pinch, or the tracker losing the hand for a few
    frames, continues the same stroke.
    """

    # Pen down only when thumb and index tips touch, up when they are clearly apart.
    # Pinch ratio = tip distance / palm size; tips touching reads about 0.1-0.2 (the
    # landmarks sit at the finger ends), an inch apart about 0.45. Drawing only.
    DRAW_CLOSE, DRAW_OPEN = 0.2, 0.32
    PEN_UP_FRAMES = 4      # pinch must look open this many frames in a row to lift the pen
    LOST_FRAMES = 8        # frames a hand may vanish and still continue its stroke
    FULL_S, FADE_S = 3.0, 5.0   # a point stays bright, then fades out over FADE_S
    LEVELS = 8             # brightness steps used when drawing the fade
    MAX_POINTS = 6000
    STEP = 5.0             # px between curve samples (in 480p units)

    def __init__(self):
        self.reset()

    def reset(self):
        self._strokes: list[Stroke] = []     # finished and live strokes, oldest first
        self._pen = {}                       # hand id -> pen state

    def clear(self):
        self.reset()

    # -- input -------------------------------------------------------------
    def _extend(self, pen, p, t, k):
        """Add a smoothed point and fill the gap from the previous one with a curve
        (quadratic through the midpoints), so fast strokes stay round, not jagged."""
        pen["sm"] = p if pen["sm"] is None else 0.45 * pen["sm"] + 0.55 * p
        ctrl = pen["ctrl"]
        ctrl.append(pen["sm"].copy())
        del ctrl[:-3]
        s = pen["stroke"]
        s.tail = pen["sm"].copy()
        if len(ctrl) < 3:
            if not s.pts:
                s.add(ctrl[-1], t)
            return
        a, b, c = _mid(ctrl[0], ctrl[1]), ctrl[1], _mid(ctrl[1], ctrl[2])
        n = int(np.clip(np.linalg.norm(c - a) / (self.STEP * k), 1, 60))
        u = np.linspace(0, 1, n + 1)[1:, None]
        for q in (1 - u) ** 2 * a + 2 * (1 - u) * u * b + u ** 2 * c:
            s.add(q, t)

    def _update_pens(self, infos, t, k):
        seen = set()
        for h in infos:
            seen.add(h.id)
            pen = self._pen.get(h.id)
            # Pinched = pen down. A fist does not lift the pen: a pinch with the other
            # fingers curled can look like a fist and must keep drawing.
            down_now = h.ratio < (self.DRAW_OPEN if pen and pen["down"] else self.DRAW_CLOSE)
            if pen is None:
                pen = self._pen[h.id] = {"down": False, "up_count": 0, "lost": 0,
                                         "stroke": None, "ctrl": [], "sm": None}
            pen["lost"] = 0
            if down_now:
                pen["up_count"] = 0
                if not pen["down"]:
                    pen["down"] = True
                    pen["stroke"] = Stroke(h.color)
                    pen["ctrl"], pen["sm"] = [], None
                    self._strokes.append(pen["stroke"])
                self._extend(pen, h.mid.astype(float), t, k)
            elif pen["down"]:
                pen["up_count"] += 1
                if pen["up_count"] >= self.PEN_UP_FRAMES:
                    self._lift(pen)
                else:
                    self._extend(pen, h.mid.astype(float), t, k)  # maybe a flicker: keep going
        for hid in list(self._pen):
            if hid not in seen:
                pen = self._pen[hid]
                pen["lost"] += 1
                if pen["lost"] > self.LOST_FRAMES:
                    self._lift(pen)
                    del self._pen[hid]

    @staticmethod
    def _lift(pen):
        s = pen["stroke"]
        if s is not None and pen["ctrl"]:
            s.add(pen["ctrl"][-1], s.ts[-1] if s.ts else 0.0)  # end exactly at the last point
        if s is not None:
            s.tail = None
        pen["down"], pen["stroke"], pen["ctrl"], pen["sm"] = False, None, [], None

    def _expire(self, t):
        life = self.FULL_S + self.FADE_S
        live = {id(p["stroke"]) for p in self._pen.values() if p["stroke"] is not None}
        kept = []
        for s in self._strokes:
            # drop the faded-out start of each stroke; keep the rest connected
            cut = 0
            while cut < len(s.ts) and t - s.ts[cut] > life:
                cut += 1
            if cut:
                del s.pts[:cut], s.ts[:cut]
            if s.pts or id(s) in live:
                kept.append(s)
        self._strokes = kept
        total = sum(len(s.pts) for s in kept)
        while total > self.MAX_POINTS and len(kept) > 1:
            total -= len(kept[0].pts)
            kept.pop(0)

    # -- drawing -----------------------------------------------------------
    def _draw(self, layer, k):
        thick, core = max(5, int(9 * k)), max(2, int(3 * k))
        for s in self._strokes:
            if not s.pts:
                continue
            pts, levels = s.pts, s.level
            if s.tail is not None:  # live stroke: reach all the way to the pen
                pts, levels = pts + [s.tail], np.append(levels, levels[-1])
            pts = np.round(np.asarray(pts)).astype(np.int32)
            # Draw runs of equal brightness as polylines. Neighbouring runs share their
            # end point, so the line is continuous where the brightness steps.
            start = 0
            for i in range(1, len(pts) + 1):
                if i == len(pts) or levels[i] != levels[start]:
                    run = pts[start:i + 1 if i < len(pts) else i]
                    f = levels[start] / self.LEVELS
                    if f > 0:
                        self._polyline(layer, run, scale_col(s.colour, f),
                                       scale_col((255, 255, 255), f), thick, core)
                    start = i

    @staticmethod
    def _polyline(layer, run, colour, white, thick, core):
        arr = run.reshape(-1, 1, 2)
        for c, w in ((colour, thick), (white, core)):
            if len(run) == 1:
                cv2.circle(layer, tuple(int(v) for v in run[0]), max(1, w // 2), c, -1, cv2.LINE_AA)
            else:
                cv2.polylines(layer, [arr], False, c, w, cv2.LINE_AA)

    def render(self, frame, infos, k: float, t: float = 0.0):
        self._update_pens(infos, t, k)
        self._expire(t)
        for s in self._strokes:
            age = t - np.asarray(s.ts)
            f = np.clip(1 - (age - self.FULL_S) / self.FADE_S, 0, 1)
            s.level = np.ceil(f * self.LEVELS).astype(int)
        layer = np.zeros_like(frame)
        self._draw(layer, k)
        out = composite_glow(frame, layer, dim=0.75)
        for h in infos:  # pen cursor: filled while drawing, hollow while hovering
            p = (int(h.mid[0]), int(h.mid[1]))
            pen = self._pen.get(h.id)
            drawing = bool(pen and pen["down"])
            cv2.circle(out, p, max(4, int(9 * k)), col(h.color), -1 if drawing else max(1, int(2 * k)),
                       cv2.LINE_AA)
            cv2.circle(out, p, max(5, int(11 * k)), (255, 255, 255), 1, cv2.LINE_AA)
        return out


class Fireworks:
    """Pinch and let go: a big burst in your colour, with a ring, glitter and echoes."""

    def __init__(self):
        self.ps = ParticleSystem(1800)
        self.rng = np.random.default_rng()
        self._echoes = []

    def reset(self):
        self.ps.reset()
        self._echoes = []

    def _burst(self, pos, colour, k, scale=1.0):
        n = lambda v: int(v * scale)  # noqa: E731
        self.ps.emit(pos, n(170), colour, speed=(2, 12), decay=(0.010, 0.026), scale=k)
        self.ps.emit(pos, n(80), colour, speed=(8, 8), decay=(0.018, 0.03), ring=True, scale=k)
        self.ps.emit(pos, n(60), (255, 255, 255), speed=(1, 6), decay=(0.006, 0.014), scale=k,
                     whiten=0)

    def render(self, frame, infos, clicks, t: float, k: float):
        byid = {h.id: h for h in infos}
        for cid in clicks:
            h = byid.get(cid)
            if h is not None:
                self._burst(h.mid, h.color, k)
                for delay in (0.15, 0.32):  # smaller echo bursts around the first one
                    off = self.rng.normal(0, 35 * k, 2)
                    self._echoes.append((t + delay, h.mid + off, h.color))
        due = [e for e in self._echoes if e[0] <= t]
        self._echoes = [e for e in self._echoes if e[0] > t]
        for _, pos, colour in due:
            self._burst(pos, colour, k, 0.5)
        for h in infos:
            if h.open < 0.15:  # squeezing sheds a stream of sparks
                self.ps.emit(h.mid, 8, h.color, speed=(1, 5), decay=(0.04, 0.08), up=1.5, scale=k)
        self.ps.step(gravity=0.16 * k, drag=0.015)
        layer = np.zeros_like(frame)
        self.ps.draw(layer)
        return composite_glow(frame, layer, dim=0.7)


class Bubbles:
    """Bubbles in the colours of the hands in view, and no other colours. Pop your own
    colour for +1; touching the colour of another hand that is in view resets your count.
    With one hand every bubble is yours, so there is nothing to avoid.
    """

    MAX = 8
    SPAWN_P = 0.06        # chance of a new bubble each frame
    GONE_S = 1.0          # a hand's bubbles vanish this long after it leaves the picture

    def __init__(self):
        self.rng = np.random.default_rng()
        self.ps = ParticleSystem(400)
        self.reset()

    def reset(self):
        self.pos = np.zeros((0, 2))
        self.r = np.zeros(0)
        self.vy = np.zeros(0)
        self.ph = np.zeros(0)
        self.owner = np.zeros(0, int)       # hand id
        self.tint = np.zeros((0, 3))
        self.score = {}
        self.last_seen = {}
        self.ps.reset()

    def _spawn(self, owner, tint, w, h, k):
        r = self.rng.uniform(22, 42) * k
        self.pos = np.vstack([self.pos, [self.rng.uniform(0.06 * w, 0.94 * w), h + r]])
        self.r = np.append(self.r, r)
        self.vy = np.append(self.vy, -self.rng.uniform(0.45, 0.95) * k)
        self.ph = np.append(self.ph, self.rng.uniform(0, 6.28))
        self.owner = np.append(self.owner, owner)
        self.tint = np.vstack([self.tint, tint])

    def _keep(self, mask):
        self.pos, self.r, self.vy = self.pos[mask], self.r[mask], self.vy[mask]
        self.ph, self.owner, self.tint = self.ph[mask], self.owner[mask], self.tint[mask]

    def render(self, frame, infos, t: float, k: float):
        h, w = frame.shape[:2]
        colours = {i.id: i.color for i in infos}
        for pid in colours:
            self.last_seen[pid] = t

        # A hand that left takes its bubbles with it (after a short grace for tracking
        # dropouts), so only colours of hands in view are ever on screen.
        if len(self.r):
            here = np.array([o in colours or t - self.last_seen.get(o, -1e9) < self.GONE_S
                             for o in self.owner], bool)
            if not here.all():
                for j in np.flatnonzero(~here):
                    self.ps.emit(self.pos[j], 10, self.tint[j], speed=(0.5, 2.5),
                                 decay=(0.04, 0.08), scale=k)
                self._keep(here)

        if colours and len(self.r) < self.MAX and self.rng.random() < self.SPAWN_P:
            counts = {pid: int((self.owner == pid).sum()) for pid in colours}
            pid = min(counts, key=lambda p: (counts[p], self.rng.random()))
            self._spawn(pid, colours[pid], w, h, k)

        if len(self.r):
            self.pos[:, 1] += self.vy
            self.pos[:, 0] += np.sin(t * 1.2 + self.ph) * 0.5 * k
            self._keep(self.pos[:, 1] > -self.r)

        if len(self.r) and infos:
            tips = np.stack([i.tips for i in infos])  # (hands, 5, 2)
            d = np.hypot(self.pos[:, None, None, 0] - tips[None, :, :, 0],
                         self.pos[:, None, None, 1] - tips[None, :, :, 1]).min(axis=2)  # (b, hands)
            hit = d < self.r[:, None]
            popped = hit.any(axis=1)
            for j in np.flatnonzero(popped):
                hi = int(np.argmin(np.where(hit[j], d[j], 1e9)))
                player = infos[hi].id
                if self.owner[j] == player:
                    self.score[player] = self.score.get(player, 0) + 1
                    self.ps.emit(self.pos[j], 26, self.tint[j], speed=(1.5, 6),
                                 decay=(0.03, 0.06), scale=k)
                elif self.owner[j] in colours:          # another hand's colour: back to zero
                    self.score[player] = 0
                    self.ps.emit(self.pos[j], 30, (60, 60, 255), speed=(2, 7),
                                 decay=(0.03, 0.06), scale=k, whiten=0.1)
                else:                                   # its hand just left: a harmless pop
                    self.ps.emit(self.pos[j], 12, self.tint[j], speed=(1, 4),
                                 decay=(0.04, 0.08), scale=k)
            self._keep(~popped)
        self.ps.step(gravity=0.12 * k, drag=0.02)

        layer = np.zeros_like(frame)
        for (x, y), r, tint in zip(self.pos, self.r, self.tint):
            c, ri = (int(x), int(y)), int(r)
            cv2.circle(layer, c, ri, col(tint * 0.7), max(1, int(2 * k)), cv2.LINE_AA)
            cv2.circle(layer, c, max(1, ri - 2), col(tint * 0.2), -1)
            cv2.circle(layer, (c[0] - ri // 3, c[1] - ri // 3), max(2, ri // 6),
                       (255, 255, 255), -1, cv2.LINE_AA)
        self.ps.draw(layer)
        return composite_glow(frame, layer, dim=1.0, gain=1.2)   # scores: drawn by the UI


class EnergyBall:
    """Charge an energy ball with hands held close; thrust to fire it.

    Hands close together charge the ball over about 3 s. Its size pulses with finger
    spread. A quick thrust while charged fires it:
    - both hands pushed the same way (up, down, left, right, diagonal): a beam that way;
    - hands pushed toward the camera: a blast that fills the screen;
    - hands flung apart: the ball goes the way the palms face (toward the camera if
      they face it), from MediaPipe's depth estimate.
    """

    CHARGE_S = 3.0
    APART_RATE = 6.0   # palm-sizes per second
    ZOOM_RATE = 1.2    # palm growth (log) per second
    PUSH_RATE = 6.0    # palm-sizes per second, both hands moving the same way
    AIM_RATE = 2.5     # slower common motion than this has no clear direction
    BEAM_S = 0.7
    BLAST_S = 0.9
    SHAKE_S = 0.35

    def __init__(self):
        self.rng = np.random.default_rng()
        self.ps = ParticleSystem(700)
        self.reset()

    def reset(self):
        self.c = 0.0
        self.last_t = None
        self.prev = None            # (distance in palm sizes, log palm size, centre in pixels)
        self.apart = 0.0
        self.zoom = 0.0
        self.move = np.zeros(2)     # velocity of the point between the hands, palm sizes / s
        self.beam = None
        self.cool_until = 0.0
        self.shake_until = 0.0
        self.prompt = None
        self.ps.reset()

    def render(self, frame, infos, clicks, t: float, k: float):
        h, w = frame.shape[:2]
        dt = float(np.clip(t - self.last_t, 0.005, 0.2)) if self.last_t is not None else 1 / 30
        self.last_t = t
        layer = np.zeros_like(frame)

        for hd in infos:  # flames around every hand, bigger as the ball charges
            n = int(round(2 + 3 * self.c + 4 * hd.open))
            flame = mix_col(hd.color, (30, 160, 255))
            self.ps.emit(hd.palm, n, flame, speed=(0.3, 2.2), decay=(0.04, 0.09), up=3.0,
                         scale=k, jitter=hd.palm_r * 0.45, whiten=0.3)
        self.ps.step(gravity=-0.12 * k, drag=0.03)

        prompt = None
        pair = closest_pair(infos)
        if pair is None:
            self.c = max(0.0, self.c - dt / 1.5)
            self.prev = None
        else:
            a, b = pair
            dist = float(np.hypot(*(a.palm - b.palm)))
            mr = (a.palm_r + b.palm_r) / 2
            centre = (a.palm + b.palm) / 2
            now = (dist / mr, float(np.log(mr)), centre)
            if self.prev is not None:
                self.apart = 0.5 * self.apart + 0.5 * (now[0] - self.prev[0]) / dt
                self.zoom = 0.5 * self.zoom + 0.5 * (now[1] - self.prev[1]) / dt
                self.move = 0.5 * self.move + 0.5 * (now[2] - self.prev[2]) / dt / mr
            self.prev = now
            colour = mix_col(a.color, b.color)
            spread = (a.open + b.open) / 2

            if t >= self.cool_until:
                charging = dist < 3.0 * mr
                self.c = float(np.clip(self.c + (dt / self.CHARGE_S if charging else -dt / 1.5),
                                       0, 1))
            push = float(np.hypot(*self.move))
            if (self.c >= 0.3 and t >= self.cool_until
                    and (self.apart > self.APART_RATE or self.zoom > self.ZOOM_RATE
                         or push > self.PUSH_RATE)):
                self._fire(centre, colour, self._aim(a, b), t, k)
            elif self.c > 0.02:
                self._draw_ball(layer, a, b, centre, colour, spread, t, k)
                if self.c > 0.6:
                    prompt = (centre[0], centre[1] - 150 * k * self.c)

        self.prompt = prompt   # the UI shows "Thrust!" here when the ball is charged
        out = self._draw_shot(frame, layer, t, k)
        if t < self.shake_until:  # short screen shake after firing
            amp = 10 * k * (self.shake_until - t) / self.SHAKE_S
            if self.beam is not None and self.beam["dir"] is None:
                amp *= 1.8
            m = np.float32([[1, 0, self.rng.uniform(-amp, amp)], [0, 1, self.rng.uniform(-amp, amp)]])
            out = cv2.warpAffine(out, m, (w, h), borderMode=cv2.BORDER_REFLECT)
        return out

    def _aim(self, a, b):
        """Unit 2D direction to fire in, or None for 'toward the camera'."""
        push = float(np.hypot(*self.move))
        if push > self.AIM_RATE and push * 1.5 > max(self.apart, 0.0):
            return self.move / push                   # the way both hands were thrust
        if self.zoom > self.ZOOM_RATE:
            return None                               # pushed at the camera
        normals = [n for n in (a.normal, b.normal) if n is not None]
        if normals:                                   # the way the palms face
            n = np.mean(normals, axis=0)
            flat = float(np.hypot(n[0], n[1]))
            if flat > 0.35 and flat > -n[2]:
                return n[:2] / flat
        return None

    def _draw_ball(self, layer, a, b, centre, colour, spread, t, k):
        r = (10 + 75 * self.c) * k * (0.85 + 0.5 * spread) * (1 + 0.08 * np.sin(t * (6 + 10 * spread)))
        ci = (int(centre[0]), int(centre[1]))
        for i, (scale, level) in enumerate(((1.0, 0.18), (0.72, 0.32), (0.48, 0.5))):
            wobble = 1 + 0.06 * np.sin(t * 9 + i * 2)
            cv2.circle(layer, ci, max(1, int(r * scale * wobble)), scale_col(colour, level), -1,
                       cv2.LINE_AA)
        cv2.circle(layer, ci, max(1, int(r * 0.16)), (200, 200, 200), -1, cv2.LINE_AA)
        lightning(layer, a.palm, centre, a.color, self.rng, k, thick=int(3 * k))
        lightning(layer, b.palm, centre, b.color, self.rng, k, thick=int(3 * k))
        ang = t * 3 + np.arange(12) * (2 * np.pi / 12)
        orbit = centre + np.stack([np.cos(ang), np.sin(ang)], 1) * r * 1.25
        plot_points(layer, orbit, np.tile(np.array(colour, float), (12, 1)))

    def _fire(self, centre, colour, direction, t, k):
        """direction: unit 2D vector for a beam, or None for a blast toward the camera."""
        self.beam = {"t0": t, "origin": np.asarray(centre, float), "dir": direction,
                     "c": self.c, "colour": colour}
        n = 110 if direction is not None else 220
        self.ps.emit(centre, n, colour, speed=(3, 13 if direction is not None else 18),
                     decay=(0.015, 0.035), scale=k)
        self.shake_until = t + self.SHAKE_S
        self.cool_until = t + 0.9
        self.c, self.apart, self.zoom, self.move = 0.0, 0.0, 0.0, np.zeros(2)

    @staticmethod
    def _reach(o, d, w, h):
        """Distance from o along d to the edge of the picture."""
        out = []
        for i, size in ((0, w), (1, h)):
            if d[i] > 1e-6:
                out.append((size - o[i]) / d[i])
            elif d[i] < -1e-6:
                out.append(-o[i] / d[i])
        return max(0.0, min(out)) if out else 0.0

    def _draw_shot(self, frame, layer, t, k):
        bm = self.beam
        h, w = frame.shape[:2]
        if bm is not None and bm["dir"] is not None and t - bm["t0"] < self.BEAM_S:
            age = t - bm["t0"]
            fade = 1.0 - max(0.0, (age - 0.3) / (self.BEAM_S - 0.3))
            o, d = bm["origin"], np.asarray(bm["dir"], float)
            length = min(1.0, age / 0.12) * (self._reach(o, d, w, h) + 40 * k)
            e = o + d * length
            side = np.array([-d[1], d[0]])
            half = (14 + 34 * bm["c"]) * k * fade
            # wide coloured beam with a thinner white core, and a flare where it starts
            for scale, colour in ((1.0, scale_col(bm["colour"], 0.5 * fade)),
                                  (0.35, scale_col((255, 255, 255), 0.8 * fade))):
                poly = np.array([o + side * half * scale, e + side * half * 0.75 * scale,
                                 e - side * half * 0.75 * scale, o - side * half * scale], np.int32)
                cv2.fillConvexPoly(layer, poly, col(colour))
            oi = (int(o[0]), int(o[1]))
            cv2.circle(layer, oi, max(2, int(half * 1.1)), scale_col(bm["colour"], 0.5 * fade), -1,
                       cv2.LINE_AA)
            cv2.circle(layer, oi, max(2, int(half * 0.45)), scale_col((255, 255, 255), 0.8 * fade), -1,
                       cv2.LINE_AA)
            self.ps.emit(e, 8, bm["colour"], speed=(2, 8), decay=(0.03, 0.06), scale=k)
        elif bm is not None and bm["dir"] is None and t - bm["t0"] < self.BLAST_S:
            # toward the camera: the ball rushes at the viewer and bursts over the screen
            age = (t - bm["t0"]) / self.BLAST_S
            o = (int(bm["origin"][0]), int(bm["origin"][1]))
            grow = min(1.0, age / 0.35) ** 0.7
            r = (40 + grow * 0.9 * float(np.hypot(w, h))) * (0.6 + 0.4 * bm["c"])
            fade = 1.0 - max(0.0, (age - 0.3) / 0.7)
            cv2.circle(layer, o, int(r), scale_col(bm["colour"], 0.35 * fade), -1, cv2.LINE_AA)
            cv2.circle(layer, o, int(r), scale_col((255, 255, 255), 0.7 * fade),
                       max(2, int(18 * k * fade)), cv2.LINE_AA)
            cv2.circle(layer, o, max(2, int(r * 0.35 * (1 - age))), scale_col((255, 255, 255), 0.8 * fade),
                       -1, cv2.LINE_AA)
        else:
            self.beam = None
        self.ps.draw(layer)
        return composite_glow(frame, layer, dim=0.7, gain=1.0)


class Storm:
    """Rainbow particles that fall like snow and get pushed around by your hands."""

    N = 320

    def __init__(self):
        self.rng = np.random.default_rng()
        self.reset()

    def reset(self):
        self.pos = None

    def _init(self, w, h):
        self.pos = self.rng.uniform([0, 0], [w, h], (self.N, 2))
        self.vel = self.rng.normal(0, 0.3, (self.N, 2))
        hue = self.rng.integers(0, 180, self.N).astype(np.uint8)
        hsv = np.stack([hue, np.full(self.N, 210, np.uint8), np.full(self.N, 255, np.uint8)],
                       axis=1).reshape(-1, 1, 3)
        self.base = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(-1, 3).astype(float) * 0.95
        self.col = self.base.copy()
        self._shape = (w, h)

    _shape = None

    def render(self, frame, infos, k: float):
        h, w = frame.shape[:2]
        if self.pos is None or self._shape != (w, h):
            self._init(w, h)
        self.vel[:, 1] += 0.05 * k
        self.vel *= 0.97
        reach = 110 * k
        for hd in infos:
            for p in (hd.mid, hd.palm):
                d = self.pos - p
                dist = np.hypot(d[:, 0], d[:, 1])
                m = dist < reach
                if not m.any():
                    continue
                f = (1 - dist[m] / reach)[:, None]
                push = d[m] / (dist[m, None] + 1e-3) * f * (1.2 + 2.0 * hd.open) * k
                self.vel[m] += push + hd.vel * 0.25 * f
                self.col[m] = self.col[m] * 0.8 + np.array(hd.color, float) * 0.2
        self.pos += self.vel
        self.col += (self.base - self.col) * 0.02
        self.pos[:, 0] %= w
        fell = self.pos[:, 1] > h
        self.pos[fell, 1] = -2
        self.pos[fell, 0] = self.rng.uniform(0, w, fell.sum())
        self.vel[fell] = 0
        self.pos[:, 1] = np.maximum(self.pos[:, 1], -50)
        layer = np.zeros_like(frame)
        plot_points(layer, self.pos, self.col)
        return composite_glow(frame, layer, dim=0.6, gain=1.5)


class FireHands:
    """Every hand is on fire. Spreading the fingers makes the flames bigger. A quick
    flick or push of a hand throws a fireball in that direction; it flies off, fades
    and leaves a few embers behind."""

    FLICK = 11.0        # palm sizes per second that count as a throw
    TELEPORT = 70.0     # faster than this is a tracking jump, not a throw
    COOLDOWN_S = 0.45
    LIFE_S = 1.1
    MAX_BALLS = 8
    FIRE = ((0, 70, 230), (0, 140, 255), (40, 210, 255))  # BGR: red, orange, yellow

    def __init__(self):
        self.rng = np.random.default_rng()
        self.flames = ParticleSystem(900)
        self.embers = ParticleSystem(500)
        self.reset()

    def reset(self):
        self.flames.reset()
        self.embers.reset()
        self.balls = []      # dicts: pos, vel (px/s), r, t0
        self.track = {}      # hand id -> {"pos", "vel", "frames", "cool"}
        self.last_t = None

    def _flick(self, h, t, dt):
        """Smoothed palm velocity in px/s; returns a throw velocity or None."""
        tr = self.track.get(h.id)
        if tr is None:
            self.track[h.id] = {"pos": h.palm.copy(), "vel": np.zeros(2), "frames": 0, "cool": 0.0,
                                "seen": t}
            return None
        v = (h.palm - tr["pos"]) / dt
        tr["pos"], tr["seen"] = h.palm.copy(), t
        tr["frames"] += 1
        speed_now = float(np.hypot(*v)) / h.palm_r
        if speed_now > self.TELEPORT:      # id swap or tracker jump: ignore this frame
            tr["vel"] = np.zeros(2)
            return None
        tr["vel"] = 0.5 * tr["vel"] + 0.5 * v
        speed = float(np.hypot(*tr["vel"])) / h.palm_r
        if tr["frames"] >= 3 and speed > self.FLICK and t >= tr["cool"]:
            tr["cool"] = t + self.COOLDOWN_S
            return tr["vel"].copy()
        return None

    def render(self, frame, infos, t: float, k: float):
        h_, w = frame.shape[:2]
        dt = float(np.clip(t - self.last_t, 0.005, 0.2)) if self.last_t is not None else 1 / 30
        self.last_t = t
        for hid in [i for i, tr in self.track.items() if t - tr["seen"] > 0.5]:
            del self.track[hid]

        for h in infos:
            # flames: more and taller with spread fingers, rising from the fingertips and palm
            size = 0.35 + 0.65 * h.spread
            for src, n in ((h.tips, 2 + int(4 * size)), (h.palm[None, :], 4 + int(10 * size))):
                for p in src:
                    self.flames.emit(p, n, self.FIRE[1], speed=(0.1, 0.6 + 0.8 * size),
                                     decay=(0.07, 0.13 - 0.05 * size), up=1.2 + 2.8 * size,
                                     scale=k, jitter=h.palm_r * 0.08 * (1 + size), whiten=0)
            throw = self._flick(h, t, dt)
            if throw is not None and len(self.balls) < self.MAX_BALLS:
                d = throw / (np.hypot(*throw) + 1e-6)
                speed = max(float(np.hypot(*throw)) * 1.4, 900 * k)
                start = h.palm + d * h.palm_r * 0.8
                self.balls.append({"pos": start, "vel": d * speed, "t0": t,
                                   "r": h.palm_r * (0.45 + 0.35 * h.spread)})
                self.flames.emit(start, 40, self.FIRE[1], speed=(2, 7), decay=(0.04, 0.08),
                                 scale=k)

        layer = np.zeros_like(frame)
        alive = []
        for b in self.balls:
            age = t - b["t0"]
            b["pos"] = b["pos"] + b["vel"] * dt
            x, y = b["pos"]
            off = x < -b["r"] or x > w + b["r"] or y < -b["r"] or y > h_ + b["r"]
            if age > self.LIFE_S or off:
                if not off:  # burnt out on screen: a few embers fall from it
                    self.embers.emit(b["pos"], 14, self.FIRE[1], speed=(0.5, 3), decay=(0.012, 0.025),
                                     scale=k, whiten=0.2)
                continue
            alive.append(b)
            fade = 1 - max(0.0, age - 0.5) / (self.LIFE_S - 0.5)
            r = b["r"] * (0.7 + 0.3 * fade)
            ci = (int(x), int(y))
            # trail of flame behind it, and the odd ember dropping off
            self.flames.emit(b["pos"], 6, self.FIRE[int(self.rng.integers(0, 3))], speed=(0.3, 2),
                             decay=(0.07, 0.12), scale=k, jitter=r * 0.4, whiten=0.2)
            if self.rng.random() < 0.3:
                self.embers.emit(b["pos"], 1, self.FIRE[2], speed=(0.3, 1.5), decay=(0.015, 0.03),
                                 scale=k, whiten=0.1)
            for scale, c, lvl in ((1.0, self.FIRE[0], 0.55), (0.7, self.FIRE[1], 0.75),
                                  (0.4, self.FIRE[2], 0.9), (0.18, (255, 255, 255), 0.9)):
                cv2.circle(layer, ci, max(1, int(r * scale)), scale_col(c, lvl * fade), -1,
                           cv2.LINE_AA)
        self.balls = alive

        self.flames.step(gravity=-0.08 * k, drag=0.04)
        self.embers.step(gravity=0.10 * k, drag=0.02)
        self._draw_flames(layer)
        self.embers.draw(layer)
        return composite_glow(frame, layer, dim=0.8, gain=1.3)

    def _draw_flames(self, layer):
        """Flames as soft blobs: particles are plotted at quarter size, blurred and
        scaled up, which is cheap and looks like fire rather than sparks. A young
        particle is yellow and hot, an old one dark red."""
        fl = self.flames
        if len(fl.life) == 0:
            return
        h, w = layer.shape[:2]
        small = np.zeros((max(1, h // 4), max(1, w // 4), 3), np.uint8)
        life = fl.life[:, None]
        red, yellow = np.array(self.FIRE[0], np.float32), np.array(self.FIRE[2], np.float32)
        colours = (red + (yellow - red) * life ** 2) * np.minimum(1.0, life * 1.6) * 0.6
        plot_points(small, fl.pos / 4, colours)
        small = cv2.GaussianBlur(small, (3, 3), 0)
        big = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        cv2.add(layer, big, dst=layer)


class TimeEcho:
    """Ghost copies of whatever moves trail behind it, tinted in the nearest hand's colour.
    Pinch and let go to freeze a ghost of the moment in place for a few seconds.

    A slowly learned background tells what is moving. Each stored frame keeps a small
    mask of its moving parts; older frames are painted under the live picture, fainter
    the older they are, and never over a moving part of the live picture, so you stay in
    front of your own ghosts. Masks are worked out at quarter size, so it stays cheap.
    """

    DELAYS = (4, 8, 12, 16, 20)              # frames back
    ALPHAS = (0.72, 0.58, 0.45, 0.33, 0.22)
    FREEZE_S, MAX_FROZEN = 4.0, 3
    TINT = (240, 201, 76)                    # with no hands in view

    def __init__(self):
        self.reset()

    def reset(self):
        self._hist = deque(maxlen=max(self.DELAYS) + 1)   # (frame, 1/4 frame, 1/4 moving mask)
        self._bg = None
        self._frozen = []                                 # (frame, mask, colour, time)

    def _moving(self, small):
        """Mask (uint8 0..255, soft edges) of what differs from the learned background."""
        diff = cv2.absdiff(small, cv2.convertScaleAbs(self._bg)).max(axis=2)
        m = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)[1]
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        m = cv2.dilate(m, np.ones((3, 3), np.uint8))
        return cv2.GaussianBlur(m, (5, 5), 0)

    @staticmethod
    def _differs(a, b):
        """0..1 at 1/4 size: where an old frame differs from the live one (a ghost of
        something that has not moved since would only tint it)."""
        d = cv2.absdiff(a, b).max(axis=2)
        return np.clip((d.astype(np.float32) - 14) / 20, 0, 1)

    @staticmethod
    def _tint_map(infos, sw, sh, w, h, default):
        """Quarter-size picture where every pixel has the colour of the nearest hand."""
        if not infos:
            return np.full((sh, sw, 3), default, np.uint8)
        ys, xs = np.mgrid[0:sh, 0:sw]
        sx, sy = w / sw, h / sh
        d = np.stack([(xs * sx - p.palm[0]) ** 2 + (ys * sy - p.palm[1]) ** 2 for p in infos])
        cols = np.array([col(p.color) for p in infos], np.uint8)
        return cols[np.argmin(d, axis=0)]

    @staticmethod
    def _paint(out, ghost, mask_f, tint, w, h):
        """Blend a tinted ghost into out, only inside the box around its mask. mask_f and
        tint are quarter size; only the box is scaled up."""
        x, y, bw, bh = cv2.boundingRect((mask_f > 0.03).astype(np.uint8))
        if bw == 0:
            return out
        sh, sw = mask_f.shape
        sx0, sy0, sx1, sy1 = max(0, x - 1), max(0, y - 1), min(sw, x + bw + 1), min(sh, y + bh + 1)
        x0, y0, x1, y1 = sx0 * w // sw, sy0 * h // sh, sx1 * w // sw, sy1 * h // sh
        if sx1 == sw:
            x1 = w
        if sy1 == sh:
            y1 = h
        size = (x1 - x0, y1 - y0)
        m = cv2.resize(mask_f[sy0:sy1, sx0:sx1], size, interpolation=cv2.INTER_LINEAR)
        tn = cv2.resize(tint[sy0:sy1, sx0:sx1], size, interpolation=cv2.INTER_LINEAR)
        tinted = cv2.addWeighted(ghost[y0:y1, x0:x1], 0.6, tn, 0.55, 0)
        out[y0:y1, x0:x1] = cv2.blendLinear(tinted, out[y0:y1, x0:x1], m, 1.0 - m)
        return out

    def render(self, frame, infos, clicks, t: float, k: float):
        h, w = frame.shape[:2]
        sw, sh = max(1, w // 4), max(1, h // 4)
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        if self._bg is None or self._bg.shape[:2] != (sh, sw):
            self._bg = small.astype(np.float32)
            self._hist.clear()
        moving = self._moving(small)
        # learn the background: fast where nothing moves, slower under moving parts
        cv2.accumulateWeighted(small, self._bg, 0.05, mask=cv2.bitwise_not(moving))
        cv2.accumulateWeighted(small, self._bg, 0.02)   # stale spots clear in about 2 s

        here = 1.0 - moving.astype(np.float32) / 255   # 0 where the live picture moves
        tint = self._tint_map(infos, sw, sh, w, h, self.TINT)
        out = frame.copy()
        n = len(self._hist)
        for delay, alpha in sorted(zip(self.DELAYS, self.ALPHAS), reverse=True):  # oldest first
            if n > delay:
                ghost, gsmall, gmask = self._hist[n - 1 - delay]
                m = gmask.astype(np.float32) * (alpha / 255) * here * self._differs(gsmall, small)
                out = self._paint(out, ghost, m, tint, w, h)

        byid = {i.id: i for i in infos}
        for cid in clicks:          # pinch released: freeze this moment
            if cid in byid and moving.max() > 0:
                self._frozen.append((frame.copy(), small.copy(), moving.copy(), col(byid[cid].color), t))
        self._frozen = [f for f in self._frozen if t - f[4] < self.FREEZE_S][-self.MAX_FROZEN:]
        for img, fsmall, fmask, colour, t0 in self._frozen:
            fade = 0.85 * (1 - (t - t0) / self.FREEZE_S)
            m = fmask.astype(np.float32) * (fade / 255) * here * self._differs(fsmall, small)
            out = self._paint(out, img, m, np.full((sh, sw, 3), colour, np.uint8), w, h)

        self._hist.append((frame.copy(), small, moving))
        return out
