"""Records frames and writes an animated GIF (encoding happens after capture)."""
from pathlib import Path

import cv2
from PIL import Image


class GifRecorder:
    def __init__(self, path: str, seconds: float = 8.0, fps: float = 12.0,
                 width: int = 480, countdown: float = 3.0):
        self.path = Path(path)
        self.seconds, self.fps, self.width, self.countdown = seconds, fps, width, countdown
        self._start = None
        self._last_capture = -1e9
        self._frames: list[Image.Image] = []

    def update(self, frame, now: float) -> str:
        """Feed each displayed frame. Returns 'countdown:N', 'recording' or 'done'."""
        if self._start is None:
            self._start = now
        elapsed = now - self._start
        if elapsed < self.countdown:
            return f"countdown:{int(self.countdown - elapsed) + 1}"
        if elapsed >= self.countdown + self.seconds:
            return "done"
        if now - self._last_capture >= 1.0 / self.fps:
            self._last_capture = now
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (self.width, int(h * self.width / w)),
                               interpolation=cv2.INTER_AREA)
            self._frames.append(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        return "recording"

    def save(self) -> Path:
        if not self._frames:
            raise RuntimeError("no frames recorded")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        frames = [f.convert("P", palette=Image.Palette.ADAPTIVE, colors=128)
                  for f in self._frames]
        frames[0].save(self.path, save_all=True, append_images=frames[1:],
                       duration=int(1000 / self.fps), loop=0, optimize=True)
        return self.path
