"""Photo booth: countdown, flash, save to a git-ignored folder.

Photos contain people's faces, so they go only to photos/ (listed in .gitignore).
"""
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2

COUNTDOWN_S = 3.0
FLASH_S = 0.35
SAVED_S = 2.5


@dataclass
class BoothState:
    countdown: int | None = None   # 3, 2, 1 while counting down
    countdown_frac: float = 1.0    # what is left of the current second, 1 .. 0
    capture: bool = False          # True on the one frame that should be saved
    flash: float = 0.0             # 0..1 white flash strength
    message: str | None = None     # set for a couple of seconds after a photo is saved


class PhotoBooth:
    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self._start = None
        self._shot = None
        self._saved_name = None

    @property
    def busy(self) -> bool:
        return self._start is not None

    def start(self, now: float) -> bool:
        if self.busy:
            return False
        self._start = now
        return True

    def update(self, now: float) -> BoothState:
        st = BoothState()
        if self._start is not None:
            elapsed = now - self._start
            if elapsed < COUNTDOWN_S:
                left = COUNTDOWN_S - elapsed
                st.countdown = math.ceil(left)
                st.countdown_frac = left - (st.countdown - 1)
            else:
                st.capture = True
                self._shot = now
                self._start = None
        if self._shot is not None:
            since = now - self._shot
            if since < FLASH_S:
                st.flash = 1.0 - since / FLASH_S
            if since < SAVED_S and self._saved_name:
                st.message = f"Photo saved: {self._saved_name}"
            elif since >= SAVED_S:
                self._shot = None
        return st

    def save(self, image) -> Path:
        self.folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("photo_%Y%m%d_%H%M%S")
        path = self.folder / f"{stamp}.jpg"
        n = 1
        while path.exists():
            path = self.folder / f"{stamp}_{n}.jpg"
            n += 1
        if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"could not write {path}")
        self._saved_name = path.name
        return path
