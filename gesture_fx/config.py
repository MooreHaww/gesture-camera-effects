"""Configuration and command-line flags."""
import argparse
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models" / "hand_landmarker.task"


@dataclass
class Config:
    camera: int = 0
    width: int = 640
    height: int = 480
    mirror: bool = True
    model_path: Path = DEFAULT_MODEL
    record_gif: str | None = None
    gif_seconds: float = 8.0
    benchmark: float | None = None
    no_display: bool = False
    max_hands: int = 4
    kiosk: bool = False
    photo_dir: Path = ROOT / "photos"
    attract_after: float = 6.0
    clear_hold_s: float = 0.4
    # Gesture thresholds (pinch ratio = thumb-index distance / palm size)
    pinch_closed: float = 0.2
    pinch_open: float = 1.1
    click_close: float = 0.35
    click_release: float = 0.6
    switch_hold_s: float = 1.5   # fingertip dwell time on the NEXT / PREVIOUS buttons
    square_pad: bool = True      # pad frames to a square before hand tracking
    brand: str | None = None     # event name shown in a corner (--brand)
    logo: Path | None = None     # small logo image next to it (--logo)
    start_mode: int = 1          # 1-10, as on the keyboard (10 = key 0)
    peace_hold_s: float = 1.5
    swap_hands: bool = False
    spread_closed: float = 0.8
    spread_open: float = 1.8
    smooth_alpha: float = 0.5


def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser(prog="gesture_fx", description="Gesture Camera Effects")
    p.add_argument("--camera", type=int, default=0, help="camera index")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--no-mirror", action="store_true", help="do not flip the image")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--record-gif", metavar="PATH", help="record a demo GIF and exit")
    p.add_argument("--gif-seconds", type=float, default=8.0)
    p.add_argument("--benchmark", type=float, metavar="SECONDS",
                   help="run for N seconds, print average FPS, exit")
    p.add_argument("--no-display", action="store_true",
                   help="do not open a window (useful with --benchmark)")
    p.add_argument("--max-hands", type=int, default=4,
                   help="most hands tracked at once (more hands cost more CPU)")
    p.add_argument("--kiosk", action="store_true",
                   help="event mode: fullscreen, hidden cursor, large hint text")
    p.add_argument("--photo-dir", type=Path, default=ROOT / "photos",
                   help="where photos are saved (keep it out of git)")
    p.add_argument("--swap-hands", action="store_true",
                   help="swap Left/Right if the left-hand and right-hand pinch are reversed")
    p.add_argument("--mode", type=int, default=1, choices=range(1, 11), metavar="1-10",
                   help="mode to start in (10 = the mode on key 0)")
    p.add_argument("--switch-hold", type=float, default=1.5, metavar="SECONDS",
                   help="how long an index fingertip must stay on the NEXT/PREVIOUS button")
    p.add_argument("--no-square-pad", action="store_true",
                   help="track on the raw frame instead of one padded to a square")
    p.add_argument("--brand", nargs="?", const=None, metavar="NAME",
                   help='show this name in a corner, e.g. --brand "Some Club" (no name: nothing shown)')
    p.add_argument("--logo", type=Path, metavar="IMAGE",
                   help="small logo image (PNG with transparency works best) shown with --brand")
    p.add_argument("--attract-after", type=float, default=6.0, metavar="SECONDS",
                   help="show the how-to screen after this long without a hand")
    a = p.parse_args(argv)
    return Config(camera=a.camera, width=a.width, height=a.height,
                  mirror=not a.no_mirror, model_path=a.model,
                  record_gif=a.record_gif, gif_seconds=a.gif_seconds,
                  benchmark=a.benchmark, no_display=a.no_display,
                  max_hands=a.max_hands, kiosk=a.kiosk, swap_hands=a.swap_hands, photo_dir=a.photo_dir,
                  attract_after=a.attract_after, start_mode=a.mode,
                  switch_hold_s=a.switch_hold, square_pad=not a.no_square_pad,
                  brand=(a.brand or "").strip() or None, logo=a.logo)
