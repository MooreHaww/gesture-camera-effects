"""Crash and exit logging to crash.log, so a crash can be diagnosed after the fact.

Covers three ways the app can stop:
- a Python exception (full traceback),
- a native crash inside OpenCV or MediaPipe (faulthandler dumps every thread's stack),
- a normal exit for an unexpected reason (the reason is written as "Stopped: ...").
"""
import faulthandler
import platform
import sys
import threading
import traceback
from datetime import datetime

from .config import ROOT

LOG_PATH = ROOT / "crash.log"
_fh = None
_seen: dict = {}


def install():
    """Open the log, hook native crashes and uncaught thread exceptions."""
    global _fh
    try:
        _fh = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
    except OSError:
        return
    faulthandler.enable(file=_fh, all_threads=True)

    def thread_hook(args):
        name = args.thread.name if args.thread else "?"
        write(f"Uncaught exception in thread {name}",
              (args.exc_type, args.exc_value, args.exc_traceback))

    threading.excepthook = thread_hook


def versions() -> str:
    import cv2
    import mediapipe
    import numpy
    return (f"python {platform.python_version()}, {platform.platform()}, "
            f"opencv {cv2.__version__}, mediapipe {mediapipe.__version__}, numpy {numpy.__version__}")


def write(title: str, exc_info=None, context: dict | None = None):
    lines = [f"\n=== {datetime.now():%Y-%m-%d %H:%M:%S} {title} ==="]
    if context:
        lines.append("context: " + ", ".join(f"{k}={v}" for k, v in context.items()))
    if exc_info:
        lines.append("".join(traceback.format_exception(*exc_info)).rstrip())
    text = "\n".join(lines) + "\n"
    try:
        if _fh is not None:
            _fh.write(text)
            _fh.flush()
        else:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(text)
    except OSError:
        pass


def write_once(key, title: str, context: dict | None = None) -> bool:
    """Log the current exception only the first time this key is seen (avoids a log flood).
    Returns True when it was written."""
    _seen[key] = _seen.get(key, 0) + 1
    if _seen[key] > 1:
        return False
    write(title, sys.exc_info(), context)
    return True
