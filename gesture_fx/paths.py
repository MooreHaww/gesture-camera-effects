"""Where the app finds its bundled files and where it writes its own.

From source (a PC or the Raspberry Pi) both are the project folder, as before.

In the portable Windows build (PyInstaller, one-folder) the bundled files (model,
fonts, MediaPipe data) sit in the build's _internal folder, and everything the app
writes (photos/, debug/, crash.log, settings.json) goes next to GestureFX.exe, so a
copy on a USB stick keeps its photos and its camera choice with it. Nothing depends on
the working directory. When that folder is read-only (a locked stick, a CD, Program
Files) the app writes to a GestureFX folder in the temp directory instead and says so.
"""
import os
import sys
import tempfile
from pathlib import Path

FROZEN = bool(getattr(sys, "frozen", False))
SOURCE_ROOT = Path(__file__).resolve().parent.parent

# Read-only files that ship with the app
RESOURCE_DIR = Path(getattr(sys, "_MEIPASS", SOURCE_ROOT)) if FROZEN else SOURCE_ROOT
# The folder the user sees: next to GestureFX.exe, or the project folder
APP_DIR = Path(sys.executable).resolve().parent if FROZEN else SOURCE_ROOT
FALLBACK_DIR = Path(tempfile.gettempdir()) / "GestureFX"


def is_writable(folder: Path) -> bool:
    """True when a file can really be created in folder (a read-only USB stick or a
    folder without write permission both fail here, whatever the attributes say)."""
    probe = folder / f".write_test_{os.getpid()}"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        with open(probe, "w") as f:
            f.write("ok")
        probe.unlink()
        return True
    except OSError:
        return False


def _pick_data_dir() -> tuple[Path, bool]:
    if is_writable(APP_DIR):
        return APP_DIR, False
    FALLBACK_DIR.mkdir(parents=True, exist_ok=True)
    return FALLBACK_DIR, True


# Where photos/, debug/, crash.log and settings.json go. USED_FALLBACK is True when the
# app folder was read-only and the temp folder is used instead.
DATA_DIR, USED_FALLBACK = _pick_data_dir()


def fallback_message() -> str | None:
    if not USED_FALLBACK:
        return None
    return (f"The folder {APP_DIR} is read-only, so photos and settings are saved in "
            f"{DATA_DIR} instead.")
