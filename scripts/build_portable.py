"""Build the portable Windows folder: dist/GestureFX/GestureFX.exe.

    .venv\\Scripts\\python scripts\\build_portable.py

PyInstaller one-folder build, so it starts fast and nothing is unpacked to the temp
folder. Bundled inside: the hand landmarker model (downloaded first if models/ is
empty), the Inter fonts, and MediaPipe's data files and DLL. The launchers and
README_PORTABLE.txt from scripts/portable/ are copied next to the exe. Copy the whole
dist/GestureFX folder to a USB stick; the other laptop needs no Python and no install.
"""
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

NAME = "GestureFX"
DIST = ROOT / "dist"
OUT = DIST / NAME
WORK = ROOT / "build"
MODEL = ROOT / "models" / "hand_landmarker.task"
FONTS = ROOT / "gesture_fx" / "assets" / "fonts"
LAUNCHERS = ROOT / "scripts" / "portable"
# Not used by the app; leaving them out keeps the folder smaller
EXCLUDE = ["tkinter", "_tkinter", "pytest", "_pytest", "pyflakes", "PyInstaller", "IPython"]


def folder_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def copy_launchers():
    """Copy Start.bat, Start_Kiosk.bat and README_PORTABLE.txt with Windows line
    endings (cmd.exe misreads .bat files with Unix line endings)."""
    for src in sorted(LAUNCHERS.iterdir()):
        text = src.read_text(encoding="utf-8").replace("\r\n", "\n")
        (OUT / src.name).write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
        print(f"  {src.name}")


def main() -> int:
    if sys.platform != "win32":
        print("The portable build is for Windows; run this on Windows.", file=sys.stderr)
        return 1
    try:
        import PyInstaller.__main__
    except ImportError:
        print("PyInstaller is missing: pip install -r requirements-dev.txt", file=sys.stderr)
        return 1

    if not MODEL.exists():
        import download_model
        download_model.main()

    t0 = time.perf_counter()
    sep = os.pathsep
    args = [
        str(ROOT / "scripts" / "gesturefx_entry.py"),
        "--name", NAME,
        "--onedir",
        "--windowed",            # no console window behind the app
        "--noconfirm", "--clean",
        "--distpath", str(DIST),
        "--workpath", str(WORK),
        "--specpath", str(WORK),
        "--paths", str(ROOT),
        "--add-data", f"{MODEL}{sep}models",
        "--add-data", f"{FONTS}{sep}gesture_fx/assets/fonts",
        "--collect-data", "mediapipe",
        "--collect-binaries", "mediapipe",
        "--log-level", "WARN",
    ]
    for mod in EXCLUDE:
        args += ["--exclude-module", mod]
    PyInstaller.__main__.run(args)

    exe = OUT / f"{NAME}.exe"
    internal = OUT / "_internal"
    checks = {
        "exe": exe,
        "model": internal / "models" / "hand_landmarker.task",
        "Inter-Bold": internal / "gesture_fx" / "assets" / "fonts" / "Inter-Bold.ttf",
        "Inter-Regular": internal / "gesture_fx" / "assets" / "fonts" / "Inter-Regular.ttf",
    }
    missing = [k for k, p in checks.items() if not p.exists()]
    if not list(internal.rglob("libmediapipe*.dll")):
        missing.append("MediaPipe DLL")
    if missing:
        print(f"Build incomplete, missing: {', '.join(missing)}", file=sys.stderr)
        return 1

    print("Launchers:")
    copy_launchers()
    # Files a test run of the build may have left (the folder must start clean)
    for leftover in ("crash.log", "settings.json"):
        (OUT / leftover).unlink(missing_ok=True)
    for leftover in ("photos", "debug"):
        shutil.rmtree(OUT / leftover, ignore_errors=True)

    print(f"\nBuilt {exe} in {time.perf_counter() - t0:.0f} s, "
          f"folder {folder_size(OUT) / 1e6:.0f} MB")
    print(f"Copy the whole folder {OUT} to the USB stick.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
