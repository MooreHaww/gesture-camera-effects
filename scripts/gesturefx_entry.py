"""Entry point for the portable Windows build (PyInstaller needs a plain script;
gesture_fx/__main__.py uses a relative import). From source, use python -m gesture_fx."""
from gesture_fx.app import main

if __name__ == "__main__":
    raise SystemExit(main())
