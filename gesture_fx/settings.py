"""settings.json next to the app: remembers the chosen camera between runs."""
import json
from pathlib import Path

from .paths import DATA_DIR

PATH = DATA_DIR / "settings.json"


def load(path: Path = PATH) -> dict:
    """The saved settings, or {} when the file is missing or unreadable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save(data: dict, path: Path = PATH) -> bool:
    """Write the settings (via a temp file, so a crash cannot leave half a file)."""
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(path)
        return True
    except OSError:
        return False


def update(path: Path = PATH, **values) -> bool:
    data = load(path)
    data.update(values)
    return save(data, path)
