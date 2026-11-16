"""Portable build helpers: camera choice, settings.json, writable-folder check."""
from gesture_fx import cameras, paths, settings
from gesture_fx.cameras import CameraInfo
from gesture_fx.config import parse_args

CAMS = [CameraInfo(0, "Integrated Webcam"), CameraInfo(1, "Logitech C920"),
        CameraInfo(2, "Logitech C920")]


def test_saved_camera_is_found_by_name_when_its_index_moved():
    assert cameras.choose_saved(CAMS[:2], {"index": 0, "name": "Logitech C920"}) == CAMS[1]


def test_saved_index_breaks_ties_between_cameras_with_the_same_name():
    assert cameras.choose_saved(CAMS, {"index": 2, "name": "Logitech C920"}) == CAMS[2]
    assert cameras.choose_saved(CAMS, {"index": 7, "name": "Logitech C920"}) == CAMS[1]


def test_missing_saved_camera_gives_none_so_the_picker_shows():
    assert cameras.choose_saved(CAMS, {"index": 1, "name": "Razer Kiyo"}) is None
    assert cameras.choose_saved(CAMS, None) is None
    assert cameras.choose_saved(CAMS, "junk") is None
    assert cameras.choose_saved([], {"index": 0, "name": "Integrated Webcam"}) is None


def test_saved_camera_without_a_name_matches_by_index():
    assert cameras.choose_saved(CAMS, {"index": 1}) == CAMS[1]
    assert cameras.choose_saved(CAMS, {"index": 5}) is None


def test_next_camera_wraps_around():
    assert cameras.next_camera(CAMS, 0) == CAMS[1]
    assert cameras.next_camera(CAMS, 2) == CAMS[0]
    assert cameras.next_camera(CAMS, 9) == CAMS[0]
    assert cameras.next_camera([], 0) is None


def test_settings_round_trip_and_bad_file(tmp_path):
    path = tmp_path / "settings.json"
    assert settings.load(path) == {}
    assert settings.update(path, camera=CAMS[1].as_setting())
    assert settings.update(path, other=1)
    assert settings.load(path) == {"camera": {"index": 1, "name": "Logitech C920"}, "other": 1}
    path.write_text("{not json", encoding="utf-8")
    assert settings.load(path) == {}
    path.write_text("[1, 2]", encoding="utf-8")
    assert settings.load(path) == {}


def test_is_writable(tmp_path):
    assert paths.is_writable(tmp_path)
    assert list(tmp_path.iterdir()) == []          # the probe file is removed again
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert not paths.is_writable(blocker / "sub")  # a file where a folder should be


def test_camera_flag_is_an_override_and_defaults_to_none():
    assert parse_args([]).camera is None
    assert parse_args(["--camera", "2"]).camera == 2


def test_read_only_app_folder_falls_back_to_temp(tmp_path, monkeypatch):
    fallback = tmp_path / "GestureFX"
    monkeypatch.setattr(paths, "FALLBACK_DIR", fallback)
    monkeypatch.setattr(paths, "is_writable", lambda folder: False)
    assert paths._pick_data_dir() == (fallback, True)
    assert fallback.is_dir()
    monkeypatch.setattr(paths, "is_writable", lambda folder: True)
    assert paths._pick_data_dir() == (paths.APP_DIR, False)
