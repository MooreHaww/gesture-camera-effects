"""Main loop: camera -> tracking -> players -> mode -> overlay -> window.

Hand tracking runs on its own thread (tracking.TrackerThread). Each camera frame is
handed to it, and the loop waits at most --track-wait ms for that frame's hands: a PC
tracks in about 8 ms, so every frame gets its own hands exactly as before; a Raspberry
Pi needs about 60 ms, so it does not wait, and players.HandPredictor carries the hands
forward between results while the picture and effects keep the camera's frame rate.
"""
import os
import sys
import time
import traceback
from collections import defaultdict

import cv2
import numpy as np

from . import buttons as nav
from . import cameras, camscreen, crashlog, gestures, modes, overlay, paths, settings
from .camera import Camera, CameraError
from .cameras import CameraInfo
from .config import ROOT, Config, parse_args
from .display import Display
from .effects import STYLES
from .photobooth import PhotoBooth
from .players import HandPredictor, Players
from .recorder import GifRecorder
from .tracking import HandTracker, TrackerThread
from .ui import UI, NavView, UIState

WINDOW = "Gesture Camera Effects"
BENCH_WARMUP = 1.0
INFO_OPEN_S = 8.0   # the info card closes itself after this long
CUE_S = 0.6
CAM_FIRST_FRAME_S = 3.0   # a camera that opens but sends nothing for this long is busy
WINDOWS = sys.platform == "win32"

# What the loop is doing right now; written into crash.log next to any error.
CONTEXT: dict = {}


def save_debug(frame, infos, canvas):
    """Write debug/landmarks_<time>.png (the tracked frame, no effects, with every
    landmark) and debug/window_<time>.png (exactly what the window showed)."""
    folder = ROOT / "debug"
    folder.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    img = frame.copy()
    overlay.draw_hands(img, infos, True)
    for h in infos:
        for p in h.lm:
            cv2.drawMarker(img, (int(round(p[0])), int(round(p[1]))), (0, 0, 255),
                           cv2.MARKER_CROSS, 8, 1)
    cv2.imwrite(str(folder / f"landmarks_{stamp}.png"), img)
    cv2.imwrite(str(folder / f"window_{stamp}.png"), canvas)
    print(f"Saved debug/landmarks_{stamp}.png and debug/window_{stamp}.png "
          f"(frame {frame.shape[1]}x{frame.shape[0]}, window {canvas.shape[1]}x{canvas.shape[0]})")


def try_open(info: CameraInfo, cfg: Config) -> Camera | None:
    """The camera, open and delivering frames, or None (missing, busy, blocked)."""
    try:
        cam = Camera(info.index, cfg.width, cfg.height)
    except CameraError:
        return None
    if cam.wait_first(CAM_FIRST_FRAME_S):
        return cam
    cam.release()
    return None


def _camera_problem(info: CameraInfo | None) -> str:
    what = f"Camera {info.index} ({info.name})" if info else "No camera found"
    if not WINDOWS:
        return f"{what}: could not open it, or it gives no picture."
    tips = camscreen.BUSY_LINES if info else camscreen.NO_CAMERA_LINES
    return f"{what}. " + " ".join(tips)


def start_camera(cfg: Config, display, reason: str | None = None):
    """Open the camera to use: (Camera, CameraInfo), or None when there is none and
    the user quit (or there is no window to ask in).

    --camera N, and every system but Windows (the Raspberry Pi), open that index (0 by
    default) as before. On Windows without --camera: the camera saved in settings.json
    if it is plugged in, else the picker when there is a choice to make. On Windows a
    missing or busy camera shows a message that keeps retrying instead of ending the app.
    """
    if cfg.camera is not None or not WINDOWS:
        index = cfg.camera if cfg.camera is not None else 0
        names = cameras.dshow_names() if WINDOWS else None
        info = CameraInfo(index, names[index] if names and index < len(names)
                          else f"Camera {index}")
        cam = try_open(info, cfg)
        if cam is not None:
            return cam, info
        crashlog.write(_camera_problem(info))
        if display is None or not WINDOWS:
            print(_camera_problem(info), file=sys.stderr)
            return None
        kind, value = camscreen.message_screen(
            display, f"{info.name} is busy or gives no picture",
            camscreen.BUSY_LINES, "Trying again every few seconds.   Q quits",
            poll=lambda: try_open(info, cfg))
        return (value, info) if kind == "poll" else None

    force_pick = False
    while True:
        cams = cameras.list_cameras()
        if not cams:
            crashlog.write("No camera found")
            if display is None:
                print(_camera_problem(None), file=sys.stderr)
                return None
            kind, _ = camscreen.message_screen(
                display, "No camera found", camscreen.NO_CAMERA_LINES,
                "Looking for a camera every few seconds.   Q quits",
                poll=lambda: cameras.list_cameras() or None)
            if kind != "poll":
                return None
            continue
        saved = settings.load().get("camera")
        info = None if force_pick else cameras.choose_saved(cams, saved)
        if info is None:
            if display is not None and (len(cams) > 1 or saved or force_pick):
                why = reason
                if why is None and saved and not force_pick:
                    why = f"The saved camera ({saved.get('name', 'unknown')}) is not plugged in."
                info = camscreen.pick_camera(display, cams, reason=why)
                if info is None:
                    return None
            else:
                info = cams[0]
        force_pick = False
        cam = try_open(info, cfg)
        if cam is None:
            crashlog.write(_camera_problem(info))
            if display is None:
                print(_camera_problem(info), file=sys.stderr)
                return None
            kind, cam = camscreen.message_screen(
                display, f"{info.name} is busy or gives no picture", camscreen.BUSY_LINES,
                "Trying again every few seconds.   C: choose another camera   Q: quit",
                poll=lambda: try_open(info, cfg), keys="c")
            if kind == "key":
                force_pick = True
                continue
            if kind != "poll":
                return None
        settings.update(camera=info.as_setting())
        crashlog.write(f"Camera {info.index}: {info.name}")
        return cam, info


def switch_camera(cfg: Config, cam: Camera, info: CameraInfo, now: float, show_toast):
    """The c key: move to the next camera; keep the current one if the next fails.
    Returns (camera or None when neither opens, its info)."""
    cams = cameras.list_cameras()
    nxt = cameras.next_camera(cams, info.index)
    if nxt is None or nxt.index == info.index:
        show_toast("Only one camera found", now)
        return cam, info
    cam.release()
    new = try_open(nxt, cfg)
    if new is not None:
        if WINDOWS:
            settings.update(camera=nxt.as_setting())
        crashlog.write(f"Camera switched to {nxt.index}: {nxt.name}")
        show_toast(nxt.name, now)
        return new, nxt
    crashlog.write(f"Camera {nxt.index} ({nxt.name}) gives no picture, staying on {info.name}")
    show_toast(f"{nxt.name} is busy", now)
    return try_open(info, cfg), info


def self_test(cfg: Config) -> int:
    """--self-test: load the model, track one blank frame, check the fonts, list the
    cameras, write the result to crash.log and exit. Checks a build without a camera."""
    from . import ui
    try:
        tracker = HandTracker(cfg.model_path, cfg.max_hands, cfg.square_pad)
        hands = tracker.detect(np.zeros((480, 640, 3), np.uint8), 0)
        tracker.close()
        fonts = [p for p in (ui.FONT_DIR / "Inter-Bold.ttf", ui.FONT_DIR / "Inter-Regular.ttf")
                 if p.exists()]
        cams = cameras.list_cameras()
    except Exception:
        crashlog.write("Self-test FAILED", sys.exc_info())
        traceback.print_exc()
        return 1
    ok = len(fonts) == 2
    msg = (f"Self-test {'OK' if ok else 'FAILED'}: model loaded, {len(hands)} hands on a "
           f"blank frame, {len(fonts)} of 2 Inter fonts found, cameras: "
           f"{', '.join(f'{c.index}={c.name}' for c in cams) or 'none'}, "
           f"data folder {paths.DATA_DIR}, portable build: {paths.FROZEN}")
    print(msg)
    crashlog.write(msg)
    return 0 if ok else 1


def run(cfg: Config) -> int:
    # The model loads first, then the window opens, then the camera: the picker and any
    # camera message need the window.
    tracker = TrackerThread(HandTracker(cfg.model_path, cfg.max_hands, cfg.square_pad))
    display = None if cfg.no_display else Display(WINDOW, (cfg.width, cfg.height), cfg.kiosk)
    note = paths.fallback_message()
    if note:
        print(note, file=sys.stderr)
        crashlog.write(note)
        if display:
            camscreen.message_screen(
                display, "Saving to a temporary folder",
                [note, "Photos saved there are not on the USB stick. Copy the GestureFX "
                       "folder to a drive you can write to, or unlock the stick."],
                "Any key continues", duration=8.0)
    opened = start_camera(cfg, display)
    if opened is None:
        tracker.close()
        cv2.destroyAllWindows()
        crashlog.write("Stopped: no camera")
        return 1
    cam, cam_info = opened
    predictor = HandPredictor()
    used_seq = -1          # frame number of the last tracker result taken in
    track_ms = 0.0
    if cfg.pi:
        print(f"Raspberry Pi profile: up to {cfg.max_hands} hands, lighter effects, "
              f"tracking wait {cfg.track_wait_ms:g} ms (--no-pi turns it off)")
    players = Players(cfg)
    btn_size = cam.size
    buttons = nav.layout(*btn_size, cfg.switch_hold_s)
    portal_pinch, portal_pair = gestures.BothPinch(), None
    cues = []  # (start time, points) of recent portal style changes, for the flash rings
    ui, ui_gif = UI(), UI()
    names = tuple(n[:1].upper() + n[1:] for n in modes.NAMES)
    clear_hold = gestures.HoldDetector(cfg.clear_hold_s, 1.5)
    peace_hold = gestures.HoldDetector(cfg.peace_hold_s, 5.0)
    booth = PhotoBooth(cfg.photo_dir)
    recorder = GifRecorder(cfg.record_gif, cfg.gif_seconds) if cfg.record_gif else None
    renderer = modes.ModeRenderer(cfg.lite)

    mode, style_idx = (cfg.start_mode - 1) % len(modes.NAMES), 0
    show_skeleton, show_stats = True, False
    info_until = 0.0
    toast, toast_t0 = None, -99.0
    stats_text, stats_t, stats_runs = None, -1.0, 0
    fps, last_t = 0.0, time.perf_counter()
    t0 = last_hand_t = last_t
    bench = defaultdict(lambda: [0, 0.0, 0, 0.0])  # hands -> [frames, s, results, tracking ms]
    exit_code, reason = 0, "unknown"
    frame_no = 0

    def show_toast(msg, now):
        nonlocal toast, toast_t0
        toast, toast_t0 = msg[:1].upper() + msg[1:], now - t0

    try:
        while True:
            frame = cam.read() if cam is not None else None
            if frame is None and display is not None and WINDOWS:
                # Unplugged or taken by another app: the messages and picker take over
                crashlog.write(f"Camera {cam_info.index} ({cam_info.name}) stopped "
                               "delivering frames", context=dict(CONTEXT))
                if cam is not None:
                    cam.release()
                cam = None
                opened = start_camera(cfg, display, reason="The camera stopped sending pictures.")
                if opened is None:
                    exit_code, reason = 1, "camera lost and the user quit"
                    break
                cam, cam_info = opened
                continue
            if frame is None:
                print("Camera stopped delivering frames for 5 s", file=sys.stderr)
                exit_code, reason = 1, "camera delivered no frames for 5 s"
                break
            if cfg.mirror:
                frame = cv2.flip(frame, 1)
            frame_no += 1

            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            t = now - t0
            if dt > 0:
                fps = 1.0 / dt if fps == 0 else 0.9 * fps + 0.1 / dt
            k = frame.shape[0] / 480.0

            # Hand the frame to the tracker thread, then take its newest result. Gestures
            # (pinch clicks, fists, ...) are worked out once per result; between results
            # the predictor only moves the hands.
            tracker.submit(frame.copy(), int(t * 1000), frame_no)
            res = tracker.wait(frame_no, cfg.track_wait_ms / 1000)
            fresh = res is not None and res.seq != used_seq
            if fresh:
                used_seq, track_ms = res.seq, res.ms
                predictor.observe(players.update(res.hands[:cfg.max_hands], frame.shape), res.seq)
            new_clicks = players.clicks if fresh else []
            infos = predictor.predict(frame_no)
            if infos:
                last_hand_t = now
            CONTEXT.update(frame=frame_no, t=f"{t:.1f}s", mode=modes.NAMES[mode],
                           hands=len(infos), fps=f"{fps:.1f}")

            # NEXT / PREVIOUS buttons: an index fingertip held inside one for switch_hold_s
            # changes the mode once; it re-arms when the fingertip leaves. Mouse clicks too.
            h_f, w_f = frame.shape[:2]
            if (w_f, h_f) != btn_size:  # the camera delivers another size than requested
                buttons, btn_size = nav.layout(w_f, h_f, cfg.switch_hold_s), (w_f, h_f)
            prev_mode = mode
            mouse = display.pop_clicks() if display else []
            for b in buttons:
                pressed = b.dwell.update(bool(nav.tips_inside(b, infos)) and not booth.busy, now)
                pressed |= any(b.contains(p) for p in mouse)
                if pressed and b.step == 0:          # info button: open or close the card
                    info_until = 0.0 if now < info_until else now + INFO_OPEN_S
                elif pressed:
                    mode = (mode + b.step) % len(modes.NAMES)
            # Hands pointing at a button are kept away from the effects
            busy_ids = nav.blocked(infos, buttons)
            fx_infos = [h for h in infos if h.id not in busy_ids]

            byid = {h.id: h for h in infos}
            fists = [h for h in fx_infos if h.fist]
            clicks = [c for c in new_clicks if c not in busy_ids]
            clear = clear_hold.update(bool(fists) and mode == modes.NEON, now)
            if mode == modes.STYLES and clicks:
                step = sum(-1 if byid[c].label == "Left" else 1 for c in clicks)
                style_idx = (style_idx + (1 if step >= 0 else -1)) % len(STYLES)
                show_toast(STYLES[style_idx], now)

            # Portal: the two frame hands pinch together and let go = next style inside
            fq = modes.frame_quad(fx_infos) if mode == modes.PORTAL else None
            pair = (fq[1].id, fq[2].id) if fq else None
            if pair != portal_pair:
                portal_pinch.reset()
                portal_pair = pair
            if pair and portal_pinch.update(pair[0] in players.pinched, pair[1] in players.pinched):
                style_idx = (style_idx + 1) % len(STYLES)
                show_toast(STYLES[style_idx], now)
                cues.append((now, [fq[1].mid, fq[2].mid]))
            cues = [c for c in cues if now - c[0] < CUE_S]

            peace_hands = [h for h in infos if gestures.is_peace(h.lm)]
            if peace_hold.update(len(peace_hands) >= 2 and not booth.busy, now):
                booth.start(now)

            # One mode's bug must not end the show: log it, reset that mode, carry on.
            try:
                out = renderer.render(mode, frame, fx_infos, clicks, clear,
                                      STYLES[style_idx], t)
            except Exception:
                key = (mode,) + tuple(traceback.format_exception_only(*sys.exc_info()[:2]))
                if crashlog.write_once(key, f"Error in mode '{modes.NAMES[mode]}' (app kept running)",
                                       dict(CONTEXT)):
                    traceback.print_exc()
                renderer.reset_mode(mode)
                show_toast("Effect error, see crash.log", now)
                out = frame
            if out is frame:
                out = frame.copy()

            shot = booth.update(now)
            if shot.capture:  # save the picture before any hand drawing or text goes on it
                try:
                    print(f"Photo saved: {booth.save(out).name}")
                except OSError as e:
                    print(f"Could not save photo: {e}", file=sys.stderr)

            for h in infos:
                overlay.draw_pinch(out, h)
            overlay.draw_hands(out, infos, show_skeleton)

            if mode != prev_mode:
                show_toast(modes.NAMES[mode], now)
            if show_stats and (now - stats_t > 0.5 or stats_text is None):
                rate = (tracker.runs - stats_runs) / max(now - stats_t, 1e-3) if stats_t > 0 else 0
                stats_runs = tracker.runs
                stats_text, stats_t = (f"{fps:.0f} fps   {rate:.0f} tracks/s   {track_ms:.0f} ms   "
                                       f"{len(infos)} hands"), now
            peace_p = peace_hold.progress(now)
            hud = UIState(
                mode=mode, names=names, t=t, kiosk=cfg.kiosk, toast=toast, toast_t0=toast_t0,
                stats=stats_text if show_stats else None, info_open=now < info_until,
                buttons=[NavView(tuple(b.centre), b.radius, b.step,
                                 b.dwell.progress(now) if nav.tips_inside(b, infos) else 0.0,
                                 bool(nav.tips_inside(b, infos))) for b in buttons],
                countdown=shot.countdown, countdown_frac=shot.countdown_frac, flash=shot.flash,
                saved=shot.message is not None,
                attract=(now - last_hand_t > cfg.attract_after) and not booth.busy,
                cues=[(pts, (now - tc) / CUE_S) for tc, pts in cues],
                photo_hold=(tuple((peace_hands[0].palm + peace_hands[1].palm) / 2), peace_p)
                if peace_p > 0 and len(peace_hands) >= 2 else None,
                brand=cfg.brand, logo=str(cfg.logo) if cfg.logo else None)
            if mode == modes.BUBBLES:   # each player's score above their hand
                hud.counters = [((h.palm[0], h.palm[1] - h.palm_r * 2.2), h.color,
                                 renderer.bubbles.score.get(h.id, 0)) for h in fx_infos]
            if mode == modes.ENERGY and renderer.energy.prompt is not None:
                hud.prompt = (renderer.energy.prompt, "Thrust!")

            if recorder:
                # The GIF gets its own copy with the UI drawn at capture size (without the
                # recording dot and countdown, which are set only for the window below)
                gif_frame = out.copy()
                ui_gif.draw(gif_frame, hud)
                state = recorder.update(gif_frame, now)
                if state == "done":
                    reason = "GIF recording finished"
                    break
                if state.startswith("countdown"):
                    hud.countdown, hud.countdown_frac = int(state.split(":")[1]), 1.0
                hud.recording = state == "recording"

            bench_n = bench[len(infos)]
            if cfg.benchmark is not None and now - t0 > BENCH_WARMUP:
                bench_n[0] += 1
                bench_n[1] += dt
                if fresh:
                    bench_n[2] += 1
                    bench_n[3] += res.ms
                if now - t0 > BENCH_WARMUP + cfg.benchmark:
                    reason = "benchmark finished"
                    break

            if display:
                canvas = display.canvas_for(out)
                ui.draw(canvas, hud, display.frame_map)
                display.present(canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    reason = "q or Esc pressed"
                    break
                idx = modes.KEYS.find(chr(key)) if 32 <= key < 127 else -1
                if idx >= 0:
                    mode = idx
                elif key == ord("p"):
                    booth.start(now)
                elif key == ord("x"):
                    renderer.neon.clear()
                elif key == ord("c"):   # next camera
                    cam, cam_info = switch_camera(cfg, cam, cam_info, now, show_toast)
                elif key == ord("i"):
                    info_until = 0.0 if now < info_until else now + INFO_OPEN_S
                elif key == ord("v"):
                    show_stats = not show_stats
                elif key == ord("f"):
                    display.toggle_fullscreen()
                elif key == ord("d"):  # landmark check: raw frame + landmarks, and the window
                    save_debug(frame, infos, canvas)
                elif key == ord("l"):
                    show_skeleton = not show_skeleton
                elif key == ord("s"):  # screenshot with the HUD; may show faces, so git-ignored
                    path = ROOT / "debug" / f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
                    path.parent.mkdir(exist_ok=True)
                    shot_img = out.copy()
                    hud.flash, hud.saved, hud.countdown = 0.0, False, None
                    UI().draw(shot_img, hud)
                    cv2.imwrite(str(path), shot_img)
                    print(f"Saved {path}")
                if display.closed():
                    reason = "window closed"
                    break
    except BaseException:
        reason = "crashed, details follow"
        raise
    finally:
        crashlog.write(f"Stopped: {reason}", context=dict(CONTEXT))
        if cam is not None:
            cam.release()
        tracker.close()
        cv2.destroyAllWindows()

    if recorder:
        try:
            print(f"Saved {recorder.save()}")
        except RuntimeError as e:
            print(e, file=sys.stderr)
            exit_code = 1
    frames = sum(v[0] for v in bench.values())
    if cfg.benchmark is not None and frames:
        secs = sum(v[1] for v in bench.values())
        w, h = cam.size if cam is not None else (cfg.width, cfg.height)
        runs = sum(v[2] for v in bench.values())
        print(f"Benchmark: {w}x{h}, {cfg.benchmark:.0f} s, up to {cfg.max_hands} hands"
              f"{', Pi profile' if cfg.pi else ''}, tracking wait {cfg.track_wait_ms:g} ms")
        if cam is not None and cam.fps is not None:
            print(f"  camera:          {cam.fps:5.1f} FPS delivered (the display cannot beat this)")
        print(f"  display/effects: {frames / secs:5.1f} FPS")
        print(f"  hand tracking:   {runs / secs:5.1f} results/s, "
              f"{sum(v[3] for v in bench.values()) / max(runs, 1):5.1f} ms per result")
        for n in sorted(bench):
            f, s, r, ms = bench[n]
            if f:
                print(f"  {n} hands: {f / s:5.1f} FPS display, {r / s:5.1f} results/s, "
                      f"{ms / max(r, 1):5.1f} ms tracking ({f} frames)")
    return exit_code


def main(argv=None) -> int:
    if sys.stdout is None:   # the windowed portable build has no console
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")
    crashlog.install()
    cfg = parse_args(argv)
    args = sys.argv[1:] if argv is None else argv
    crashlog.write("Session start", context={"args": " ".join(args) or "(none)",
                                             "versions": crashlog.versions(),
                                             "app folder": paths.APP_DIR,
                                             "data folder": paths.DATA_DIR})
    try:
        if cfg.self_test:
            return self_test(cfg)
        return run(cfg)
    except KeyboardInterrupt:
        crashlog.write("Stopped: Ctrl+C", context=dict(CONTEXT))
        return 130
    except Exception:
        crashlog.write("CRASH", sys.exc_info(), dict(CONTEXT))
        traceback.print_exc()
        print(f"\nThe app crashed. The details are in {crashlog.LOG_PATH}", file=sys.stderr)
        return 1
