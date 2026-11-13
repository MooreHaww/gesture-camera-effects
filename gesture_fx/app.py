"""Main loop: camera -> tracking -> players -> mode -> overlay -> window."""
import sys
import time
import traceback
from collections import defaultdict

import cv2
import numpy as np

from . import buttons as nav
from . import crashlog, gestures, modes, overlay
from .camera import Camera
from .config import ROOT, Config, parse_args
from .display import Display
from .effects import STYLES
from .photobooth import PhotoBooth
from .players import Players
from .recorder import GifRecorder
from .tracking import HandTracker
from .ui import UI, NavView, UIState

WINDOW = "Gesture Camera Effects"
BENCH_WARMUP = 1.0
INFO_OPEN_S = 8.0   # the info card closes itself after this long
CUE_S = 0.6

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


def run(cfg: Config) -> int:
    cam = Camera(cfg.camera, cfg.width, cfg.height)
    tracker = HandTracker(cfg.model_path, cfg.max_hands, cfg.square_pad)
    display = None if cfg.no_display else Display(WINDOW, cam.size, cfg.kiosk)
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
    renderer = modes.ModeRenderer()

    mode, style_idx = (cfg.start_mode - 1) % len(modes.NAMES), 0
    show_skeleton, show_stats = True, False
    info_until = 0.0
    toast, toast_t0 = None, -99.0
    stats_text, stats_t = None, -1.0
    fps, last_t = 0.0, time.perf_counter()
    t0 = last_hand_t = last_t
    bench = defaultdict(lambda: [0, 0.0, 0.0])  # hands -> [frames, seconds, tracking ms]
    exit_code, reason = 0, "unknown"
    frame_no = 0

    def show_toast(msg, now):
        nonlocal toast, toast_t0
        toast, toast_t0 = msg[:1].upper() + msg[1:], now - t0

    try:
        while True:
            frame = cam.read()
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

            t_a = time.perf_counter()
            hands = tracker.detect(frame, int(t * 1000))[:cfg.max_hands]
            track_ms = (time.perf_counter() - t_a) * 1000
            infos = players.update(hands, frame.shape)
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
            clicks = [c for c in players.clicks if c not in busy_ids]
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
                stats_text, stats_t = f"{fps:.0f} fps   {track_ms:.0f} ms   {len(infos)} hands", now
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
                bench_n[2] += track_ms
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
                elif key == ord("c"):
                    renderer.neon.clear()
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
        w, h = cam.size
        print(f"Benchmark: {frames / secs:.1f} FPS average, "
              f"{sum(v[2] for v in bench.values()) / frames:.1f} ms tracking average, "
              f"{w}x{h}, {cfg.benchmark:.0f} s, up to {cfg.max_hands} hands")
        for n in sorted(bench):
            f, s, ms = bench[n]
            if f:
                print(f"  {n} hands: {f / s:5.1f} FPS, {ms / f:5.1f} ms tracking ({f} frames)")
    return exit_code


def main(argv=None) -> int:
    crashlog.install()
    cfg = parse_args(argv)
    args = sys.argv[1:] if argv is None else argv
    crashlog.write("Session start", context={"args": " ".join(args) or "(none)",
                                             "versions": crashlog.versions()})
    try:
        return run(cfg)
    except KeyboardInterrupt:
        crashlog.write("Stopped: Ctrl+C", context=dict(CONTEXT))
        return 130
    except Exception:
        crashlog.write("CRASH", sys.exc_info(), dict(CONTEXT))
        traceback.print_exc()
        print(f"\nThe app crashed. The details are in {crashlog.LOG_PATH}", file=sys.stderr)
        return 1
