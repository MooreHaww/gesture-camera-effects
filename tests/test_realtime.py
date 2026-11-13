"""Tracking thread, hand prediction between results, and the Raspberry Pi profile."""
import time

import numpy as np
import pytest

from gesture_fx.config import PC_TRACK_WAIT_MS, parse_args
from gesture_fx.players import HandInfo, HandPredictor
from gesture_fx.tracking import TrackerThread


def info(x, hid=1, vel=(0.0, 0.0)):
    lm = np.zeros((21, 2), np.float32)
    lm[:, 0] = x
    lm[9] = (x, -100)      # palm size 100 px
    return HandInfo(id=hid, lm=lm, mid=lm[4].copy(), palm=(lm[0] + lm[9]) / 2, palm_r=100.0,
                    open=1.0, ratio=1.0, fist=False, color=(255, 255, 255),
                    vel=np.array(vel, float), tips=lm[[4, 8, 12, 16, 20]])


def test_predictor_passes_hands_through_when_every_frame_is_tracked():
    p = HandPredictor()
    for seq in range(5):
        hands = [info(10.0 * seq)]
        p.observe(hands, seq)
        assert p.predict(seq)[0] is hands[0]   # the PC case: nothing changes


def test_predictor_carries_a_moving_hand_forward_between_results():
    p = HandPredictor()
    p.observe([info(0.0)], 0)
    p.observe([info(20.0)], 2)                 # 10 px per frame
    assert p.predict(3)[0].lm[0, 0] == pytest.approx(30.0)
    assert p.predict(4)[0].palm[0] == pytest.approx(40.0)


def test_predictor_stops_after_max_ahead_frames():
    p = HandPredictor()
    p.observe([info(0.0)], 0)
    p.observe([info(20.0)], 2)
    far = p.predict(2 + 50)[0].lm[0, 0]
    assert far == pytest.approx(20.0 + 10.0 * HandPredictor.MAX_AHEAD)


def test_predictor_ignores_a_tracking_jump_and_jitter():
    p = HandPredictor()
    p.observe([info(0.0)], 0)
    p.observe([info(400.0)], 1)                # 4 palm sizes in one frame: a jump
    assert p.predict(3)[0].lm[0, 0] == pytest.approx(400.0)
    p.observe([info(400.5)], 2)                # half a pixel: jitter, keep still
    assert p.predict(5)[0].lm[0, 0] == pytest.approx(400.5)


def test_predictor_eases_a_correction_instead_of_snapping():
    p = HandPredictor()
    p.observe([info(0.0)], 0)
    p.observe([info(20.0)], 2)                 # 10 px per frame
    assert p.predict(3)[0].lm[0, 0] == pytest.approx(30.0)
    p.observe([info(30.0)], 4)                 # it slowed down: 5 px per frame
    # the guess for frame 5 is 35; coming from 30 + 5, it is reached without a jump
    assert p.predict(5)[0].lm[0, 0] == pytest.approx(35.0)
    p.observe([info(30.0)], 6)                 # stopped, but 35 was drawn last frame
    a, b = p.predict(6)[0].lm[0, 0], p.predict(7)[0].lm[0, 0]
    assert 35.0 > a > b > 30.0                 # glides back to 30 instead of jumping


def test_predictor_gives_velocity_per_camera_frame():
    p = HandPredictor()
    p.observe([info(0.0, vel=(4.0, 0.0))], 0)
    p.observe([info(8.0, vel=(8.0, 0.0))], 2)  # Players measured 8 px per result
    assert p.predict(2)[0].vel[0] == pytest.approx(4.0)


class SlowTracker:
    def __init__(self, ms):
        self.ms, self.seen, self.closed = ms, [], False

    def detect(self, frame, ts):
        time.sleep(self.ms / 1000)
        self.seen.append(frame)
        return [frame]

    def close(self):
        self.closed = True


def test_tracker_thread_fast_tracker_keeps_every_frame_in_step():
    th = TrackerThread(SlowTracker(2))
    try:
        for seq in range(5):
            th.submit(seq, seq * 33, seq)
            assert th.wait(seq, 1.0).seq == seq
    finally:
        th.close()


def test_tracker_thread_slow_tracker_never_blocks_and_tracks_newest_frame():
    tr = SlowTracker(60)
    th = TrackerThread(tr)
    try:
        t0 = time.perf_counter()
        for seq in range(10):
            th.submit(seq, seq * 33, seq)
            th.wait(seq, 0.0)
            time.sleep(0.01)
        assert time.perf_counter() - t0 < 0.3          # the loop did not wait
        th.wait(9, 1.0)
        assert tr.seen[-1] == 9                       # newest frame was tracked
        assert len(tr.seen) < 10                      # older frames were skipped
    finally:
        th.close()
    assert tr.closed


def test_tracker_thread_reports_a_tracker_error():
    class Broken(SlowTracker):
        def detect(self, frame, ts):
            raise ValueError("boom")
    th = TrackerThread(Broken(0))
    th.submit(0, 0, 0)
    with pytest.raises(RuntimeError):
        th.wait(0, 1.0)
    th.close()


def test_pi_profile_defaults_and_overrides():
    pc = parse_args(["--no-pi"])
    assert (pc.pi, pc.lite, pc.max_hands, pc.track_wait_ms) == (False, False, 4, PC_TRACK_WAIT_MS)
    pi = parse_args(["--pi"])
    assert (pi.pi, pi.lite, pi.max_hands, pi.track_wait_ms) == (True, True, 2, 0.0)
    own = parse_args(["--pi", "--max-hands", "3", "--track-wait", "5"])
    assert (own.max_hands, own.track_wait_ms) == (3, 5.0)
