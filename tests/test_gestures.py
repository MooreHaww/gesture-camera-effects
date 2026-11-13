import numpy as np
import pytest

from gesture_fx import gestures as g


def hand(thumb=(0, 0), index=(0, 0), palm=100.0):
    """Hand with wrist at origin, middle knuckle `palm` away, given tip positions."""
    lm = np.zeros((21, 2), dtype=float)
    lm[g.MIDDLE_MCP] = (0, -palm)
    lm[g.THUMB_TIP] = thumb
    lm[g.INDEX_TIP] = index
    return lm


def test_pinch_ratio_is_distance_over_palm():
    assert g.pinch_ratio(hand((0, 0), (50, 0), palm=100)) == pytest.approx(0.5)


def test_pinch_ratio_ignores_scale():
    near = g.pinch_ratio(hand((0, 0), (100, 0), palm=200))
    far = g.pinch_ratio(hand((0, 0), (50, 0), palm=100))
    assert near == pytest.approx(far)


def test_pinch_ratio_degenerate_palm_is_zero():
    assert g.pinch_ratio(np.zeros((21, 2))) == 0.0


def test_pinch_midpoint():
    mid = g.pinch_midpoint(hand((10, 20), (30, 60)))
    assert mid.tolist() == [20, 40]


def test_openness_maps_and_clamps():
    assert g.openness(0.0, 0.2, 1.1) == 0.0
    assert g.openness(1.5, 0.2, 1.1) == 1.0
    assert g.openness(0.65, 0.2, 1.1) == pytest.approx(0.5)


def test_polygon_area():
    assert g.polygon_area([(0, 0), (10, 0), (10, 10), (0, 10)]) == pytest.approx(100)
    assert g.polygon_area([(0, 0), (10, 10), (10, 0), (0, 10)]) == pytest.approx(0)  # bow-tie


def test_portal_quad_uses_exact_fingertips():
    quad = g.portal_quad((0, 100), (0, 0), (100, 100), (100, 0))
    assert {tuple(p) for p in quad.tolist()} == {(0, 100), (0, 0), (100, 100), (100, 0)}
    assert g.polygon_area(quad) == pytest.approx(10000)


def test_portal_quad_never_crosses_when_one_hand_is_flipped():
    normal = g.portal_quad((0, 100), (0, 0), (100, 100), (100, 0))
    flipped = g.portal_quad((0, 100), (0, 0), (100, 0), (100, 100))  # B thumb/index swapped
    assert g.polygon_area(flipped) == pytest.approx(g.polygon_area(normal))


def test_portal_quad_tilted_hands_give_a_simple_quad():
    a_t, a_i = (10, 120), (30, 10)
    b_t, b_i = (150, 100), (140, 0)
    quad = g.portal_quad(a_t, a_i, b_t, b_i)
    other = g.portal_quad(a_t, a_i, b_i, b_t)
    assert g.polygon_area(quad) > 5000
    assert g.polygon_area(quad) == pytest.approx(g.polygon_area(other))


def test_hand_identity_keeps_ids_when_order_changes():
    ident = g.HandIdentity(max_dist=50)
    assert ident.update([(100, 100), (300, 100)]) == [0, 1]
    assert ident.update([(305, 102), (103, 98)]) == [1, 0]   # list order swapped


def test_hand_identity_new_hand_takes_lowest_free_id():
    ident = g.HandIdentity(max_dist=50, grace=0)
    ident.update([(100, 100), (300, 100)])
    assert ident.update([(300, 100)]) == [1]                  # hand 0 left
    assert ident.update([(300, 100), (500, 300)]) == [1, 0]   # newcomer reuses id 0


def test_hand_identity_returning_hand_keeps_id_within_grace():
    ident = g.HandIdentity(max_dist=50, grace=3)
    ident.update([(100, 100), (300, 100)])
    for _ in range(3):
        ident.update([(300, 100)])
    assert ident.update([(300, 100), (104, 100)]) == [1, 0]


def test_hand_identity_too_far_is_a_new_hand():
    ident = g.HandIdentity(max_dist=50)
    ident.update([(100, 100)])
    assert ident.update([(400, 400)]) == [1]


def test_hand_identity_label_breaks_ties_between_close_hands():
    ident = g.HandIdentity(max_dist=100)
    assert ident.update([(100, 100), (140, 100)], ["Left", "Right"]) == [0, 1]
    # the hands swap list order and cross a little: labels keep each colour on its hand
    assert ident.update([(128, 100), (112, 100)], ["Right", "Left"]) == [1, 0]


def test_hand_identity_far_return_keeps_colour_by_label():
    ident = g.HandIdentity(max_dist=50, grace=90)
    ident.update([(100, 100), (500, 100)], ["Left", "Right"])
    for _ in range(10):
        ident.update([(100, 100)], ["Left"])                  # right hand lost for a while
    assert ident.update([(100, 100), (250, 300)], ["Left", "Right"]) == [0, 1]


def test_hand_identity_predicts_fast_motion():
    ident = g.HandIdentity(max_dist=60)
    x = 100.0
    for _ in range(5):
        x += 55                                               # 55 px per frame
        assert ident.update([(x, 100), (500, 400)], ["Right", "Left"]) == [0, 1]
    assert ident.update([(x + 110, 100), (500, 400)], ["Right", "Left"]) == [0, 1]  # 2 frames' jump


def hand_with_fingers(index, middle, ring, pinky, apart=True):
    """Wrist at origin; each finger extended (True) or curled (False)."""
    lm = np.zeros((21, 2), dtype=float)
    lm[g.MIDDLE_MCP] = (0, -100)
    spec = {(8, 6): (index, -30 if apart else 0), (12, 10): (middle, 30 if apart else 0),
            (16, 14): (ring, 60), (20, 18): (pinky, 90)}
    for (tip, pip), (ext, x) in spec.items():
        lm[pip] = (x, -120)
        lm[tip] = (x, -200 if ext else -60)
    return lm


def test_is_peace():
    assert g.is_peace(hand_with_fingers(True, True, False, False))
    assert not g.is_peace(hand_with_fingers(True, True, True, True))        # open hand
    assert not g.is_peace(hand_with_fingers(False, False, False, False))    # fist
    assert not g.is_peace(hand_with_fingers(True, False, False, False))     # pointing
    assert not g.is_peace(hand_with_fingers(True, True, False, False, apart=False))  # fingers together


def test_finger_spread():
    lm = np.zeros((21, 2), dtype=float)
    lm[g.MIDDLE_MCP] = (0, -100)
    lm[g.THUMB_TIP], lm[20] = (-30, -100), (30, -100)       # span 0.6 palms: together
    assert g.finger_spread(lm) == 0.0
    lm[g.THUMB_TIP], lm[20] = (-100, -100), (100, -100)     # span 2 palms: wide
    assert g.finger_spread(lm) == 1.0


def test_both_pinch_fires_once_after_both_let_go():
    bp = g.BothPinch()
    seq = [(True, False), (True, True), (False, True), (False, False), (False, False)]
    assert [bp.update(a, b) for a, b in seq] == [False, False, False, True, False]


def test_both_pinch_needs_both_closed_at_once():
    bp = g.BothPinch()
    seq = [(True, False), (False, False), (False, True), (False, False)]   # one at a time
    assert not any(bp.update(a, b) for a, b in seq)


def test_dwell_fires_once_then_rearms_after_leaving():
    d = g.Dwell(hold_s=1.5, grace_s=0.15)
    assert not d.update(True, 0.0)
    assert d.progress(0.75) == pytest.approx(0.5)
    assert d.update(True, 1.5)
    assert not d.update(True, 4.0)           # fingertip still inside: no second press
    d.update(False, 4.5)
    d.update(False, 4.7)                     # fingertip left: re-armed
    assert not d.update(True, 5.0)
    assert d.update(True, 6.5)


def test_dwell_cancels_when_the_fingertip_leaves():
    d = g.Dwell(hold_s=1.5, grace_s=0.15)
    d.update(True, 0.0)
    d.update(False, 1.0)
    d.update(False, 1.2)                     # gone long enough: cancelled
    assert not d.update(True, 1.6)           # timer restarted at 1.6
    assert d.progress(1.6) == 0.0


def test_dwell_ignores_one_frame_flicker():
    d = g.Dwell(hold_s=1.5, grace_s=0.15)
    d.update(True, 0.0)
    d.update(False, 0.7)                     # tracker missed the hand for one frame
    assert d.update(True, 1.5)


def test_smoother():
    s = g.Smoother(0.5)
    assert s.update(10) == 10
    assert s.update(20) == 15
    s.reset()
    assert s.update(4) == 4


def test_pinch_click_fires_once_on_release_with_hysteresis():
    c = g.PinchClick(close=0.3, release=0.6)
    results = [c.update(r) for r in (1.0, 0.2, 0.4, 0.5, 0.7, 0.2, 0.8)]
    # closed at 0.2; 0.4 and 0.5 are inside the hysteresis band; release at 0.7
    assert results == [False, False, False, False, True, False, True]


def test_pinch_click_needs_close_below_release():
    with pytest.raises(AssertionError):
        g.PinchClick(close=0.6, release=0.3)


def fist_hand(curled: bool):
    lm = np.zeros((21, 2), dtype=float)
    for tip, pip in zip(g.FINGER_TIPS, g.FINGER_PIPS):
        lm[pip] = (0, -50)
        lm[tip] = (0, -20) if curled else (0, -100)
    lm[g.MIDDLE_MCP] = (0, -40)
    return lm


def test_is_fist():
    assert g.is_fist(fist_hand(True))
    assert not g.is_fist(fist_hand(False))


def test_hold_detector_fires_after_hold_then_cooldown():
    h = g.HoldDetector(hold_s=1.0, cooldown_s=2.0)
    assert not h.update(True, 0.0)
    assert h.progress(0.5) == pytest.approx(0.5)
    assert not h.update(True, 0.9)
    assert h.update(True, 1.0)
    assert not h.update(True, 1.5)   # cooldown
    assert not h.update(True, 2.9)   # still cooling down; hold restarts after
    assert not h.update(True, 3.0)
    assert h.update(True, 4.0)


def test_hold_detector_resets_when_released():
    h = g.HoldDetector(hold_s=1.0, cooldown_s=0.0)
    h.update(True, 0.0)
    h.update(False, 0.5)
    assert not h.update(True, 1.2)   # timer restarted at 1.2
    assert h.update(True, 2.2)
