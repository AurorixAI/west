"""Each rule fires on its event and stays silent on the look-alike that is not one."""
from __future__ import annotations

import numpy as np
import pytest

from src import rules
from tests.synth import BICYCLE, CAR, DOG, MOTORCYCLE, PERSON, RED, Scene, times


def labels(segs, name):
    return [(round(s, 1), round(e, 1)) for s, e, lbl, _ in segs if lbl == name]


@pytest.fixture
def scene():
    s = Scene(120.0)
    s.background()
    return s


def test_background_traffic_is_silent(scene):
    """Normal traffic in every lane, all signals green: no event of any class."""
    assert rules.detect(scene.context()) == []


# -- red light / stop line ---------------------------------------------------
@pytest.fixture
def quiet_scene():
    """Dense traffic until t=40 s teaches the lanes; after that only the actors of the test."""
    s = Scene(120.0)
    s.background(0, 40, every=2.0)
    return s


@pytest.fixture
def red_scene(quiet_scene):
    """Signal red over [50, 80] s, and nobody else on the road runs it."""
    quiet_scene.set_signal((50, 80, RED))
    return quiet_scene


def test_red_light_run(red_scene):
    scene = red_scene
    v = scene.path("vehicle", CAR, 60.0, [(900, 600), (900, 2100)], 400)
    ev = labels(rules.red_light(scene.context()), "red_light")
    i = int(np.argmax(v.foot[:, 1] > 1055))
    assert len(ev) == 1 and abs(ev[0][0] - v.t[i]) < 0.25
    assert ev[0][1] <= v.end + 1e-6


def test_crossing_just_after_red_onset_is_not_red_light(red_scene):
    scene = red_scene
    # this one reaches the stop line ~0.5 s after the signal turned red
    scene.path("vehicle", CAR, 50.5 - 455 / 400, [(900, 600), (900, 2100)], 400)
    assert rules.red_light(scene.context()) == []


def test_red_light_needs_an_observable_signal(scene):
    scene.path("vehicle", CAR, 60.0, [(900, 600), (900, 2100)], 400)
    scene.set_signal((0, 120, RED))           # never green: the ROI shows no working signal
    assert rules.red_light(scene.context()) == []


def test_stop_line_violation_until_green(red_scene):
    scene = red_scene
    scene.path("vehicle", CAR, 55.0, [(900, 700), (900, 1150), (900, 2100)], 400, dwell={1: 30})
    ctx = scene.context()
    ev = labels(rules.stop_line(ctx), "stop_line")
    assert len(ev) == 1
    assert 55.5 < ev[0][0] < 57.0 and abs(ev[0][1] - 80.1) < 0.2
    assert rules.red_light(ctx) == []         # stopped past the line: not a red-light run


# -- wrong way / U-turn ------------------------------------------------------
def test_wrong_way(scene):
    v = scene.path("vehicle", CAR, 40.0, [(3800, 1800), (2000, 1800)], 400)
    ev = labels(rules.wrong_way(scene.context()), "wrong_way")
    assert len(ev) == 1
    assert ev[0][0] - v.start < 1.0 and v.end - ev[0][1] < 1.0


def test_u_turn(scene):
    ang = np.linspace(-np.pi / 2, np.pi / 2, 30)
    loop = [(3000 + 250 * np.cos(a), 1850 + 250 * np.sin(a)) for a in ang]
    scene.path("vehicle", CAR, 30.0, [(2200, 1600)] + loop + [(2200, 2100)], 300)
    ev = labels(rules.illegal_u_turn(scene.context()), "illegal_u_turn")
    assert len(ev) == 1 and 2.0 < ev[0][1] - ev[0][0] < 6.0


def test_lane_following_is_not_a_u_turn(scene):
    assert rules.illegal_u_turn(scene.context()) == []


def test_u_turn_off_the_carriageway_is_ignored(scene):
    """Regression (sample clip, fine-tuned detector): a 'car' reflected in a glass facade."""
    ang = np.linspace(-np.pi / 2, np.pi / 2, 30)
    loop = [(3300 + 150 * np.cos(a), 400 + 150 * np.sin(a)) for a in ang]
    scene.path("vehicle", CAR, 30.0, [(2900, 250)] + loop + [(2900, 550)], 300)
    assert rules.illegal_u_turn(scene.context()) == []


# -- stopped vehicle / congestion ---------------------------------------------
def test_stopped_vehicle_while_traffic_flows_past(scene):
    scene.path("vehicle", CAR, 30.0, [(2000, 1800), (2600, 1800), (3800, 1800)], 400, dwell={1: 20})
    ev = labels(rules.stopped_vehicle(scene.context()), "stopped_vehicle")
    assert len(ev) == 1 and abs(ev[0][0] - 31.5) < 0.25 and abs(ev[0][1] - 51.5) < 0.25


def test_queue_is_not_a_stopped_vehicle(quiet_scene):
    s = quiet_scene
    for k in range(4):                        # four cars queue nose to tail, nobody passes
        x = 2600 - 150 * k
        s.path("vehicle", CAR, 40.0 + k, [(2000, 1800), (x, 1800), (3800, 1800)], 400, dwell={1: 25})
    assert rules.stopped_vehicle(s.context()) == []


def _crawl(s, t0, t1):
    """Eight cars nose to tail on the avenue, creeping at 0.1 sizes/s."""
    for k in range(8):
        x0 = 2000 + 150 * k
        s.path("vehicle", CAR, t0, [(x0, 1800), (x0 + 12 * (t1 - t0), 1800)], 12)


def test_congestion_persisting_through_green(quiet_scene):
    s = quiet_scene
    s.set_signal((0, 10, RED))                # a working signal, green from 10 s on
    _crawl(s, 45, 105)
    ev = labels(rules.congestion(s.context()), "congestion")
    assert len(ev) == 1 and abs(ev[0][0] - 45) <= 1.5 and abs(ev[0][1] - 105) <= 1.5


def test_short_jam_needs_longer_evidence_without_a_signal(quiet_scene):
    _crawl(quiet_scene, 45, 105)              # 60 s, but the signal never showed red
    assert rules.congestion(quiet_scene.context()) == []


def test_red_phase_queue_is_not_congestion(quiet_scene):
    s = quiet_scene
    s.set_signal((42, 118, RED))
    _crawl(s, 45, 105)
    assert rules.congestion(s.context()) == []


# -- pedestrians -------------------------------------------------------------
def test_jaywalking_across_the_avenue(scene):
    p = scene.path("person", PERSON, 50.0, [(3000, 1500), (3000, 2100)], 80, size=150)
    ev = labels(rules.jaywalking(scene.context()), "jaywalking")
    assert len(ev) == 1
    ctx = scene.context()
    on = p.t[ctx.on_road(p.foot, inset=0.6 * p.size) & (ctx.geom.crosswalk_index(p.foot, 0.15 * p.size) < 0)]
    assert abs(ev[0][0] - on[0]) < 0.3 and abs(ev[0][1] - on[-1]) < 0.3


def test_pedestrian_on_the_zebra_is_not_jaywalking(scene):
    scene.path("person", PERSON, 50.0, [(700, 1300), (1500, 1191)], 80, size=150)
    assert rules.jaywalking(scene.context()) == []


def test_cyclist_on_the_road_is_not_a_pedestrian(scene):
    scene.path("person", PERSON, 50.0, [(2000, 1750), (3800, 1750)], 300, size=150)
    scene.path("vehicle", BICYCLE, 50.0, [(2000, 1760), (3800, 1760)], 300, size=110)
    assert rules.jaywalking(scene.context()) == []


def test_rider_waiting_in_a_lane_without_a_detected_bike_is_not_jaywalking(scene):
    """Regression (real clip, CPU model): the courier was found, his scooter was not."""
    t = times(10.0, 30.0)
    scene.add("person", PERSON, t, np.tile([[3000.0, 1800.0]], (len(t), 1)), 120)
    assert rules.jaywalking(scene.context()) == []


def test_failure_to_yield(quiet_scene):
    scene = quiet_scene
    scene.path("person", PERSON, 58.0, [(700, 1300), (1150, 1240)], 60, size=150)
    v = scene.path("vehicle", CAR, 60.0, [(900, 600), (900, 2100)], 400)
    ev = labels(rules.failure_to_yield(scene.context()), "failure_to_yield")
    inside = v.t[(v.foot[:, 1] > 1221) & (v.foot[:, 1] < 1326)]
    assert len(ev) == 1 and abs(ev[0][0] - inside[0]) < 0.3 and abs(ev[0][1] - inside[-1]) < 0.3


def test_car_standing_on_the_zebra_then_driving_on_past_a_pedestrian(quiet_scene):
    """Regression (real clip): the car stopped on the crossing and people walked round it."""
    scene = quiet_scene
    scene.path("vehicle", CAR, 60.0, [(900, 600), (900, 1280), (900, 2100)], 400, dwell={1: 10})
    scene.path("person", PERSON, 63.0, [(700, 1300), (1150, 1240)], 50, size=150)   # a slow 0.33 heights/s
    ev = labels(rules.failure_to_yield(scene.context()), "failure_to_yield")
    assert len(ev) == 1 and ev[0][0] < 61.8 and ev[0][1] > 71.7


def test_queued_car_over_the_zebra_that_waits_for_people_is_not_failure_to_yield(quiet_scene):
    scene = quiet_scene
    scene.path("vehicle", CAR, 55.0, [(900, 600), (900, 1280), (900, 2100)], 400, dwell={1: 25})
    scene.path("person", PERSON, 58.0, [(700, 1300), (1150, 1240), (1400, 1100)], 60, size=150)
    assert rules.failure_to_yield(scene.context()) == []


def test_courier_wheeling_a_scooter_over_the_zebra_is_not_failure_to_yield(quiet_scene):
    """Regression (sample clip, fine-tuned detector): the 'pedestrian' is the scooter's own rider."""
    scene = quiet_scene
    route = [(700, 1300), (1150, 1240)]
    scene.path("vehicle", MOTORCYCLE, 60.0, route, 150, size=100)
    scene.path("person", PERSON, 60.6, route, 150, size=150)    # walks 90 px behind it
    assert rules.failure_to_yield(scene.context()) == []


def test_no_failure_to_yield_when_crossing_is_empty(scene):
    assert rules.failure_to_yield(scene.context()) == []


# -- accident / obstacle -----------------------------------------------------
def test_side_impact_accident(scene):
    # At t=72 s A (driving east) hits B (coming from the north); both slide a
    # few pixels and stay where they are.
    scene.path("vehicle", CAR, 72.0 - 930 / 400, [(2000, 1800), (2930, 1800), (2940, 1800), (2941, 1800)],
               400, dwell={2: 30})
    scene.path("vehicle", CAR, 72.0 - 490 / 300, [(3000, 1300), (3000, 1790), (3003, 1800), (3004, 1800)],
               300, dwell={2: 30})
    ev = labels(rules.accident(scene.context()), "accident")
    assert len(ev) == 1
    assert abs(ev[0][0] - 72.0) < 0.3 and 1.0 <= ev[0][1] - ev[0][0] < 3.0


def test_accident_shortly_before_the_video_ends():
    """The wrecks can only be seen standing for the 4 s the video has left."""
    s = Scene(76.0)
    s.background(0, 40, every=2.0)
    s.path("vehicle", CAR, 72.0 - 930 / 400, [(2000, 1800), (2930, 1800), (2940, 1800), (2941, 1800)],
           400, dwell={2: 30})
    s.path("vehicle", CAR, 72.0 - 490 / 300, [(3000, 1300), (3000, 1790), (3003, 1800), (3004, 1800)],
           300, dwell={2: 30})
    ev = labels(rules.accident(s.context()), "accident")
    assert len(ev) == 1 and abs(ev[0][0] - 72.0) < 0.3


def test_joining_a_queue_is_not_an_accident(quiet_scene):
    s = quiet_scene
    s.path("vehicle", CAR, 40.0, [(2000, 1800), (2600, 1800), (3800, 1800)], 400, dwell={1: 40})
    # the follower brakes over ~3 s and stops just behind (boxes overlap in the image)
    t = times(45.0, 90.0)
    x = np.where(t < 48.0, 2000 + 300 * (t - 45) - 50 * (t - 45) ** 2, 2000 + 300 * 3 - 50 * 9)
    s.add("vehicle", CAR, t, np.stack([x, np.full_like(t, 1800)], 1), 120)
    assert rules.accident(s.context()) == []


def test_driving_past_a_parked_car_at_a_low_sample_rate_is_not_an_accident(quiet_scene):
    """Regression (real clip): with ~1 s between samples a moving car read as stopping dead."""
    s = quiet_scene
    s.path("vehicle", CAR, 45.0, [(2600, 1800), (2601, 1800)], 1, size=120)     # parked for 60 s
    t = times(50.0, 55.0)[::12]                                                     # one sample per 1.2 s
    x = 2000 + 300 * (t - 50)
    s.add("vehicle", CAR, t, np.stack([x, np.full_like(t, 1790)], 1), 120)
    assert rules.accident(s.context()) == []


def test_speed_survives_sparse_sampling():
    from src.tracking import Track
    from collections import Counter
    t = np.arange(0, 10, 1.2)
    box = np.stack([100 * t, np.zeros_like(t), 100 * t + 50, np.full_like(t, 50)], 1)
    tr = Track(1, "vehicle", CAR, t, box, Counter()).finalize()
    assert np.allclose(tr.speed[1:-1], 2.0, atol=0.05)
    assert np.allclose(rules.speed_with_window(tr, 0.5)[1:-1], 2.0, atol=0.05)


def test_dog_on_the_road(scene):
    scene.path("animal", DOG, 20.0, [(3000, 1500), (3000, 2100)], 150, size=60)
    ev = labels(rules.road_obstacle(scene.context()), "road_obstacle")
    assert len(ev) == 1 and ev[0][1] - ev[0][0] > 1.5


def test_backpack_mistaken_for_a_dog_is_not_an_obstacle(scene):
    scene.path("person", PERSON, 20.0, [(3000, 1500), (3000, 2100)], 80, size=180)
    scene.path("animal", DOG, 20.0, [(3000, 1440), (3000, 2040)], 80, size=50)   # on his back
    assert rules.road_obstacle(scene.context()) == []


def test_unsure_animal_is_not_an_obstacle(scene):
    dog = scene.path("animal", DOG, 20.0, [(3000, 1500), (3000, 2100)], 150, size=60)
    dog.conf = np.full(len(dog.t), 0.3, np.float32)
    assert rules.road_obstacle(scene.context()) == []


# -- post-processing -----------------------------------------------------------
def test_post_process_merges_clamps_and_drops_blips():
    out = rules.post_process([(1, 3, "jaywalking", (1,)), (2.5, 5, "jaywalking", (2,)), (5.3, 6, "jaywalking", (3,)),
                              (10, 10.2, "accident", (4, 5)), (-1, 2, "red_light", (6,)),
                              (58, 70, "congestion", ())], 60.0)
    assert out == [[0.0, 2.0, "red_light"], [1.0, 6.0, "jaywalking"], [58.0, 60.0, "congestion"]]


def test_merge_keeps_the_actors_of_merged_events():
    out = rules.merge([(1, 3, "jaywalking", (1,)), (2.5, 5, "jaywalking", (2,))], 60.0)
    assert out == [(1.0, 5.0, "jaywalking", (1, 2))]
