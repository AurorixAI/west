"""Rule engine: complete tracks + scene model -> event segments.

Part A is offline, so every rule sees whole trajectories and can place both
boundaries exactly where the annotation conventions put them (e.g. a red-light
run starts when the front of the vehicle crosses the stop line). Each rule is
deliberately specific: under macro-F1 an extra class that is not in the test
set costs as much as a missed one.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from src import config as C
from src.scene import FlowField, SceneGeometry, in_poly
from src.signal_state import GREEN, RED, SignalTimeline
from src.tracking import Track, window_mean, window_velocity

Segment = tuple[float, float, str, tuple[int, ...]]   # start, end, label, actor track ids


def runs(mask: np.ndarray, t: np.ndarray, max_gap: float) -> list[tuple[int, int]]:
    """Index ranges [i0, i1] of True samples, bridging gaps of at most max_gap seconds."""
    idx = np.nonzero(mask)[0]
    if not len(idx):
        return []
    out, s = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if t[b] - t[a] > max_gap:
            out.append((s, a))
            s = b
    out.append((s, idx[-1]))
    return out


def speed_with_window(tr: Track, window: float) -> np.ndarray:
    """Speed (sizes/s) from a shorter smoothing window than Track.speed, for abruptness tests."""
    half = window / 2
    raw = np.stack([(tr.box[:, 0] + tr.box[:, 2]) / 2, tr.box[:, 3]], 1)
    vel = window_velocity(tr.t, window_mean(tr.t, raw, half / 2), half)
    return np.linalg.norm(vel, axis=1) / np.maximum(tr.size, 1.0)


@dataclass
class Context:
    tracks: list[Track]
    geom: SceneGeometry
    flow: FlowField
    signal: SignalTimeline
    duration: float
    vehicles: list[Track] = field(init=False)
    pedestrians: list[Track] = field(init=False)
    animals: list[Track] = field(init=False)

    def __post_init__(self):
        self.vehicles = [t for t in self.tracks if t.category == "vehicle"]
        self.animals = [t for t in self.tracks if t.category == "animal"]
        # frame index: time key -> [(track, sample index)]
        self.at: dict[int, list[tuple[Track, int]]] = defaultdict(list)
        for tr in self.tracks:
            for i, t in enumerate(tr.t):
                self.at[self.key(t)].append((tr, i))
        self.pedestrians = [t for t in self.tracks if t.category == "person" and not self._is_rider(t)]

    def on_road(self, pts: np.ndarray, core: bool = False, inset: float | np.ndarray = 0.0,
                island_margin: float | np.ndarray = 0.0) -> np.ndarray:
        """Traced carriageway, or where the video's own moving vehicles drive.

        The trace covers a short clip at a red light, where the learned mask is
        still thin; the learned mask covers any road the trace missed.
        """
        return self.geom.on_carriageway(pts, inset, island_margin) | self.flow.is_road(pts, core)

    @staticmethod
    def key(t: float) -> int:
        return int(round(t * 1000))

    def _is_rider(self, p: Track) -> bool:
        """A 'person' riding a bicycle/motorcycle is traffic, not a pedestrian."""
        if np.median(p.speed) > C.RIDER_SPEED:
            return True
        hits = 0
        for i, t in enumerate(p.t):
            fx, fy = p.foot[i]
            for tr, j in self.at.get(self.key(t), ()):
                if tr.category == "vehicle" and tr.cls in C.COCO_TWO_WHEELER:
                    x1, y1, x2, y2 = tr.box[j]
                    mx, my = 0.2 * (x2 - x1), 0.2 * (y2 - y1)
                    if x1 - mx <= fx <= x2 + mx and y1 - my <= fy <= y2 + my:
                        hits += 1
                        break
        return hits >= 0.5 * len(p.t)


# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------
def stopped_vehicle(ctx: Context) -> list[Segment]:
    """Stationary >= 10 s on the carriageway while traffic overtakes it.

    Buses are left out: dwelling at a stop is their normal operation.
    """
    out = []
    for v in ctx.vehicles:
        if v.cls == C.COCO_BUS:
            continue
        for i0, i1 in runs(v.speed < C.STOP_SPEED, v.t, 1.0):
            i0, i1 = refine_stationary(v, i0, i1)
            t0, t1 = v.t[i0], v.t[i1]
            if t1 - t0 < C.STOPPED_MIN_SEC:
                continue
            loc = np.median(v.foot[i0:i1 + 1], axis=0)
            if not ctx.on_road(loc)[0]:
                continue
            if t1 - t0 < C.STOPPED_QUEUE_EXEMPT_SEC and _passers(ctx, v, loc, t0, t1) < C.STOPPED_MIN_PASSERS:
                continue                      # everybody around it is stopped too: a queue
            out.append((t0, t1, "stopped_vehicle", (v.tid,)))
    return out


def _passers(ctx: Context, v: Track, loc: np.ndarray, t0: float, t1: float) -> int:
    """Distinct vehicles that overtake the stationary ``v`` during [t0, t1].

    Overtaking = moving, along its own direction of travel, from behind ``v``
    to ahead of it, within three sizes laterally. Vehicles that join or leave
    a queue around it never get ahead of it while it is stopped, so a queue
    scores zero. Uses each passer's own heading, not the learned lanes, so it
    also works on a short clip whose flow field is still thin.
    """
    size = float(np.median(v.size))
    seen = 0
    for o in ctx.vehicles:
        if o is v or o.end < t0 or o.start > t1:
            continue
        m = (o.t >= t0) & (o.t <= t1) & (o.speed >= C.MOVE_SPEED)
        if m.sum() < 2:
            continue
        d = o.foot[m][-1] - o.foot[m][0]
        if np.linalg.norm(d) < 1e-6:
            continue
        d = d / np.linalg.norm(d)
        rel = o.foot[m] - loc
        along = rel @ d
        lateral = np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0])
        behind = np.nonzero(along < -0.5 * size)[0]
        ahead = np.nonzero(along > 0.5 * size)[0]
        if len(behind) and len(ahead) and ahead[-1] > behind[0]:
            k = behind[0] + int(np.argmin(np.abs(along[behind[0]:ahead[-1] + 1])))
            if lateral[k] <= 3 * size:
                seen += 1
    return seen


def refine_stationary(tr: Track, i0: int, i1: int, tol: float = 0.2) -> tuple[int, int]:
    """Widen a stationary run to the raw samples still at the stop location.

    The run comes from the smoothed speed, which reacts half a smoothing
    window late at both ends; the raw ground point says exactly when the
    object arrived and when it left.
    """
    raw = np.stack([(tr.box[:, 0] + tr.box[:, 2]) / 2, tr.box[:, 3]], 1)
    loc = np.median(raw[i0:i1 + 1], axis=0)
    lim = tol * float(np.median(tr.size[i0:i1 + 1]))
    while i0 > 0 and np.linalg.norm(raw[i0 - 1] - loc) <= lim:
        i0 -= 1
    while i1 < len(tr.t) - 1 and np.linalg.norm(raw[i1 + 1] - loc) <= lim:
        i1 += 1
    return i0, i1


def jaywalking(ctx: Context) -> list[Segment]:
    out = []
    for p in ctx.pedestrians:
        on_road = ctx.on_road(p.foot, core=True, inset=C.KERB_INSET * p.size,
                              island_margin=C.ISLAND_MARGIN * p.size)
        if not on_road.any():
            continue
        off_crossing = ctx.geom.crosswalk_index(p.foot, margin=C.CROSSWALK_MARGIN * p.size) < 0
        for i0, i1 in runs(on_road & off_crossing, p.t, 0.5):
            if p.t[i1] - p.t[i0] < C.JAYWALK_MIN_SEC:
                continue
            # A figure standing still in a traffic lane is a rider waiting at the
            # light whose scooter the detector missed; people on the road walk.
            if np.median(p.speed[i0:i1 + 1]) < C.JAYWALK_MIN_SPEED:
                continue
            out.append((p.t[i0], p.t[i1], "jaywalking", (p.tid,)))
    return out


def failure_to_yield(ctx: Context) -> list[Segment]:
    """A vehicle moves across a crossing while a pedestrian is on it close by.

    Stopping on the zebra itself is not yielding, so a car that stands on the
    crossing while people walk round it and then drives on is reported; a car
    queued over the zebra that pulls away once the people have gone is not.
    Bicycles are left out: they are often wheeled across with the pedestrians.
    """
    out = []
    for v in ctx.vehicles:
        if v.cls == C.COCO_BICYCLE:
            continue
        cw = ctx.geom.vehicle_crosswalk_index(v.box)
        for k in range(len(ctx.geom.crosswalks)):
            for i0, i1 in runs(cw == k, v.t, 0.3):   # a car at speed is on a zebra for ~0.3 s
                ped = _pedestrian_while_moving(ctx, v, k, i0, i1)
                if ped is not None:
                    out.append((v.t[i0], v.t[i1], "failure_to_yield", (v.tid, ped)))
    return out


def _pedestrian_while_moving(ctx: Context, v: Track, k: int, i0: int, i1: int) -> int | None:
    """Track id of a pedestrian on the carriageway part of crossing k, near the vehicle
    at a moment the vehicle is moving; None if there is none."""
    peds = set(id(p) for p in ctx.pedestrians)
    # a scooter wheeled across with the people moves at their pace
    min_speed = C.WALKING_PACE if v.cls in C.COCO_TWO_WHEELER else C.MOVE_SPEED
    for i in range(i0, i1 + 1):
        if v.speed[i] < min_speed:
            continue
        for p, j in ctx.at.get(ctx.key(v.t[i]), ()):
            if id(p) not in peds:
                continue
            pf = p.foot[j:j + 1]
            if ctx.geom.crosswalk_index(pf)[0] != k or \
                    not ctx.on_road(pf, inset=C.YIELD_KERB_INSET * p.size[j], island_margin=0.0)[0]:
                continue                      # waiting on the kerb or the island
            if np.linalg.norm(pf[0] - v.foot[i]) <= C.YIELD_MAX_DIST * v.size[i]:
                return p.tid
    return None


def _stop_line_crossings(ctx: Context, v: Track) -> list[tuple[float, int]]:
    """(time, sample index after the crossing) of robust far->near stop-line crossings."""
    side = ctx.geom.stop_line_side(v.foot)
    span = ctx.geom.on_stop_line_span(v.foot, margin=0.5 * float(np.median(v.size)))
    out = []
    for i in np.nonzero((side[:-1] < 0) & (side[1:] >= 0) & span[1:])[0] + 1:
        before = (v.t >= v.t[i] - 1.0) & (v.t < v.t[i])
        after = (v.t > v.t[i]) & (v.t <= v.t[i] + 1.0)
        if not (side[before] < -0.3 * v.size[i]).any() or not (side[after] > 0.3 * v.size[i]).any():
            continue                          # jitter around the line, not a real crossing
        a, b = side[i - 1], side[i]
        tc = v.t[i - 1] + (v.t[i] - v.t[i - 1]) * (-a / (b - a))
        out.append((float(tc), int(i)))
    return out


def red_light(ctx: Context) -> list[Segment]:
    if not ctx.signal.observable:
        return []
    out = []
    for v in ctx.vehicles:
        for tc, i in _stop_line_crossings(ctx, v):
            since = ctx.signal.red_since(tc)
            if since is None or tc - since < C.RED_MIN_SEC:
                continue
            after = (v.t >= tc) & (v.t <= tc + 3.0)
            parked = after & (v.speed < C.STOP_SPEED) & in_poly(ctx.geom.stop_zone, v.foot)
            if parked.sum() >= 5:
                continue                      # stopped past the line instead: stop_line
            out.append((tc, min(v.end, tc + C.RED_EVENT_MAX_SEC), "red_light", (v.tid,)))
    return out


def stop_line(ctx: Context) -> list[Segment]:
    if not ctx.signal.observable:
        return []
    out = []
    for v in ctx.vehicles:
        side = ctx.geom.stop_line_side(v.foot)
        if not (side < 0).any():
            continue                          # never approached from behind the line
        zone = in_poly(ctx.geom.stop_zone, v.foot) & (side > 0.2 * v.size)
        for i0, i1 in runs((v.speed < C.STOP_SPEED) & zone, v.t, 0.5):
            i0, i1 = refine_stationary(v, i0, i1)
            t0 = v.t[i0]
            if v.t[i1] - t0 < C.STOP_LINE_MIN_STOP_SEC or ctx.signal.at(t0) != RED:
                continue
            if not (side[:i0] < 0).any():
                continue
            green = ctx.signal.next_green(t0)
            out.append((t0, green if green is not None else ctx.duration, "stop_line", (v.tid,)))
    return out


def wrong_way(ctx: Context) -> list[Segment]:
    out = []
    for v in ctx.vehicles:
        moving = v.speed >= C.MOVE_SPEED
        if moving.sum() < 5:
            continue
        flow_dir, ok = ctx.flow.direction(v.foot, exclude_tid=v.tid)
        u = v.vel / np.maximum(np.linalg.norm(v.vel, axis=1, keepdims=True), 1e-9)
        cos = (u * flow_dir).sum(1)
        judged = moving & ok
        against = judged & (cos < C.WRONG_WAY_COS)
        for i0, i1 in runs(against, v.t, 1.0):
            if v.t[i1] - v.t[i0] < C.WRONG_WAY_MIN_SEC:
                continue
            seg = slice(i0, i1 + 1)
            if against[seg].sum() < 0.8 * max(judged[seg].sum(), 1):
                continue
            dist = np.linalg.norm(v.foot[i1] - v.foot[i0]) / np.median(v.size[seg])
            if dist < 3.0:
                continue
            out.append((v.t[i0], v.t[i1], "wrong_way", (v.tid,)))
    return out


def illegal_u_turn(ctx: Context) -> list[Segment]:
    out = []
    for v in ctx.vehicles:
        m = v.speed >= C.MOVE_SPEED
        if m.sum() < 10:
            continue
        t = v.t[m]
        steps = np.linalg.norm(np.diff(v.foot[m], axis=0), axis=1) / v.size[m][1:]
        if (steps > 2.0).any():
            continue                          # teleport: an ID switch, not a manoeuvre
        h = np.degrees(np.unwrap(np.arctan2(v.vel[m, 1], v.vel[m, 0])))
        best = None
        for i in range(len(t)):
            j_hi = np.searchsorted(t, t[i] + C.UTURN_MAX_SEC, "right")
            d = np.abs(h[i:j_hi] - h[i])
            j = int(np.argmax(d)) + i
            if d.max() >= C.UTURN_MIN_DEG and (best is None or d.max() > best[2]):
                best = (i, j, float(d.max()))
        if best is None:
            continue
        i, j, total = best
        d = np.abs(h[i:j + 1] - h[i])
        s = i + int(np.argmax(d >= 15.0))
        e = i + int(np.argmax(d >= total - 15.0))
        if t[e] > t[s]:
            out.append((t[s], t[e], "illegal_u_turn", (v.tid,)))
    return out


def congestion(ctx: Context) -> list[Segment]:
    dirs, groups = ctx.flow.direction_groups()
    if not len(dirs):
        return []
    n_bins = int(np.ceil(ctx.duration)) + 1
    present = np.zeros((len(dirs), n_bins))
    slow = np.zeros((len(dirs), n_bins))
    for v in ctx.vehicles:
        r, c = ctx.flow.cells(v.foot)
        g = groups[r, c]
        b = v.t.astype(int)
        for gi in range(len(dirs)):
            m = g == gi
            if not m.any():
                continue
            bins, first = np.unique(b[m], return_index=True)
            present[gi, bins] += 1
            slow[gi, bins] += v.speed[m][first] < C.MOVE_SPEED
    out = []
    t_bins = np.arange(n_bins, dtype=float)
    green = np.array([ctx.signal.at(t) == GREEN for t in t_bins]) if ctx.signal.observable else None
    for gi in range(len(dirs)):
        jam = (present[gi] >= C.CONGESTION_MIN_VEHICLES) & (slow[gi] >= C.CONGESTION_SLOW_FRAC * present[gi])
        for i0, i1 in runs(jam, t_bins, 3.0):
            dur = t_bins[i1] - t_bins[i0]
            if dur < C.CONGESTION_MIN_SEC:
                continue
            # a queue that forms on red and drains on green is not congestion
            if green is not None and green[i0:i1 + 1].sum() < 15:
                continue
            if green is None and dur < 2 * C.CONGESTION_MIN_SEC:
                continue
            out.append((t_bins[i0], min(t_bins[i1] + 1.0, ctx.duration), "congestion", ()))
    return out


def accident(ctx: Context) -> list[Segment]:
    """Contact between two road users followed by an abrupt stop of both.

    Queue-joining also brings boxes together and ends in a stop, but the
    deceleration is gradual; occlusion brings boxes together but nobody stops.
    """
    users = ctx.vehicles + ctx.pedestrians
    fast = {id(u): speed_with_window(u, 0.5) for u in users}
    out, done = [], set()
    for key in sorted(ctx.at):
        present = [(tr, i) for tr, i in ctx.at[key] if id(tr) in fast]
        if len(present) < 2:
            continue
        boxes = np.array([tr.box[i] for tr, i in present])
        x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
        y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
        x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
        y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iou = inter / np.maximum(area[:, None] + area[None, :] - inter, 1e-9)
        for a, b in zip(*np.nonzero(np.triu(iou > 0.05, 1))):
            (ta, ia), (tb, ib) = present[a], present[b]
            pair = tuple(sorted((ta.tid, tb.tid)))
            if pair in done or (ta.category == "person" and tb.category == "person"):
                continue
            ground = np.linalg.norm(ta.foot[ia] - tb.foot[ib]) / max(ta.size[ia], tb.size[ib])
            if ground > 0.8:
                continue
            t = key / 1000.0
            if _collision(ta, fast[id(ta)], tb, fast[id(tb)], t, ctx.duration):
                done.add(pair)
                settle = max(_settle_time(ta, t), _settle_time(tb, t))
                out.append((t, max(settle, t + 1.0), "accident", pair))
    return out


def _collision(ta: Track, sa: np.ndarray, tb: Track, sb: np.ndarray, t: float, end: float) -> bool:
    def mean_in(tr, s, lo, hi):
        m = (tr.t >= t + lo) & (tr.t <= t + hi)
        return float(s[m].mean()) if m.any() else np.nan

    struck = False
    for tr, s in ((ta, sa), (tb, sb)):
        before, after = mean_in(tr, s, -1.5, -0.2), mean_in(tr, s, 0.3, 1.5)
        if before >= 2 * C.MOVE_SPEED and after <= C.ACCIDENT_DROP_RATIO * before:
            struck = True
    if not struck:
        return False
    # both are seen standing where they collided; when the video ends soon
    # after the impact, whatever is left of it has to show them standing
    lo, hi = t + 3.0, min(t + 3.0 + C.ACCIDENT_STAY_SEC, end)
    if hi - lo < 0.5 * C.ACCIDENT_STAY_SEC:
        lo, hi = t + 1.5, end
        if hi - lo < C.ACCIDENT_TAIL_SEC:
            return False
    for tr in (ta, tb):
        m = (tr.t >= lo) & (tr.t <= hi)
        if m.sum() < 2 or tr.t[m][-1] - tr.t[m][0] < 0.5 * (hi - lo) \
                or np.median(tr.speed[m]) > C.STOP_SPEED:
            return False
    return True


def _settle_time(tr: Track, t: float) -> float:
    """First time after t from which the object stays stationary (or its track end)."""
    for i0, i1 in runs(tr.speed < C.STOP_SPEED, tr.t, 1.0):
        if tr.t[i1] >= t:
            i0, _ = refine_stationary(tr, i0, i1)
            return float(max(tr.t[i0], t))
    return tr.end


def road_obstacle(ctx: Context) -> list[Segment]:
    """An animal on the carriageway.

    COCO detectors readily call a backpack or a crouching child a dog, so the
    animal must be confidently detected and must not sit on a person's box.
    """
    out = []
    for a in ctx.animals:
        if a.conf is not None and np.median(a.conf) < C.OBSTACLE_MIN_CONF:
            continue
        if _on_person_fraction(ctx, a) > 0.3:
            continue
        for i0, i1 in runs(ctx.on_road(a.foot), a.t, 1.0):
            if a.t[i1] - a.t[i0] >= C.OBSTACLE_MIN_SEC:
                out.append((a.t[i0], a.t[i1], "road_obstacle", (a.tid,)))
    return out


def _on_person_fraction(ctx: Context, a: Track) -> float:
    hits = 0
    for i, t in enumerate(a.t):
        cx, cy = (a.box[i, 0] + a.box[i, 2]) / 2, (a.box[i, 1] + a.box[i, 3]) / 2
        for tr, j in ctx.at.get(ctx.key(t), ()):
            if tr.category == "person":
                x1, y1, x2, y2 = tr.box[j]
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    hits += 1
                    break
    return hits / max(len(a.t), 1)


RULES = {
    "accident": accident, "red_light": red_light, "wrong_way": wrong_way,
    "illegal_u_turn": illegal_u_turn, "stopped_vehicle": stopped_vehicle, "jaywalking": jaywalking,
    "failure_to_yield": failure_to_yield, "stop_line": stop_line, "congestion": congestion,
    "road_obstacle": road_obstacle,
}


def detect(ctx: Context, enabled=C.ENABLED_CLASSES) -> list[Segment]:
    out = []
    for name, rule in RULES.items():
        if name in enabled:
            out += rule(ctx)
    return out


def merge(events: list[Segment], duration: float, merge_gap: float = C.MERGE_GAP_SEC,
          min_len: float = C.MIN_EVENT_SEC) -> list[Segment]:
    """Clamp, merge same-class segments that overlap or nearly touch, drop blips.

    Merging concurrent same-class events into one segment is also the
    annotators' convention, and it guarantees no same-class overlap.
    """
    by_class: dict[str, list[list]] = defaultdict(list)
    for s, e, lbl, actors in events:
        s, e = max(0.0, float(s)), min(float(duration), float(e))
        if e > s:
            by_class[lbl].append([s, e, set(actors)])
    out = []
    for lbl, segs in by_class.items():
        segs.sort(key=lambda x: (x[0], x[1]))
        merged = [segs[0]]
        for s, e, actors in segs[1:]:
            if s <= merged[-1][1] + merge_gap:
                merged[-1][1] = max(merged[-1][1], e)
                merged[-1][2] |= actors
            else:
                merged.append([s, e, actors])
        out += [(round(s, 2), round(e, 2), lbl, tuple(sorted(a))) for s, e, a in merged if e - s >= min_len]
    out.sort(key=lambda x: (x[0], x[1], x[2]))
    return out


def post_process(events: list[Segment], duration: float, **kw) -> list[list]:
    """The official output format: [[start_sec, end_sec, label], ...]."""
    return [[s, e, lbl] for s, e, lbl, _ in merge(events, duration, **kw)]
