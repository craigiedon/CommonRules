from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Optional, Mapping, Dict, DefaultDict

import numpy as np

from stl import *


def gen_hor_map(spec: STLExp) -> Dict[STLExp, Tuple[int, int]]:
    h_map: Dict[STLExp, Tuple[int, int]] = {spec: (0, 0)}

    to_visit = [spec]
    while len(to_visit) > 0:
        p = to_visit.pop()
        match p:
            case (G(_, t_start, t_end) |
                  F(_, t_start, t_end) |
                  U(_, _, t_start, t_end)):
                hor = h_map[p][0] + t_start, h_map[p][1] + t_end
            case (H(_, t_start, t_end) |
                  O(_, t_start, t_end) |
                  S(_, _, t_start, t_end)):
                hor = h_map[p][0] - t_end, h_map[p][1] - t_start
            case _:
                hor = h_map[p]
        for c in stl_children(p):
            h_map[c] = hor
            to_visit.append(c)
    # Recurse or bfs/dfs-type search somehow
    return h_map


@dataclass(frozen=True)
class WorkList:
    ts: np.ndarray  # Time Stamps
    vs: np.ndarray  # Values


def init_wmap(spec: STLExp) -> Dict[STLExp, WorkList]:
    return {e: WorkList(np.array([]), np.array([])) for e in stl_tree(spec)}


def add_wl_point(wl: WorkList, t: int, x: float) -> WorkList:
    return WorkList(np.append(wl.ts, t), np.append(wl.vs, x))


def wl_neg(wl: WorkList) -> WorkList:
    return WorkList(wl.ts, -wl.vs)


def fill_p_const_signal(desired_ts: List[int], sig_ts: np.ndarray, sig_vs: np.ndarray) -> np.ndarray:
    assert len(sig_ts) == len(sig_vs)
    assert len(sig_ts) > 0

    filled_signal = []
    s_idx = 0
    for t in desired_ts:
        while s_idx < len(sig_ts) and sig_ts[s_idx] < t:
            s_idx += 1

        if s_idx >= len(sig_ts):
            filled_signal.append(sig_vs[-1])
        elif s_idx == 0:
            filled_signal.append(sig_vs[0])
        else:
            prev_dist = (sig_ts[s_idx - 1] - t) ** 2
            curr_dist = (sig_ts[s_idx] - t) ** 2
            closest_idx = s_idx if curr_dist <= prev_dist else s_idx - 1
            filled_signal.append(sig_vs[closest_idx])
    assert len(filled_signal) == len(desired_ts)
    return np.array(filled_signal)


def wl_pointwise_op(wls: List[WorkList], op: Callable) -> WorkList:

    combined_ts: List[int] = sorted(set([t for wl in wls for t in wl.ts]))

    filled_vs = np.stack([fill_p_const_signal(combined_ts, wl.ts, wl.vs) for wl in wls])

    merged_vs = op(filled_vs, axis=0)

    return WorkList(np.array(combined_ts), merged_vs)


def wl_min(wls: List[WorkList]) -> WorkList:
    return wl_pointwise_op(wls, np.min)


def wl_max(wls: List[WorkList]) -> WorkList:
    return wl_pointwise_op(wls, np.max)


def update_work_list(wl_map: Dict[STLExp, WorkList], hor_map: Dict[STLExp, Tuple[int, int]],
                     spec: STLExp, x: Any, t: int) -> None:
    match spec:
        case GEQ0(f):
            if hor_map[spec][0] <= t <= hor_map[spec][1]:
                r_val = f(x)
                wl_map[spec] = add_wl_point(wl_map[spec], t, r_val)
        case Neg(e):
            update_work_list(wl_map, hor_map, e, x, t)
            wl_map[spec] = wl_neg(wl_map[e])
        case And(exps):
            for e in exps:
                update_work_list(wl_map, hor_map, e, x, t)
            sub_wls = [wl_map[e] for e in exps if len(wl_map[e].ts) > 0]
            if len(sub_wls) > 0:
                wl_map[spec] = wl_min(sub_wls)
        case Or(exps):
            for e in exps:
                update_work_list(wl_map, hor_map, e, x, t)
            sub_wls = [wl_map[e] for e in exps if len(wl_map[e].ts) > 0]
            if len(sub_wls) > 0:
                wl_map[spec] = wl_max(sub_wls)
        case G(e, t_start_raw, t_end_raw):
            update_work_list(wl_map, hor_map, e, x, t)
            if t < hor_map[e][0]:
                print("Not yet relevant")
                return

            t_end = min(t, t_end_raw)
            assert t_start_raw <= t_end, f"start == {t_start_raw} is not less than end == {t_end} "
            width = t_end + 1 - t_start_raw
            offset_ts = wl_map[e].ts - t_start_raw

            if len(wl_map[e].ts) > 0:
                new_vs = online_min_lemire(wl_map[e].vs, offset_ts, width, -np.inf)
                wl_map[spec] = WorkList(offset_ts, new_vs)

        case F(e, t_start_raw, t_end_raw):
            update_work_list(wl_map, hor_map, e, x, t)
            if t < hor_map[e][0]:
                print("Not yet relevant")
                return
            t_end = min(t, t_end_raw)

            assert t_start_raw <= t_end, f"start == {t_start_raw} is not less than end == {t_end} "
            width = t_end + 1 - t_start_raw
            offset_ts = wl_map[e].ts - t_start_raw

            if len(wl_map[e].ts) > 0:
                new_vs = online_max_lemire(wl_map[e].vs, offset_ts, width, -np.inf)
                wl_map[spec] = WorkList(offset_ts, new_vs)
        case U(e_1, e_2, t_start_raw, t_end_raw):
            assert 0 <= t_start_raw <= t_end_raw

            update_work_list(wl_map, hor_map, e_1, x, t)
            update_work_list(wl_map, hor_map, e_2, x, t)

            if t < hor_map[e_1][0]:
                print("Not yet relevant")
                return

            t_end = min(t, t_end_raw)

            # Let's just do the lower bounds
            rvs_left = wl_map[e_1]
            rvs_right = wl_map[e_2]

            # TODO: Is this "pointwise" function actually correct?
            worst_wls = wl_pointwise_op([rvs_left, rvs_right], np.min)

            assert len(rvs_left.ts) == len(worst_wls.ts)

            until_vs = np.zeros(len(worst_wls.ts) + 1)
            until_vs[-1] = -np.inf

            for i in reversed(range(0, len(worst_wls.ts))):
                until_vs[i] = max(worst_wls.vs[i], min(rvs_left.vs[i], until_vs[i + 1]))

            wl_map[spec] = WorkList(worst_wls.ts, until_vs[:-1])
        case O(e, t_start_raw, t_end_raw):
            update_work_list(wl_map, hor_map, e, x, t)
            if t < hor_map[e][0] or t > hor_map[e][1]:
                print("Not yet relevant")
                return

            if len(wl_map[e].ts) > 0:
                width = t_end_raw + 1 - t_start_raw
                offset_ts = wl_map[e].ts + t_start_raw

                if len(wl_map[e].ts) > 0:
                    new_vs = online_max_lemire(wl_map[e].vs[::-1], offset_ts, width, -np.inf)[::-1]
                    wl_map[spec] = WorkList(offset_ts, new_vs)

        case H(e, t_start_raw, t_end_raw):
            raise NotImplementedError
        case S(e_1, e_2, t_start_raw, t_end_raw):
            raise NotImplementedError

        case _:
            raise ValueError("STL Expression Not Recognized")


def online_min_lemire(raw_xs: np.ndarray, raw_ts: np.ndarray, width:int, fill_v: float) -> np.ndarray:
    return -online_max_lemire(-raw_xs, raw_ts, width, -fill_v)


def online_max_lemire(raw_xs: np.ndarray, raw_ts: np.ndarray, width: int, fill_v: float) -> np.ndarray:
    # assert raw_ts[0] == a
    assert len(raw_xs) == len(raw_ts)
    assert width >= 1, f"width (== {width}) should be at least 1"
    assert np.isfinite(width), f"width == f{width}, Unbounded temporal operators should be dealt with before entering min/max lemire algorithm"

    # if len(raw_xs) == 4:
    #     print("Be here!")

    U = deque([0])
    window_maxs = []

    # assert 0 < width < len(raw_xs)

    # xs = np.pad(raw_xs[a:], (0, width + 1), mode='constant', constant_values=fill_v)
    # ts = np.concatenate((raw_ts, [raw_ts[-1] + time_pad for time_pad in range(1, width + 2)]))
    # xs = np.pad(raw_xs, (0, width + 1), mode='constant', constant_values=fill_v)
    ts = np.concatenate((raw_ts, [raw_ts[-1] + time_pad for time_pad in range(1, width + 1)]))
    xs = raw_xs
    # ts = raw_ts

    # Examples
    # raw_xs = [5.0], width = 1.0
    # window_maxs = [5.0]

    # raw_xs = [5.0, 2.0], width = 1.0
    # window_maxs = [5.0, 2.0]

    # raw_xs = [5.0, 10.0], width = 2.0
    # window_maxs = [10.0, 10.0]

    for i in range(1, len(xs) + width):
        t = ts[i]
        # We've seen at least the full width time-wise, so we can start appending max values now
        if i >= width:
            window_maxs.append(xs[U[-1]])

        if i < len(xs):
            # While the current x-value is bigger than the most recent maxes, keep popping
            if xs[i] > xs[i - 1]:
                U.popleft()
                while len(U) > 0 and xs[i] > xs[U[0]]:
                    U.popleft()

            # Add current value to the front of the queue
            U.appendleft(i)

        if t >= width + ts[U[-1]]:
            U.pop()

    # for i in range(1, len(xs)):
    #     t = ts[i]
    #     # We've seen at least the full width time-wise, so we can start appending max values now
    #     if t - ts[0] > width:
    #         window_maxs.append(xs[U[-1]])
    #
    #     # While the current x-value is bigger than the most recent maxes, keep popping
    #     if xs[i] > xs[i - 1]:
    #         U.popleft()
    #         while len(U) > 0 and xs[i] > xs[U[0]]:
    #             U.popleft()
    #
    #     # Add current value to the front of the queue
    #     U.appendleft(i)
    #
    #     # Slide window if the earliest value has just gone outside time frame
    #     if t > width + ts[U[-1]]:
    #         U.pop()

    assert len(window_maxs) == len(raw_xs)
    return np.array(window_maxs)


def online_run(spec, xs) -> Tuple[np.ndarray, Dict[STLExp, WorkList]]:
    h_map = gen_hor_map(spec)
    wl_map = init_wmap(spec)

    vs_history = []
    for t, s in enumerate(xs):
        update_work_list(wl_map, h_map, spec, s, t)
        # Assumes you are looking to monitor at t=0
        if len(wl_map[spec].ts) > 0:
            vs_history.append(wl_map[spec].vs[0])
        else:
            vs_history.append(-np.inf)
    print(wl_map)

    vs_history = np.array(vs_history)
    return vs_history, wl_map


def run():
    a = 2
    b = 1
    c = 3
    example_spec = G(Or((Neg(GEQ0(lambda s: s[0])), F(GEQ0(lambda s: s[1]), b, c))), 0, a)
    h_map = gen_hor_map(example_spec)
    print(h_map)
    print("Online Monitor Stuff")
    ts = np.arange(10)
    vs = np.array([0.2, 0.5, 0.3, 0.4, 0.6, 0.8, -0.1, 0.4, 1.1, 0.1])
    max_res = online_max_lemire(vs, ts, 0, 2, -np.inf)
    min_res = online_min_lemire(vs, ts, 0, 2, np.inf)
    print("vs:", vs)
    print("Maxs:", max_res)
    print("Mins:", min_res)

    # xs = np.array([1, 2, -1, -2, 2, -1])
    # ys = np.array([-1, 2, -1, 1, 1, 1])
    # states = np.column_stack((xs, ys))
    #
    # wl_map = init_wmap(example_spec)
    # for t, s in enumerate(states):
    #     update_work_list(wl_map, h_map, example_spec, s, t)
    #     print(f'{t}: lb: {wl_map[example_spec].lbs[0]} ub:{wl_map[example_spec].ubs[0]}')
    # print(wl_map)

    # another_example = G(F(GEQ0(lambda x: x), 3, 5), 0, 2)
    another_example = G(GEQ0(lambda x: x), 3, 10)

    states = np.array([100, 3000, 55, -0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
    online_run(another_example, states)

    # TODO: Sanity test that it works on an easy-to-follow example
    # TODO: Sanity test that it gives the same answer as the batch algorithm at all intermediate points (though note - I think the batch takes an "optimistic" approach to evaluations...)
    # TODO: So...does it work with your Traffic Rules? What else needs to be added in terms of expressivity?


if __name__ == "__main__":
    run()
