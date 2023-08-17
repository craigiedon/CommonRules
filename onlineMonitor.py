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
    lbs: np.ndarray  # Lower Bounds
    ubs: np.ndarray  # Upper Bounds


def init_wmap(spec: STLExp) -> Dict[STLExp, WorkList]:
    return {e: WorkList(np.array([]), np.array([]), np.array([])) for e in stl_tree(spec)}


def add_wl_point(wl: WorkList, t: int, lb: float, ub: float) -> WorkList:
    return WorkList(np.append(wl.ts, t), np.append(wl.lbs, lb), np.append(wl.ubs, ub))


def wl_neg(wl: WorkList) -> WorkList:
    return WorkList(wl.ts, -wl.lbs, -wl.ubs)


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
    wl_tuples: Dict[int, Tuple[int, int]] = {}

    combined_ts: List[int] = sorted(set([t for wl in wls for t in wl.ts]))

    filled_lbs = np.stack([fill_p_const_signal(combined_ts, wl.ts, wl.lbs) for wl in wls])
    filled_ubs = np.stack([fill_p_const_signal(combined_ts, wl.ts, wl.ubs) for wl in wls])

    merged_lbs = op(filled_lbs, axis=0)
    merged_ubs = op(filled_ubs, axis=0)

    return WorkList(np.array(combined_ts), merged_lbs, merged_ubs)


def wl_min(wls: List[WorkList]) -> WorkList:
    return wl_pointwise_op(wls, np.min)


def wl_max(wls: List[WorkList]) -> WorkList:
    return wl_pointwise_op(wls, np.max)


def update_work_list(wl_map: Dict[STLExp, WorkList], hor_map: Dict[STLExp, Tuple[int, int]],
                     spec: STLExp, x: Any, t: int) -> None:
    hor = hor_map[spec]
    match spec:
        case LEQ0(f):
            if hor[0] <= t <= hor[1]:
                r_val = -f(x)
                wl_map[spec] = add_wl_point(wl_map[spec], t, r_val, r_val)
        case GEQ0(f):
            if hor[0] <= t <= hor[1]:
                r_val = f(x)
                wl_map[spec] = add_wl_point(wl_map[spec], t, r_val, r_val)
        case Neg(e):
            update_work_list(wl_map, hor_map, e, x, t)
            wl_map[spec] = wl_neg(wl_map[e])
        case And(exps):
            for e in exps:
                update_work_list(wl_map, hor_map, e, x, t)
            sub_wls = [wl_map[e] for e in exps]
            wl_map[spec] = wl_min(sub_wls)
        case Or(exps):
            for e in exps:
                update_work_list(wl_map, hor_map, e, x, t)
            sub_wls = [wl_map[e] for e in exps]
            wl_map[spec] = wl_max(sub_wls)
        case G(e, t_start, t_end):
            update_work_list(wl_map, hor_map, e, x, t)
            if len(wl_map[e].ts) > 0:
                new_lbs = online_min_lemire(wl_map[e].lbs, wl_map[e].ts, t_start, t_end, -np.inf)
                new_ubs = online_min_lemire(wl_map[e].ubs, wl_map[e].ts, t_start, t_end, np.inf)
                wl_map[spec] = WorkList(wl_map[e].ts, new_lbs, new_ubs)
        case F(e, t_start, t_end):
            update_work_list(wl_map, hor_map, e, x, t)
            if len(wl_map[e].ts) > 0:
                new_lbs = online_max_lemire(wl_map[e].lbs, wl_map[e].ts, t_start, t_end, -np.inf)
                new_ubs = online_max_lemire(wl_map[e].ubs, wl_map[e].ts, t_start, t_end, np.inf)
                wl_map[spec] = WorkList(wl_map[e].ts, new_lbs, new_ubs)
        case U(e_1, e_2, t_start, t_end):
            assert 0 <= t_start <= t_end
            assert np.isinf(t_end)

            update_work_list(wl_map, hor_map, e_1, x, t)
            update_work_list(wl_map, hor_map, e_2, x, t)

            # Lets just do the lower bounds
            rvs_left = wl_map[e_1]
            rvs_right = wl_map[e_2]

            # TODO: Is this "pointwise" function actually correct?
            worst_wls = wl_pointwise_op([rvs_left, rvs_right], min)

            assert len(rvs_left.ts) == len(worst_wls.ts)

            until_lbs = np.zeros(len(worst_wls.ts) + 1)
            until_lbs[-1] = -np.inf
            for i in reversed(range(0, len(worst_wls.ts))):
                until_lbs[i] = max(worst_wls.lbs[i], min(rvs_left[i], until_lbs[i + 1]))

            until_ubs = np.zeros(len(worst_wls.ts) + 1)
            until_ubs[-1] = np.inf
            for i in reversed(range(0, len(worst_wls.ts))):
                until_ubs[i] = max(worst_wls.ubs[i], min(rvs_left[i], until_ubs[i + 1]))

            wl_map[spec] = WorkList(worst_wls.ts, until_lbs[:-1], until_ubs[:-1])
        case O(e, t_start, t_end):
            update_work_list(wl_map, hor_map, e, x, t)
            if len(wl_map[e].ts) > 0:
                raise NotImplementedError

        case H(e, t_start, t_end):
            update_work_list(wl_map, hor_map, e, x, t)
            if len(wl_map[e].ts) > 0:
                raise NotImplementedError
        case S(e_1, e_2, t_start, t_end):
            update_work_list(wl_map, hor_map, e_1, x, t)
            update_work_list(wl_map, hor_map, e_2, x, t)
            raise NotImplementedError

        case _:
            raise ValueError("STL Expression Not Recognized")


def online_min_lemire(raw_xs: np.ndarray, raw_ts: np.ndarray, a: float, b: float, fill_v: float) -> np.ndarray:
    return -online_max_lemire(-raw_xs, raw_ts, a, b, -fill_v)


def online_max_lemire(raw_xs: np.ndarray, raw_ts: np.ndarray, a: int, b: int, fill_v: float) -> np.ndarray:
    assert a < b
    # assert raw_ts[0] == a
    assert len(raw_xs) == len(raw_ts)
    # TODO: Support unbounded b

    U = deque([0])
    window_maxs = []

    # fill_v = -np.inf
    width = b - a
    # assert a == 0 # TODO: Support non-zero a-values safely
    # assert 0 < width < len(raw_xs)

    # xs = np.pad(raw_xs[a:], (0, width + 1), mode='constant', constant_values=fill_v)
    # ts = np.concatenate((raw_ts, [raw_ts[-1] + time_pad for time_pad in range(1, width + 2)]))
    xs = np.pad(raw_xs, (0, width + 1), mode='constant', constant_values=fill_v)
    ts = np.concatenate((raw_ts, [raw_ts[-1] + time_pad for time_pad in range(1, width + 2)]))

    for i in range(1, len(xs)):
        t = ts[i]
        # We've seen at least the full width time-wise, so we can start appending max values now
        if t - ts[0] > width:
            window_maxs.append(xs[U[-1]])

        # While the current x-value is bigger than the most recent maxes, keep popping
        if xs[i] > xs[i - 1]:
            U.popleft()
            while len(U) > 0 and xs[i] > xs[U[0]]:
                U.popleft()

        # Add current value to the front of the queue
        U.appendleft(i)

        # Slide window if the earliest value has just gone outside time frame
        if t > width + ts[U[-1]]:
            U.pop()

    assert len(window_maxs) == len(raw_xs)
    return np.array(window_maxs)


def online_run(spec, xs) -> Tuple[np.ndarray, np.ndarray, Dict[STLExp, WorkList]]:
    h_map = gen_hor_map(spec)
    wl_map = init_wmap(spec)

    lb_history = []
    ub_history = []
    for t, s in enumerate(xs):
        update_work_list(wl_map, h_map, spec, s, t)
        if len(wl_map[spec].ts) > 0:
            # print(f'{t}: lb: {wl_map[spec].lbs[0]} ub:{wl_map[spec].ubs[0]}')
            lb_history.append(wl_map[spec].lbs[0])
            ub_history.append(wl_map[spec].ubs[0])
        else:
            # print(f'{t}: lb: {-np.inf} ub:{np.inf}')
            lb_history.append(-np.inf)
            ub_history.append(np.inf)
    print(wl_map)

    lb_history = np.array(lb_history)
    ub_history = np.array(ub_history)
    return lb_history, ub_history, wl_map


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
