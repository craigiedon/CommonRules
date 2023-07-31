from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Optional, Mapping, Dict, DefaultDict

import numpy as np

from stl import *


def horizon(spec: STLExp, parent: Optional[STLExp], running_hor: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    if parent is None:
        return 0, 0
    match parent:
        case G(e, t_start, t_end) | F(e, t_start, t_end) | U(e, t_start, t_end):
            return running_hor[0] + t_start, running_hor[1] + t_end
        case _:
            return running_hor


@dataclass(frozen=True)
class WorkList:
    ts: np.ndarray  # Time Stamps
    lbs: np.ndarray  # Lower Bounds
    ubs: np.ndarray  # Upper Bounds


def add_wl_point(wl: WorkList, t, lb, ub) -> WorkList:
    return WorkList(np.append(wl.ts, t), np.append(wl.lbs, lb), np.append(wl.ubs, ub))


def wl_neg(wl: WorkList) -> WorkList:
    return WorkList(wl.ts, -wl.lbs, -wl.ubs)


def wl_pointwise_op(wls: List[WorkList], op: Callable) -> WorkList:
    wl_tuples: Dict[int, Tuple[int, int]] = {}

    for wl in wls:
        for t, lb, ub in zip(wl.ts, wl.lbs, wl.ubs):
            if t in wl_tuples:
                wl_tuples[t] = op(wl_tuples[t][0], lb), op(wl_tuples[t][1], ub)
            else:
                wl_tuples[t] = (lb, ub)

    new_ts = np.array(list(wl_tuples.keys()))
    new_bounds = np.asarray(list(wl_tuples.values()))

    return WorkList(new_ts, new_bounds[:, 0], new_bounds[:, 1])


def wl_min(wls: List[WorkList]) -> WorkList:
    return wl_pointwise_op(wls, min)


def wl_max(wls: List[WorkList]) -> WorkList:
    return wl_pointwise_op(wls, max)


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
            wl_map[spec] = wl_pointwise_op(sub_wls, min)
        case Or(exps):
            for e in exps:
                update_work_list(wl_map, hor_map, e, x, t)
            sub_wls = [wl_map[e] for e in exps]
            wl_map[spec] = wl_pointwise_op(sub_wls, max)
        case G(e, t_start, t_end):
            update_work_list(wl_map, hor_map, e, x, t)
            new_lbs = online_sliding_max(wl_map[e].ts, wl_map[e].lbs, t_start, t_end)
            new_ubs = online_sliding_max(wl_map[e].ts, wl_map[e].ubs, t_start, t_end)
            # TODO: Is it actually max? Don't we want "min" for always? Im a little confused...
            # TODO: Answer: No! Its sliding min! (But...you can get this via negations...)
            wl_map[spec] = online_sliding_max(wl_map[e], t_start, t_end)


def online_sliding_max(ts: np.ndarray, vs: np.ndarray, a: int, b: int) -> np.ndarray:
    """
        [a,b] - The lower and upper bounds of the time interval G_[a, b]
        F - the frontier: set of times of the "descending monotonic edge"
    """
    assert a <= b
    assert len(ts) == len(vs)
    F = [0]
    i = 0

    s = t = ts[0] - b

    ys: Dict[int, float] = {}

    while t + a < ts[-1]:
        if len(F) > 0:
            t = min(ts[min(F)] - a, ts[i + 1] - b)
        else:
            t = ts[i + 1] - b

        if t == ts[i + 1] - b:
            # Remove every element indexed by the tail of F that is smaller than x_{i+1}
            while len(F) > 0 and vs[i + 1] >= vs[max(F)]:
                F.remove(max(F))
            F.append(i + 1)
            i = i + 1
        else:  # Slide window to the right
            if s > ts[0]:
                ys[s] = vs[min(F)]
            else:
                ys[ts[0]] = vs[min(F)]
            F.remove(min(F))
            s = t

    assert len(list(ys.values())) == len(ts)
    return np.array(list(ys.values()))


if __name__ == "__main__":
    print("Online Monitor Stuff")
    ts = np.arange(10)
    vs = np.array([0.2, 0.5, 0.3, 0.4, 0.6, 0.8, -0.1, 0.4, 1.1, 0.1])
    res = online_sliding_max(ts, vs, 2, 5)
