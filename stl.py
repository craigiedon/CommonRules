import abc
import dataclasses
from collections import deque
from dataclasses import dataclass
from typing import Callable, Any, List, Optional, Union, Tuple, Sequence
import numpy as np
from scipy.special import logsumexp


# Normalizes between -1 and 1
def range_norm(x: float, min_v: float, max_v: float) -> float:
    assert max_v > min_v
    return 2.0 * (x - min_v) / (max_v - min_v) - 1.0


class STLExp(abc.ABC):
    pass


@dataclass(frozen=True)
class Tru(STLExp):
    pass


@dataclass(frozen=True)
class GEQ0(STLExp):
    f: Callable


def GEQc(f: Callable, c: float) -> STLExp:
    return GEQ0(lambda x: f(x) - c)


def GEQ(f: Callable, g: Callable) -> STLExp:
    return GEQ0(lambda x: f(x) - g(x))


def LEQ0(f: Callable) -> STLExp:
    return GEQ0(lambda x: -f(x))


def LEQc(f: Callable, c: float) -> STLExp:
    return LEQ0(lambda x: f(x) - c)


def LEQ(f: Callable, g: Callable) -> STLExp:
    return LEQ0(lambda x: f(x) - g(x))


@dataclass(frozen=True)
class Neg(STLExp):
    e: STLExp


@dataclass(frozen=True)
class And(STLExp):
    exps: Tuple[STLExp, ...]


@dataclass(frozen=True)
class Or(STLExp):
    exps: Tuple[STLExp, ...]


def Implies(e_lhs: STLExp, e_rhs: STLExp) -> STLExp:
    return Or((Neg(e_lhs), e_rhs))


@dataclass(frozen=True)
class G(STLExp):
    """Globally / Always"""
    e: STLExp
    t_start: int
    t_end: int


@dataclass(frozen=True)
class F(STLExp):
    """Finally / Eventually"""
    e: STLExp
    t_start: int
    t_end: int


@dataclass(frozen=True)
class U(STLExp):
    """Until"""
    e_1: STLExp
    e_2: STLExp
    t_start: int
    t_end: int


@dataclass(frozen=True)
class S(STLExp):
    """Since"""
    e_1: STLExp
    e_2: STLExp
    t_start: int
    t_end: int


@dataclass(frozen=True)
class O(STLExp):
    """Once"""
    e: STLExp
    t_start: int
    t_end: int


@dataclass(frozen=True)
class H(STLExp):
    """Historically/Previously"""
    e: STLExp
    t_start: int
    t_end: int


def stl_children(spec: STLExp) -> Sequence[STLExp]:
    match spec:
        case And(exps) | Or(exps):
            return exps
        case U(e1, e2, _) | S(e1, e2, _):
            return [e1, e2]
        case F(e, _, _) | G(e, _, _) | Neg(e) | H(e, _, _) | O(e, _, _):
            return [e]
        case GEQ0() | Tru():
            return []
        case _:
            raise ValueError("Unknown STL Case")


def stl_tree(spec: STLExp) -> Sequence[STLExp]:
    listed_tree = []
    to_visit = [spec]
    while len(to_visit) > 0:
        p = to_visit.pop()
        listed_tree.append(p)
        to_visit.extend(stl_children(p))
    return listed_tree


def remove_nones() -> List:
    return list(filter(lambda v: v is not None, x))


def comp_sat_tru(x) -> np.ndarray:
    return np.full(len(x), np.inf)


def comp_sat_compare(spec: GEQ0, xs) -> np.ndarray:
    return np.array([spec.f(x) for x in xs])


def comp_sat_neg(x) -> np.ndarray:
    return -x


def comp_sat_compose(spec: Union[Or, And], xs: List[np.ndarray]) -> np.ndarray:
    match spec:
        case Or(_):
            return np.max(xs, axis=0)
        case And(_):
            return np.min(xs, axis=0)


def comp_sat_and(xs: List[np.ndarray]) -> np.ndarray:
    return np.min(xs, dim=1)


def lemire_min_max(xs: np.ndarray, width: int) -> Tuple[np.ndarray, np.ndarray]:
    assert 0 < width < len(xs)
    U = deque([0])
    L = deque([0])

    window_mins = []
    window_maxs = []

    for i in range(1, len(xs)):
        if i >= width:
            window_maxs.append(xs[U[-1]])
            window_mins.append(xs[L[-1]])

        if xs[i] > xs[i - 1]:
            U.popleft()
            while len(U) > 0 and xs[i] > xs[U[0]]:
                U.popleft()
        else:
            L.popleft()
            while len(L) > 0 and xs[i] < xs[L[0]]:
                L.popleft()

        U.appendleft(i)
        L.appendleft(i)

        if i == width + U[-1]:
            U.pop()
        elif i == width + L[-1]:
            L.pop()

    # window_maxs.append(xs[U[-1]])
    # window_mins.append(xs[L[-1]])

    p_w_mins = np.pad(np.array(window_mins), (0, len(xs) - len(window_mins)), mode='edge')
    p_w_maxs = np.pad(np.array(window_maxs), (0, len(xs) - len(window_maxs)), mode='edge')
    return p_w_mins, p_w_maxs


def comp_sat_eventually(a, b, x: np.ndarray) -> np.ndarray:
    assert 0 <= a <= b, f"Invalid interval bounds [{a}, {b}]"
    # Unbounded Eventually
    if a == 0 and np.isinf(b):
        return np.fmax.accumulate(x[::-1])[::-1]
    elif a > 0 and np.isinf(b):
        padded = np.pad(x, (0, a), mode='edge')
        unshifted = np.fmax.accumulate(padded[::-1])[::-1]
        return unshifted[a:]
    # Bounded Eventually
    elif np.isfinite(b):
        if a == b:
            w_maxs = np.pad(x, (0, a), mode='edge')[a:]
        else:
            _, w_maxs = lemire_min_max(np.pad(x, (0, a), mode='edge')[a:], b - a)
        return w_maxs


def comp_sat_once(a, b, x: np.ndarray) -> np.ndarray:
    assert 0 <= a <= b, f"Invalid interval bounds [{a}, {b}]"
    # Unbounded Once
    if a == 0 and np.isinf(b):
        return np.fmax.accumulate(x)
    elif a > 0 and np.isinf(b):
        padded = np.pad(x, (a, 0), mode='edge')
        unshifted = np.fmax.accumulate(padded)
        return unshifted[:-a]
    # Bounded Once
    elif np.isfinite(b):
        # Note: Have to account for accidentally slicing with a zero
        if a == 0:
            pad_shift = x
        else:
            pad_shift = np.pad(x, (a, 0), mode='edge')[:-a]

        if a == b:
            w_maxs = pad_shift
        else:
            w_maxs = lemire_min_max(pad_shift[::-1], b - a)[1][::-1]

        return w_maxs


def comp_sat_history(a, b, x: np.ndarray) -> np.ndarray:
    assert 0 <= a <= b, f"Invalid interval bounds [{a}, {b}]"
    # Unbounded History
    if a == 0 and np.isinf(b):
        return np.fmin.accumulate(x)
    elif a > 0 and np.isinf(b):
        padded = np.pad(x, (a, 0), mode='edge')
        unshifted = np.fmin.accumulate(padded)
        return unshifted[:-a]
    # Bounded Once
    elif np.isfinite(b):
        if a == 0:
            pad_shift = x
        else:
            pad_shift = np.pad(x, (a, 0), mode='edge')[:-a]

        if a == b:
            w_mins = pad_shift
        else:
            w_mins = lemire_min_max(pad_shift[::-1], b - a)[0][::-1]

        return w_mins


def comp_sat_always(a, b, x: np.ndarray) -> np.ndarray:
    assert 0 <= a <= b, f"Invalid interval bounds [{a}, {b}]"
    # Unbounded Always
    if a == 0 and np.isinf(b):
        return np.fmin.accumulate(x[::-1])[::-1]
    elif a > 0 and np.isinf(b):
        padded = np.pad(x, (0, a), mode='edge')
        unshifted = np.fmin.accumulate(padded[::-1])[::-1]
        return unshifted[a:]
    # Bounded Eventually
    elif np.isfinite(b):
        if a == b:
            w_mins = np.pad(x, (0, a), mode='edge')[a:]
        else:
            w_mins, _ = lemire_min_max(np.pad(x, (0, a), mode='edge'), b - a)
        return w_mins


def comp_sat_until(a, b, w_left, w_right) -> np.ndarray:
    assert 0 <= a < b and b > 0, f"Invalid interval bounds [{a}, {b}]"
    # Unbounded Until
    if a == 0 and np.isinf(b):
        ys = np.zeros(len(w_left))
        worst_ws = np.fmin(w_left, w_right)
        ys[-1] = worst_ws[-1]
        for i in reversed(range(0, len(ys) - 1)):
            ys[i] = np.fmax(worst_ws[i], np.fmin(w_left[i], ys[i + 1]))
        return ys
    elif a > 0 and np.isfinite(b):
        w1 = comp_sat_until(0, np.inf, w_left, w_right)
        w2 = comp_sat_always(0, a, w1)
        w3 = comp_sat_eventually(a, b, w_right)
        return comp_sat_and([w2, w3])


def comp_sat_since(a, b, w_left, w_right) -> np.ndarray:
    assert 0 <= a < b and b > 0, f"Invalid interval bounds [{a}, {b}]"
    # Unbounded Since
    if a == 0 and np.isinf(b):
        worst_ws = np.fmin(w_left, w_right)
        ys = np.zeros(len(w_left))
        ys[0] = worst_ws[0]
        for i in range(1, len(ys)):
            ys[i] = np.fmax(worst_ws[i], np.fmin(w_left[i], ys[i - 1]))
        return ys
    # Bounded Since
    elif a > 0 and np.isfinite(b):
        w1 = comp_sat_since(0, np.inf, w_left, w_right)
        w2 = comp_sat_history(0, a, w1)
        w3 = comp_sat_once(a, b, w_right)
        return comp_sat_and([w2, w3])


def stl_rob(spec: STLExp, x: Any, t: int) -> float:
    rob_signal = stl_monitor_fast(spec, x)
    return rob_signal[t]
    # return stl_monitor_fast(spec, x)[t]


def unbound_signal(t_end, x) -> float:
    return t_end if t_end < len(x) else np.inf


def stl_monitor_fast(spec: STLExp, x: Any):
    match spec:
        case Tru():
            return comp_sat_tru(x)
        case GEQ0(_):
            return comp_sat_compare(spec, x)
        case Neg(e):
            w = stl_monitor_fast(e, x)
            return comp_sat_neg(w)
        case Or(exps) | And(exps):
            ws = [stl_monitor_fast(e, x) for e in exps]
            return comp_sat_compose(spec, ws)
        case G(e, t_start, t_end):
            w = stl_monitor_fast(e, x)
            return comp_sat_always(t_start, unbound_signal(t_end, w), w)
        case F(e, t_start, t_end):
            w = stl_monitor_fast(e, x)
            return comp_sat_eventually(t_start, unbound_signal(t_end, x), w)
        case O(e, t_start, t_end):
            w = stl_monitor_fast(e, x)
            return comp_sat_once(t_start, unbound_signal(t_end, w), w)
        case H(e, t_start, t_end):
            w = stl_monitor_fast(e, x)
            return comp_sat_history(t_start, unbound_signal(t_end, x), w)
        case U(e1, e2, t_start, t_end):
            w1 = stl_monitor_fast(e1, x)
            w2 = stl_monitor_fast(e2, x)
            return comp_sat_until(t_start, unbound_signal(t_end, x), w1, w2)
        case S(e1, e2, t_start, t_end):
            w1 = stl_monitor_fast(e1, x)
            w2 = stl_monitor_fast(e2, x)
            return comp_sat_since(t_start, unbound_signal(t_end, x), w1, w2)

        case _:
            raise ValueError(f"Invalid spec: : {spec} of type {type(spec)}")


# def stl_rob(spec: STLExp, x: Any, t: int) -> Optional[float]:
#     match spec:
#         case Tru():
#             return np.inf
#         case GEQ0(f):
#             return f(x[t])
#         case LEQ0(f):
#             return -f(x[t])
#         case Neg(e):
#             return -stl_rob(e, x, t)
#         case And(exps):
#             rob_vals = remove_nones([stl_rob(e, x, t) for e in exps])
#             if len(rob_vals) == 0:
#                 return None
#             return np.min(rob_vals)
#         case Or(exps):
#             rob_vals = remove_nones([stl_rob(e, x, t) for e in exps])
#             if len(rob_vals) == 0:
#                 return None
#             return np.max(rob_vals)
#         case G(e, t_start, t_end):
#             g_interval = range(t + t_start, min(t + t_end + 1, len(x)))
#             if len(g_interval) == 0:
#                 return None
#             rob_vals = remove_nones([stl_rob(e, x, a) for a in g_interval])
#             return np.min(rob_vals)
#         case H(e, t_start, t_end):
#             h_interval = range(max(t - t_end, 0), t - t_start + 1)
#             if len(h_interval) == 0:
#                 return None
#             rob_vals = remove_nones([stl_rob(e, x, a) for a in h_interval])
#             return np.min(rob_vals)
#         case F(e, t_start, t_end):
#             f_interval = range(t + t_start, min(t + t_end + 1, len(x)))
#             if len(f_interval) == 0:
#                 return None
#             rob_vals = remove_nones([stl_rob(e, x, a) for a in f_interval])
#             return np.max(rob_vals)
#         case O(e, t_start, t_end):
#             o_interval = range(max(t - t_end, 0), t - t_start + 1)
#             if len(o_interval) == 0:
#                 return None
#             rob_vals = remove_nones([stl_rob(e, x, a) for a in o_interval])
#             return np.max(rob_vals)
#         case U(e_1, e_2, t_start, t_end):
#             u_interval = range(t + t_start, min(t + t_end + 1, len(x)))
#             if len(u_interval) == 0:
#                 return None
#
#             lhs = remove_nones([stl_rob(e_1, x, a) for a in u_interval])
#             lhs_cums = [np.min(lhs[:k + 1]) for k in range(len(lhs))]
#             rhs = remove_nones([stl_rob(e_2, x, a) for a in u_interval])
#
#             assert len(lhs) == len(rhs), f"Ill formed 'Until' ({spec}) - lhs:{len(lhs)}, rhs:{len(rhs)}"
#
#             running_vals = [min(r, lc) for r, lc in zip(rhs, lhs_cums)]
#
#             return np.max(running_vals)
#         case S(e_1, e_2, t_start, t_end):
#             s_interval = range(max(t - t_end, 0), t - t_start + 1)
#             if len(s_interval) == 0:
#                 return None
#
#             lhs = remove_nones([stl_rob(e_1, x, a) for a in s_interval])
#             lhs_cums = [np.min(lhs[k:]) for k in range(len(lhs))]
#             rhs = remove_nones([stl_rob(e_2, x, a) for a in s_interval])
#
#             assert len(lhs) == len(rhs), f"Ill formed 'Since' ({spec}) - lhs:{len(lhs)}, rhs:{len(rhs)}"
#
#             running_vals = [min(r, lc) for r, lc in zip(rhs, lhs_cums)]
#
#             return np.max(running_vals)
#
#         case _:
#             raise ValueError(f"Invalid spec: : {spec} of type {type(spec)}")


# # Smooth approximation functions for max/min operations
def smooth_min(xs: np.ndarray, b: float) -> float:
    assert b > 1.0
    xs_weighted = -b * xs
    lsexp = logsumexp(xs_weighted)
    sm = -(1.0 / b) * lsexp
    return sm


def smooth_max(xs: np.ndarray, b: float) -> float:
    assert b > 1.0
    return -smooth_min(-xs, b)


def rect_pos(x: float, b: float) -> float:
    # rp = smooth_max(np.array([x, 0.0]), b)
    rp = (1 / b) * logsumexp([0.0, b * x])
    return rp


def rect_neg(x: float, b: float) -> float:
    rp = -(1 / b) * logsumexp([0.0, -b * x])
    return rp


# Haghighi, Medhipoor, Bartocci, Belta 2019 Smooth Cumulative
def sc_rob_pos(spec: STLExp, x, t: int, b: float) -> float:
    if isinstance(spec, Tru):
        return np.inf
    if isinstance(spec, GEQ0):
        return rect_pos(spec.f(x[t]), b)
    if isinstance(spec, Neg):
        return -sc_rob_neg(spec.e, x, t, b)
    if isinstance(spec, And):
        return smooth_min(np.array([sc_rob_pos(e, x, t, b) for e in spec.exps]), b)
    if isinstance(spec, Or):
        return smooth_max(np.array([sc_rob_pos(spec.e_1, x, t, b), sc_rob_pos(spec.e_2, x, t, b)]), b)
    if isinstance(spec, G):
        rob_vals = np.array([sc_rob_pos(spec.e, x, t + k, b) for k in range(spec.t_start, spec.t_end + 1)])
        return smooth_min(rob_vals, b)
    if isinstance(spec, F):
        # Note here that the "Finally" numbers accumulate, rather than opting for a more intuitive averaging
        return np.sum(np.array([sc_rob_pos(spec.e, x, t + k, b) for k in range(spec.t_start, spec.t_end + 1)]))
    if isinstance(spec, U):
        rob_vals = []
        for k_1 in range(spec.t_start, spec.t_end + 1):
            rhs = sc_rob_pos(spec.e_2, x, t + k_1, b)
            lhs = smooth_min(np.array([sc_rob_pos(spec.e_1, x, t + k_2, b) for k_2 in range(k_1 + 1)]), b)
            rob_vals.append(smooth_min(np.array([rhs, lhs]), b))
        return np.sum(rob_vals)

    raise ValueError(f"Invalid spec: : {spec} of type {type(spec)}")


def sc_rob_neg(spec: STLExp, x, t: int, b: float) -> float:
    if isinstance(spec, Tru):
        return 0.0
    if isinstance(spec, GEQ0):
        return rect_neg(spec.f(x[t]), b)
    if isinstance(spec, Neg):
        return -sc_rob_pos(spec.e, x, t, b)
    if isinstance(spec, And):
        return smooth_min(np.array([sc_rob_neg(e, x, t, b) for e in spec.exps]), b)
    if isinstance(spec, Or):
        return smooth_max(np.array([sc_rob_neg(spec.e_1, x, t, b), sc_rob_neg(spec.e_2, x, t, b)]), b)
    if isinstance(spec, G):
        return smooth_min(np.array([sc_rob_neg(spec.e, x, t + k, b) for k in range(spec.t_start, spec.t_end + 1)]), b)
    if isinstance(spec, F):
        # Note here that the "Finally" numbers accumulate, rather than opting for a more intuitive averaging
        return np.sum(np.array([sc_rob_neg(spec.e, x, t + k, b) for k in range(spec.t_start, spec.t_end + 1)]))
    if isinstance(spec, U):
        rob_vals = []
        for k_1 in range(spec.t_start, spec.t_end + 1):
            rhs = sc_rob_neg(spec.e_2, x, t + k_1, b)
            lhs = smooth_min(np.array([sc_rob_neg(spec.e_1, x, t + k_2, b) for k_2 in range(k_1 + 1)]), b)
            rob_vals.append(smooth_min(np.array([rhs, lhs]), b))
        return np.sum(rob_vals)

    raise ValueError(f"Invalid spec: : {spec} of type {type(spec)}")


def classic_to_agm_norm(spec: STLExp, low: float, high: float) -> STLExp:
    if isinstance(spec, Tru):
        return spec
    if isinstance(spec, GEQ0):
        return dataclasses.replace(spec, f=lambda *args: 2 * range_norm(spec.f(*args), low, high))
    if isinstance(spec, (Neg, G, F)):
        return dataclasses.replace(spec, e=classic_to_agm_norm(spec.e, low, high))
    if isinstance(spec, And):
        return dataclasses.replace(spec, exps=[classic_to_agm_norm(e, low, high) for e in spec.exps])
    if isinstance(spec, (Or, U)):
        return dataclasses.replace(spec, e_1=classic_to_agm_norm(spec.e_1, low, high),
                                   e_2=classic_to_agm_norm(spec.e_2, low, high))


# Arithmetic-Geometric Mean Robustness
# Core assumption: signal values (x) are normalized between [-1, 1]
def agm_rob(spec: STLExp, x, t: int) -> float:
    if isinstance(spec, Tru):
        return 1.0
    if isinstance(spec, GEQ0):
        return 0.5 * spec.f(x[t])
    if isinstance(spec, Neg):
        return -agm_rob(spec.e, x, t)
    if isinstance(spec, And):
        robs = np.array([agm_rob(e, x, t) for e in spec.exps])
        m = len(spec.exps)
        if np.any(robs <= 0):
            return (1.0 / m) * np.sum([np.minimum(0.0, r) for r in robs])
        else:
            return np.prod([1 + r for r in robs]) ** (1.0 / m) - 1.0
    if isinstance(spec, Or):
        left_rob = agm_rob(spec.e_1, x, t)
        right_rob = agm_rob(spec.e_2, x, t)
        m = 2.0
        if left_rob >= 0.0 or right_rob >= 0.0:
            return (1.0 / m) * np.sum([np.maximum(0, r) for r in [left_rob, right_rob]])
        else:
            return 1.0 - np.prod([1.0 - r for r in [left_rob, right_rob]]) ^ (1.0 / m)
    if isinstance(spec, G):
        new_start = t + spec.t_start
        new_end = np.minimum(t + spec.t_end + 1, len(x))
        robustness_scores = [agm_rob(spec.e, x, new_t) for new_t in range(new_start, new_end)]
        N = len(robustness_scores)
        if any([r <= 0.0 for r in robustness_scores]):
            return (1.0 / N) * np.sum([np.minimum(0.0, r) for r in robustness_scores])
        else:
            return np.prod([1.0 + r for r in robustness_scores]) ** (1.0 / N) - 1.0
    if isinstance(spec, F):
        robustness_scores = [agm_rob(spec.e, x, t + k) for k in range(spec.t_start, spec.t_end + 1)]
        N = len(robustness_scores)
        if any([r > 0.0 for r in robustness_scores]):
            return (1.0 / N) * np.sum([np.maximum(0.0, r) for r in robustness_scores])
        else:
            return 1.0 - np.prod([1.0 - r for r in robustness_scores]) ** (1 / N)
    if isinstance(spec, U):
        raise NotImplementedError("Havent yet translated the code for agm 'Until' case")

    raise ValueError(f"Invalid spec: : {spec} of type {type(spec)}")
