import abc
import dataclasses
from dataclasses import dataclass
from typing import Callable, Any, List, Optional
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


@dataclass(frozen=True)
class LEQ0(STLExp):
    f: Callable


@dataclass(frozen=True)
class Neg(STLExp):
    e: STLExp


@dataclass(frozen=True)
class And(STLExp):
    exps: List[STLExp]


@dataclass(frozen=True)
class Or(STLExp):
    exps: List[STLExp]


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


def remove_nones(x: List) -> List:
    return list(filter(lambda v: v is not None, x))


def stl_rob(spec: STLExp, x: Any, t: int) -> Optional[float]:
    match spec:
        case Tru():
            return np.inf
        case GEQ0(f):
            return f(x[t])
        case LEQ0(f):
            return -f(x[t])
        case Neg(e):
            return -stl_rob(e, x, t)
        case And(exps):
            return np.min([stl_rob(e, x, t) for e in exps])
        case Or(exps):
            return np.max([stl_rob(e, x, t) for e in exps])
        case G(e, t_start, t_end):
            g_interval = range(t + t_start, min(t + t_end + 1, len(x)))
            if len(g_interval) == 0:
                return None
            rob_vals = remove_nones([stl_rob(e, x, a) for a in g_interval])
            return np.min(rob_vals)
        case H(e, t_start, t_end):
            h_interval = range(max(t - t_end, 0), t - t_start + 1)
            if len(h_interval) == 0:
                return None
            rob_vals = remove_nones([stl_rob(e, x, a) for a in h_interval])
            return np.min(rob_vals)
        case F(e, t_start, t_end):
            f_interval = range(t + t_start, min(t + t_end + 1, len(x)))
            if len(f_interval) == 0:
                return None
            rob_vals = remove_nones([stl_rob(e, x, a) for a in f_interval])
            return np.max(rob_vals)
        case O(e, t_start, t_end):
            o_interval = range(max(t - t_end, 0), t - t_start + 1)
            if len(o_interval) == 0:
                return None
            rob_vals = remove_nones([stl_rob(e, x, a) for a in o_interval])
            return np.max(rob_vals)
        case U(e_1, e_2, t_start, t_end):
            u_interval = range(t + t_start, min(t + t_end + 1, len(x)))
            if len(u_interval) == 0:
                return None

            lhs = remove_nones([stl_rob(e_1, x, a) for a in u_interval])
            lhs_cums = [np.min(lhs[:k + 1]) for k in range(len(lhs))]
            rhs = remove_nones([stl_rob(e_2, x, a) for a in u_interval])

            assert len(lhs) == len(rhs), f"Ill formed 'Until' ({spec}) - lhs:{len(lhs)}, rhs:{len(rhs)}"

            running_vals = [min(rhs[a], lhs_cums[a]) for a in u_interval]

            return np.max(running_vals)
        case S(e_1, e_2, t_start, t_end):
            s_interval = range(max(t - t_end, 0), t - t_start + 1)
            if len(s_interval) == 0:
                return None

            lhs = remove_nones([stl_rob(e_1, x, a) for a in s_interval])
            lhs_cums = [np.min(lhs[:k + 1]) for k in range(len(lhs))]
            rhs = remove_nones([stl_rob(e_2, x, a) for a in s_interval])

            assert len(lhs) == len(rhs), f"Ill formed 'Since' ({spec}) - lhs:{len(lhs)}, rhs:{len(rhs)}"

            running_vals = [min(rhs[a], lhs_cums[a]) for a in s_interval]

            return np.max(running_vals)

        case _:
            raise ValueError(f"Invalid spec: : {spec} of type {type(spec)}")


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
    if isinstance(spec, LEQ0):
        return rect_pos(-spec.f(x[t]), b)
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
    if isinstance(spec, LEQ0):
        return rect_neg(-spec.f(x[t]), b)
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
    if isinstance(spec, (GEQ0, LEQ0)):
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
    if isinstance(spec, LEQ0):
        return -0.5 * spec.f(x[t])
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
