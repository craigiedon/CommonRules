from dataclasses import dataclass
from typing import Tuple, List, Any

import numpy as np


@dataclass
class IntervalConstraint:
    value: Any
    time_interval: Tuple[float, float]


@dataclass
class TaskConfig:
    # time: float # Seconds
    dt: float  # steps/second
    x_goal: float  # Metres
    y_goal: float  # Metres
    y_bounds: Tuple[float, float]
    car_length: float  # Metres
    car_width: float  # Metres
    v_goal: float  # Metres/Sec
    v_max: float  # Metres/Sec
    acc_max: float  # Metres/Sec**2
    ang_vel_max: float  # Metres/Sec
    lanes: List[float]  # Metres
    lane_targets: List[IntervalConstraint]
    collision_field_slope: float


@dataclass
class CostWeights:
    x_prog: float = 1
    y_prog: float = 1
    v_track: float = 1
    acc: float = 1
    ang_v: float = 1
    jerk: float = 1
    road_align: float = 1
    lane_align: float = 1
    collision_pot: float = 1
    faster_left: float = 1
    braking: float = 1


@dataclass
class KinMPCRes:
    xs: np.ndarray
    ys: np.ndarray
    vs: np.ndarray
    hs: np.ndarray
    accs: np.ndarray
    ang_vels: np.ndarray


@dataclass
class PointMPCResult:
    xs: np.ndarray
    ys: np.ndarray
    vs_x: np.ndarray
    vs_y: np.ndarray
    as_x: np.ndarray
    as_y: np.ndarray


def point_to_kin_res(r: PointMPCResult) -> KinMPCRes:
    return KinMPCRes(
        xs=r.xs,
        ys=r.ys,
        vs=np.sqrt(r.vs_x ** 2 + r.vs_y ** 2),
        hs=np.arctan2(r.vs_y, r.vs_x),
        accs=np.sqrt(r.as_x ** 2 + r.as_y ** 2),
        ang_vels=np.zeros(len(r.vs_x)),
    )
