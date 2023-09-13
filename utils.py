from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import numpy as np
from commonroad.geometry.shape import Rectangle
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import Obstacle, DynamicObstacle, ObstacleType
from commonroad.scenario.state import CustomState
from commonroad.scenario.trajectory import Trajectory

from immFilter import IMMResult


def rot_mat(r: float) -> np.ndarray:
    return np.array([
        [np.cos(r), -np.sin(r)],
        [np.sin(r), np.cos(r)]]
    )


def angle_diff(a1: float, a2: float) -> float:
    return np.arctan2(np.sin(a1 - a2), np.cos(a1 - a2))


def obs_long_lats(obstacles: List[Obstacle], start_step: int, end_step: int) -> Tuple[
    Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    # Retrofitting conversion for kalman format
    obs_pred_longs = {}
    obs_pred_lats = {}

    for obs in obstacles:
        ob_long, ob_lat = ll_from_CR_state(obs, start_step, end_step)
        obs_pred_longs[obs.obstacle_id] = ob_long
        obs_pred_lats[obs.obstacle_id] = ob_lat

    return obs_pred_longs, obs_pred_lats


def ll_from_CR_state(obs: Obstacle, start_step: int, end_step: int):
    # Return long/lat state containing position, velocity and (for long) accelerations
    ob_states = [obs.state_at_time(i) for i in range(start_step, end_step)]
    ob_xs = np.array([s.position[0] for s in ob_states])
    ob_ys = np.array([s.position[1] for s in ob_states])
    ob_vs = np.array([s.velocity for s in ob_states])
    ob_accs = np.array([s.acceleration for s in ob_states])
    ob_r = np.array([s.orientation for s in ob_states])

    ob_x_vs = ob_vs * np.cos(ob_r)
    ob_y_vs = ob_vs * np.sin(ob_r)

    ob_x_accs = ob_accs * np.cos(ob_r)

    ob_long_state = np.stack((ob_xs, ob_x_vs, ob_x_accs), 1)
    ob_lat_state = np.stack((ob_ys, ob_y_vs), 1)

    assert ob_long_state.shape == (end_step - start_step, 3)
    assert ob_lat_state.shape == (end_step - start_step, 2)

    return ob_long_state, ob_lat_state


def mpc_result_to_dyn_obj(o_id, dn_state_list: List[CustomState], car_width: float,
                          car_length: float):
    dyn_obs_shape = Rectangle(width=car_width, length=car_length)
    dyn_obs_traj = Trajectory(1, dn_state_list[1:])
    dyn_obs_pred = TrajectoryPrediction(dyn_obs_traj, dyn_obs_shape)
    return DynamicObstacle(o_id,
                           ObstacleType.CAR,
                           dyn_obs_shape,
                           dn_state_list[0],
                           dyn_obs_pred)


@dataclass
class RecHorStat:
    true_long: Dict[int, np.ndarray]
    true_lat: Dict[int, np.ndarray]
    observed_long: Dict[int, np.ndarray]
    observed_lat: Dict[int, np.ndarray]
    est_long: Dict[int, IMMResult]
    est_lat: Dict[int, IMMResult]
    prediction_traj_long: Dict[int, np.ndarray]
    prediction_traj_lat: Dict[int, np.ndarray]

@dataclass
class RecedingHorizonStats:
    true_longs: List[np.ndarray] = field(default_factory=list)
    true_lats: List[np.ndarray] = field(default_factory=list)
    observed_longs: List[Optional[np.ndarray]] = field(default_factory=list)  # Some time steps may get no observation
    observed_lats: List[Optional[np.ndarray]] = field(default_factory=list)  # Some time steps may get no observation
    est_longs: List[IMMResult] = field(default_factory=list)
    est_lats: List[IMMResult] = field(default_factory=list)
    prediction_traj_longs: List[np.ndarray] = field(default_factory=list)
    prediction_traj_lats: List[np.ndarray] = field(default_factory=list)

    def append_stat(self, true_long: np.ndarray, true_lat: np.ndarray,
                    obs_long: np.ndarray, obs_lat: np.ndarray,
                    est_long: IMMResult, est_lat: IMMResult,
                    pred_traj_long, pred_traj_lat):
        self.true_longs.append(true_long)
        self.true_lats.append(true_lat)
        self.observed_longs.append(obs_long)
        self.observed_lats.append(obs_lat)
        self.est_longs.append(est_long)
        self.est_lats.append(est_lat)
        self.prediction_traj_longs.append(pred_traj_long)
        self.prediction_traj_lats.append(pred_traj_lat)


