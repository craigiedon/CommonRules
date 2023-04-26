import typing
from dataclasses import dataclass, field
from typing import List, Any, Tuple, Optional, Dict
import matplotlib.pyplot as plt
import numpy as np

import casadi
import torch
from commonroad.scenario.obstacle import DynamicObstacle, StaticObstacle, Obstacle
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import InitialState, KSState, CustomState
from commonroad.scenario.trajectory import State
from matplotlib import patches, transforms
from matplotlib.transforms import Affine2D
import torch.nn as nn

from immFilter import target_state_prediction, imm_kalman_filter, sticky_m_trans, AffineModel, StateFeedbackModel, \
    measure_from_state, unif_cat_prior, closest_lane_prior, IMMResult, imm_kalman_no_obs


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
    car_width: float  # Metres
    car_height: float  # Metres
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


# @dataclass
# class CarState:
#     x: float
#     y: float
#     v: float
#     heading: float

@dataclass
class CarMPCRes:
    xs: np.ndarray
    ys: np.ndarray
    vs: np.ndarray
    hs: np.ndarray
    accs: np.ndarray
    ang_vels: np.ndarray


@dataclass
class RectObstacle:
    x: float
    y: float
    w: float  # Width
    h: float  # Height
    rot: float  # Radians


@dataclass
class PolyConstraint:
    A: Any
    b: Any


class RectCustom(patches.Rectangle):
    def get_patch_transform(self):
        bbox = self.get_bbox()
        return (transforms.BboxTransformTo(bbox)
                + transforms.Affine2D().rotate_deg_around(
                    bbox.x0 + bbox.width / 2.0, bbox.y0 + bbox.height / 2.0, self.angle))


def obs_to_patch(obs: RectObstacle, color='r', alpha=1.0) -> patches.Rectangle:
    r = RectCustom((obs.x - obs.w / 2.0, obs.y - obs.h / 2.0), obs.w, obs.h, np.rad2deg(obs.rot), facecolor=color,
                   alpha=alpha)
    return r


def to_poly_constraints(x: float, y: float, w: float, h: float, rot: float) -> PolyConstraint:
    A = casadi.blockcat([
        [casadi.sin(rot), -casadi.cos(rot)],
        [-casadi.sin(rot), casadi.cos(rot)],
        [casadi.cos(rot), casadi.sin(rot)],
        [-casadi.cos(rot), -casadi.sin(rot)],
    ])

    b = casadi.vcat([h, h, w, w]) / 2.0 + A @ casadi.vcat([x, y])
    return PolyConstraint(A=A, b=b)


def to_numpy_polys(x: float, y: float, w: float, h: float, rot: float) -> PolyConstraint:
    A = np.array([
        [np.sin(rot), -np.cos(rot)],
        [-np.sin(rot), np.cos(rot)],
        [np.cos(rot), np.sin(rot)],
        [-np.cos(rot), -np.sin(rot)],
    ])

    b = np.array([h, h, w, w]) / 2.0 + A @ np.array([x, y])
    return PolyConstraint(A=A, b=b)


def car_mpc(T: int, start_state: State, task: TaskConfig,
            obstacles: List[Obstacle], obs_pred_longs: Dict[int, np.ndarray], obs_pred_lats: Dict[int, np.ndarray],
            cw: Optional[CostWeights] = None, prev_solve: CarMPCRes = None) -> CarMPCRes:
    opti = casadi.Opti()

    xs = opti.variable(T)
    ys = opti.variable(T)
    vs = opti.variable(T)
    hs = opti.variable(T)
    accs = opti.variable(T)
    ang_vels = opti.variable(T)
    lane_selectors = opti.variable(T, len(task.lanes))
    lane_params = opti.parameter(len(task.lanes))
    opti.set_value(lane_params, task.lanes)

    x_span = max(1.0, abs(task.x_goal - start_state.position[0]))
    y_span = max(1.0, abs(task.y_goal - start_state.position[1]))

    # Distance to destination cost
    x_progress_cost = casadi.sumsqr((xs - task.x_goal) / x_span)  # / (start_state.x - task.x_goal))
    y_progress_cost = casadi.sumsqr((ys - task.y_goal) / y_span)  # / (start_state.y - task.y_goal))

    # Track the reference velocity
    vel_track_cost = casadi.sumsqr(vs - task.v_goal)

    # Keep acceleration low
    acc_cost = casadi.sumsqr(accs)

    # Penalize braking
    braking_cost = casadi.sumsqr(casadi.fmin(0.0, accs))

    # Keep angular velocity low
    ang_vel_cost = casadi.sumsqr(ang_vels)

    # Minimize Jerk
    jerk_cost = casadi.sumsqr(ang_vels[1:] - ang_vels[:-1])

    # Align with road curvature
    road_align_cost = casadi.sumsqr(hs)

    # Lane alignment
    for lt in task.lane_targets:
        s_ts = int(lt.time_interval[0] / task.dt)
        e_ts = int(lt.time_interval[1] / task.dt)
        assert e_ts > s_ts
        opti.subject_to(lane_selectors[s_ts:e_ts, lt.value] == 1.0)

    lane_diffs = ((ys - casadi.repmat(lane_params, 1, T).T) ** 2)
    chosen_lanes = lane_selectors * lane_diffs
    lane_align_cost = casadi.sum2(casadi.sum1(chosen_lanes))

    # Obstacle Avoidance (Potential Fields)
    lateral_slack = 0.1
    if len(obstacles) > 0:
        d_pots = []
        for obs in obstacles:
            d_xs = obs_pred_longs[obs.obstacle_id][:, 0]
            d_ys = obs_pred_lats[obs.obstacle_id][:, 0]

            d_dist_x = ((xs - d_xs) / obs.obstacle_shape.length) ** 2
            d_dist_y = ((ys - d_ys) / (obs.obstacle_shape.width + lateral_slack)) ** 2

            d_pots.append(casadi.exp(-task.collision_field_slope * (d_dist_x + d_dist_y)))
        d_pots = casadi.vcat(d_pots)
        obs_cost = casadi.sum2(casadi.sum1(d_pots))
    else:
        obs_cost = 0.0

    # No moving faster than left traffic
    if len(obstacles) > 0:
        f_left_costs = []
        for obs in obstacles:
            d_xs = obs_pred_longs[obs.obstacle_id][:, 0]
            d_ys = obs_pred_lats[obs.obstacle_id][:, 0]

            d_x_vs = obs_pred_longs[obs.obstacle_id][:, 1]
            d_y_vs = obs_pred_lats[obs.obstacle_id][:, 1]
            d_vs = np.sqrt((d_x_vs ** 2) + (d_y_vs ** 2))

            d_dist_x = ((xs - d_xs) / obs.obstacle_shape.length) ** 2.0
            x_pot = 10 * casadi.exp(-d_dist_x)

            d_dist_y = casadi.fmax(0.0, (d_ys + obs.obstacle_shape.width / 2.0) - (ys - task.car_height / 2.0))

            vel_diffs = casadi.fmax(vs - d_vs, 0.0)
            f_left_costs.append(casadi.fmin(d_dist_y, x_pot) * vel_diffs)
        f_lefts_combined = casadi.vcat(f_left_costs)
        f_left_cost = casadi.sumsqr(f_lefts_combined)
    else:
        f_left_cost = 0.0

    if cw is None:
        cw = CostWeights()

    opti.minimize(cw.x_prog * x_progress_cost +
                  cw.y_prog * y_progress_cost +
                  cw.v_track * vel_track_cost +
                  cw.acc * acc_cost +
                  cw.ang_v * ang_vel_cost +
                  cw.jerk * jerk_cost +
                  cw.road_align * road_align_cost +
                  cw.lane_align * lane_align_cost +
                  cw.collision_pot * obs_cost +
                  cw.faster_left * f_left_cost +
                  cw.braking * braking_cost
                  )

    # Start State Constraints
    opti.subject_to(xs[0] == start_state.position[0])
    opti.subject_to(ys[0] == start_state.position[1])

    opti.subject_to(vs[0] == start_state.velocity)
    opti.subject_to(hs[0] == start_state.orientation)

    # Variable Bounds
    opti.subject_to(opti.bounded(-task.v_max, casadi.vec(vs), task.v_max))
    opti.subject_to(opti.bounded(-casadi.pi / 2.0, casadi.vec(hs), casadi.pi / 2.0))
    opti.subject_to(opti.bounded(-task.acc_max, casadi.vec(accs), task.acc_max))
    opti.subject_to(opti.bounded(-task.ang_vel_max, casadi.vec(ang_vels), task.ang_vel_max))
    opti.subject_to(opti.bounded(0, casadi.vec(lane_selectors), 1))

    # Lane Bounds
    opti.subject_to(opti.bounded(task.y_bounds[0] + task.car_height / 2.0, casadi.vec(ys),
                                 task.y_bounds[1] - task.car_height / 2.0))

    # State Evolution
    opti.subject_to(xs[1:] == xs[:-1] + casadi.cos(hs[:-1]) * vs[:-1] * task.dt)
    opti.subject_to(ys[1:] == ys[:-1] + casadi.sin(hs[:-1]) * vs[:-1] * task.dt)
    opti.subject_to(vs[1:] == vs[:-1] + accs[:-1] * task.dt)
    opti.subject_to(hs[1:] == hs[:-1] + ang_vels[:-1] * task.dt)

    # Lane Selection Simplex
    opti.subject_to(casadi.sum2(lane_selectors) == 1)

    opti.solver('ipopt', {"ipopt.print_level": 0, "print_time": 0, "ipopt.max_iter": 10000})
    # opti.solver('ipopt', {"ipopt.check_derivatives_for_naninf": "yes", "ipopt.max_iter": 10000})

    if prev_solve is not None:
        opti.set_initial(xs, prev_solve.xs)
        opti.set_initial(ys, prev_solve.ys)
        opti.set_initial(vs, prev_solve.vs)
        opti.set_initial(hs, prev_solve.hs)
        opti.set_initial(accs, prev_solve.accs)
        opti.set_initial(ang_vels, prev_solve.ang_vels)

    try:
        sol = opti.solve()
    except RuntimeError as e:
        print("Uh oh! An error...")
        print("xs:", opti.debug.value(xs))
        print("ys:", opti.debug.value(ys))
        print("vs:", opti.debug.value(vs))
        print("hs:", opti.debug.value(hs))
        print("accs:", opti.debug.value(accs))
        print("ang_vels:", opti.debug.value(ang_vels))
        raise e

    res = CarMPCRes(sol.value(xs), sol.value(ys), sol.value(vs), sol.value(hs), sol.value(accs), sol.value(ang_vels))

    return res


def plot_results(res: CarMPCRes, task: TaskConfig, static_obstacles: List[RectObstacle]):
    n_subplots = 4
    T = int(task.time / task.dt)

    fig, axs = plt.subplots(1, n_subplots)
    axs[0].plot(np.arange(T) * task.dt, res.xs)
    axs[0].set_xlabel("t")
    axs[0].set_ylabel("X")

    axs[1].plot(np.arange(T) * task.dt, res.ys)
    axs[1].set_ylim(-5, 5)
    axs[1].set_xlabel("t")
    axs[1].set_ylabel("Y")

    axs[2].plot(np.arange(T) * task.dt, res.vs)
    axs[3].plot(np.arange(T) * task.dt, res.hs)
    axs[3].set_ylim(-np.pi, np.pi)
    axs[3].set_xlabel("t")
    axs[3].set_ylabel("$\\theta$")

    fig_2, axs_2 = plt.subplots(figsize=(15, 2))
    for obs in static_obstacles:
        axs_2.add_patch(obs_to_patch(obs))
    for x, y, h in zip(res.xs, res.ys, res.hs):
        axs_2.add_patch(obs_to_patch(RectObstacle(x, y, task.car_width, task.car_height, h), color='b', alpha=0.2))
    axs_2.scatter(res.xs, res.ys, marker='x')
    axs_2.set_xlabel("X")
    axs_2.set_ylabel("Y")

    plt.show()


def receding_horizon(total_time: float, horizon_length: float, start_state: InitialState, scenario: Scenario,
                     task_config: TaskConfig, cws: CostWeights) -> List[State]:
    T = int(np.round(total_time / task_config.dt))
    res = None
    dn_state_list = []
    current_state = start_state

    for i in range(1, T):
        obs_pred_longs, obs_pred_lats = obs_long_lats(scenario.obstacles, i,
                                                      i + int(round(horizon_length / task_config.dt)))

        flat_cost = car_mpc(int(round(horizon_length / task_config.dt)), current_state, task_config,
                            scenario.obstacles, obs_pred_longs, obs_pred_lats, None, None)
        res = car_mpc(int(round(horizon_length / task_config.dt)), current_state, task_config,
                      scenario.obstacles, obs_pred_longs, obs_pred_lats, cws, flat_cost)
        current_state = CustomState(position=np.array([res.xs[1], res.ys[1]]), velocity=res.vs[1],
                                    orientation=res.hs[1],
                                    acceleration=res.accs[1], time_step=i)
        dn_state_list.append(current_state)

    return dn_state_list


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


def initial_long(o: Obstacle, obs_longs: Dict[int, np.ndarray], long_models: List[AffineModel]) -> IMMResult:
    id = o.obstacle_id

    m_prior = unif_cat_prior(len(long_models))
    m_mus = np.tile(obs_longs[id][0], (len(long_models), 1))
    m_covs = np.tile(np.identity(3), (len(long_models), 1, 1))
    init_res = IMMResult(m_prior, obs_longs[id][0], None, m_mus, m_covs)

    return init_res


def initial_lat(o: Obstacle, obs_lats: Dict[int, np.ndarray], lat_models: List[AffineModel]) -> IMMResult:
    id = o.obstacle_id

    m_prior = closest_lane_prior(obs_lats[id][0, 0], lat_models, 0.9)
    m_mus = np.tile(obs_lats[id][0], (len(lat_models), 1))
    m_covs = np.tile(np.identity(len(obs_lats[id][0])), (len(lat_models), 1, 1))
    init_res = IMMResult(m_prior, obs_lats[id][0], None, m_mus, m_covs)

    return init_res


def identity_observation(o: Obstacle,
                         prev_long_est: np.ndarray,
                         prev_lat_est: np.ndarray,
                         tru_long_state: np.ndarray,
                         tru_lat_state: np.ndarray,
                         dt: float) -> Tuple[np.ndarray, np.ndarray]:
    assert len(tru_long_state) == 3
    assert len(tru_lat_state) == 2

    # Longitudinal observation: Position / Velocity
    observed_long_pos = tru_long_state[0]
    observed_long_vel = (tru_long_state[0] - prev_long_est[0]) / dt
    observed_long_state = np.array([observed_long_pos, observed_long_vel])

    # Latitudinal observation: Position
    observed_lat_state = tru_lat_state[0]
    return observed_long_state, observed_lat_state


def bernoulli_drop_observation(o: Obstacle,
                               prev_long_est: np.ndarray,
                               prev_lat_est: np.ndarray,
                               tru_long_state: np.ndarray,
                               tru_lat_state: np.ndarray,
                               dt: float,
                               detection_probability: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    r = np.random.rand()

    if r > detection_probability:
        return None, None

    # Longitudinal observation: Position / Velocity
    observed_long_pos = tru_long_state[0]
    observed_long_vel = (observed_long_pos - prev_long_est[0]) / dt
    observed_long_state = np.array([observed_long_pos, observed_long_vel])

    # Latitudinal observation: Position
    observed_lat_state = tru_lat_state[0]
    return observed_long_state, observed_lat_state


def toy_drops_noise_observation(o: Obstacle,
                                prev_long_est: np.ndarray,
                                prev_lat_est: np.ndarray,
                                tru_long_state: np.ndarray,
                                tru_lat_state: np.ndarray,
                                dt: float,
                                detection_probability: float,
                                noise_var: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    r = np.random.rand()

    if r > detection_probability:
        return None, None

    # Longitudinal observation: Position / Velocity
    observed_long_pos = tru_long_state[0] + np.random.normal(0.0, noise_var)
    # observed_long_vel = (observed_long_pos - prev_long_est[0]) / time_step
    observed_long_vel = tru_long_state[1]
    if np.abs(observed_long_vel) > 100:
        print("Obs long vel:", observed_long_vel)
    # observed_long_vel = tru_long_state[1]
    observed_long_state = np.array([observed_long_pos, observed_long_vel])

    # Latitudinal observation: Position
    observed_lat_state = tru_lat_state[0] + np.random.normal(0.0, noise_var)
    return observed_long_state, observed_lat_state


def pem_observation(o: Obstacle,
                    t: int,
                    prev_long_est: np.ndarray,
                    prev_lat_est: np.ndarray,
                    tru_long_state: np.ndarray,
                    tru_lat_state: np.ndarray,
                    det_pem: nn.Module,
                    reg_pem: nn.Module,
                    norm_mus: torch.Tensor,
                    norm_stds: torch.Tensor,
                    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    r = np.random.rand()

    det_pem.eval()
    reg_pem.eval()

    s_xs = tru_long_state[0]
    s_ys = tru_lat_state[0]
    s_r = o.state_at_time(t).orientation

    s_dims = [o.obstacle_shape.length, o.obstacle_shape.width, 1.7]
    s_viz = (4 - 1) / 3.0

    state_tensor = torch.Tensor([s_xs, s_ys, torch.sin(s_r), torch.cos(s_r), s_dims, s_viz])
    state_tensor = (state_tensor - norm_mus) / norm_stds

    with torch.no_grad():
        detection_probability = det_pem(state_tensor)

    if r > detection_probability:
        return None, None

    with torch.no_grad():
        noise_loc, noise_var = reg_pem(state_tensor, full_cov=True)

    # Longitudinal observation: Position / Velocity
    observed_long_pos = tru_long_state[0] + np.random.normal(long_noise_loc, long_noise_var)
    observed_long_vel = tru_long_state[1]
    observed_long_state = np.array([observed_long_pos, observed_long_vel])

    # Latitudinal observation: Position
    observed_lat_state = tru_lat_state[0] + np.random.normal(lat_noise_loc, lat_noise_var)
    return observed_long_state, observed_lat_state


@dataclass
class RecedingHorizonStats:
    true_longs: List[np.ndarray] = field(default_factory=list)
    true_lats: List[np.ndarray] = field(default_factory=list)
    observed_longs: List[Optional[np.ndarray]] = field(default_factory=list)  # Some time steps may get no observation
    observed_lats: List[Optional[np.ndarray]] = field(default_factory=list)  # Some time steps may get no observation
    est_longs: List[np.ndarray] = field(default_factory=list)
    est_lats: List[np.ndarray] = field(default_factory=list)
    prediction_traj_longs: List[np.ndarray] = field(default_factory=list)
    prediction_traj_lats: List[np.ndarray] = field(default_factory=list)


def kalman_receding_horizon(total_time: float, horizon_length: float, start_state: InitialState, scenario: Scenario,
                            task_config: TaskConfig, long_models: List[AffineModel],
                            lat_models: List[StateFeedbackModel],
                            cws: CostWeights) -> Tuple[List[CustomState], Dict[int, RecedingHorizonStats]]:
    T = int(np.round(total_time / task_config.dt))
    res = None
    dn_state_list = []
    current_state = start_state

    obstacle_longs, obstacle_lats = obs_long_lats(scenario.obstacles, 0, T)

    # Setup initial state estimation
    est_long_res = {o.obstacle_id: initial_long(o, obstacle_longs, long_models) for o in scenario.obstacles}
    est_lat_res = {o.obstacle_id: initial_lat(o, obstacle_lats, lat_models) for o in scenario.obstacles}

    prediction_steps = int(round(horizon_length / task_config.dt)) - 1

    obs_traj_data: Dict[int, RecedingHorizonStats] = {o.obstacle_id:
        RecedingHorizonStats(
            true_longs=[est_long_res[o.obstacle_id].fused_mu],
            true_lats=[est_lat_res[o.obstacle_id].fused_mu],
            observed_longs=[None],
            observed_lats=[None],
            est_longs=[est_long_res[o.obstacle_id].fused_mu],
            est_lats=[est_lat_res[o.obstacle_id].fused_mu],
        ) for o in scenario.obstacles}

    for i in range(0, T - 1):
        print(i)
        obs_pred_longs = {}
        obs_pred_lats = {}
        for obs in scenario.obstacles:
            # At i = 0, we don't take a measurement, assume we start correct:
            if i > 0:
                # Get the current state of each obstacle
                true_long, true_lat = obstacle_longs[obs.obstacle_id][i], obstacle_lats[obs.obstacle_id][i]

                # Get its measurement
                # z_long = measure_from_state(true_long, long_models[0])
                # z_lat = measure_from_state(true_lat, lat_models[0])

                prev_long_res = est_long_res[obs.obstacle_id]
                prev_lat_res = est_lat_res[obs.obstacle_id]

                # z_long, z_lat = toy_drops_noise_observation(obs, prev_long_res.fused_mu, prev_lat_res.fused_mu,
                #                                             true_long,
                #                                             true_lat, task_config.dt, 0.5, 0.5)

                det_pem = torch.load()
                reg_pem = torch.load()
                z_long, z_lat = pem_observation(obs, i, prev_long_res.fused_mu, prev_lat_res.fused_mu,
                                                true_long, true_lat,
                                                det_pem, reg_pem, norm_mus, norm_stds)

                # One-step IMM Kalman filter
                if z_long is not None:
                    est_long_res[obs.obstacle_id] = imm_kalman_filter(long_models,
                                                                      sticky_m_trans(len(long_models), 0.95),
                                                                      prev_long_res.model_ps,
                                                                      prev_long_res.model_mus,
                                                                      prev_long_res.model_covs, z_long)
                else:
                    est_long_res[obs.obstacle_id] = imm_kalman_no_obs(long_models,
                                                                      sticky_m_trans(len(long_models), 0.95),
                                                                      prev_long_res.model_ps,
                                                                      prev_long_res.model_mus,
                                                                      prev_long_res.model_covs)
                if z_lat is not None:
                    est_lat_res[obs.obstacle_id] = imm_kalman_filter(lat_models, sticky_m_trans(len(lat_models), 0.95),
                                                                     prev_lat_res.model_ps,
                                                                     prev_lat_res.model_mus,
                                                                     prev_lat_res.model_covs, z_lat)
                else:
                    est_lat_res[obs.obstacle_id] = imm_kalman_no_obs(lat_models, sticky_m_trans(len(lat_models), 0.95),
                                                                     prev_lat_res.model_ps,
                                                                     prev_lat_res.model_mus,
                                                                     prev_lat_res.model_covs)

                # Update stats
                obs_traj_data[obs.obstacle_id].true_lats.append(true_lat)
                obs_traj_data[obs.obstacle_id].true_longs.append(true_long)
                obs_traj_data[obs.obstacle_id].observed_longs.append(z_long)
                obs_traj_data[obs.obstacle_id].observed_lats.append(z_lat)
                obs_traj_data[obs.obstacle_id].est_longs.append(est_long_res[obs.obstacle_id].fused_mu)
                obs_traj_data[obs.obstacle_id].est_lats.append(est_lat_res[obs.obstacle_id].fused_mu)

            # Predict forward using top mode
            pred_longs = target_state_prediction(est_long_res[obs.obstacle_id].fused_mu, long_models,
                                                 est_long_res[obs.obstacle_id].model_ps, prediction_steps)
            pred_lats = target_state_prediction(est_lat_res[obs.obstacle_id].fused_mu, lat_models,
                                                est_lat_res[obs.obstacle_id].model_ps, prediction_steps)

            obs_pred_longs[obs.obstacle_id] = np.concatenate(([est_long_res[obs.obstacle_id].fused_mu], pred_longs),
                                                             axis=0)
            obs_pred_lats[obs.obstacle_id] = np.concatenate(([est_lat_res[obs.obstacle_id].fused_mu], pred_lats),
                                                            axis=0)

            obs_traj_data[obs.obstacle_id].prediction_traj_longs.append(obs_pred_longs[obs.obstacle_id])
            obs_traj_data[obs.obstacle_id].prediction_traj_lats.append(obs_pred_lats[obs.obstacle_id])

        flat_cost = car_mpc(prediction_steps + 1, current_state, task_config,
                            scenario.obstacles, obs_pred_longs, obs_pred_lats, None, None)
        res = car_mpc(prediction_steps + 1, current_state, task_config,
                      scenario.obstacles, obs_pred_longs, obs_pred_lats, cws, flat_cost)
        current_state = CustomState(position=np.array([res.xs[1], res.ys[1]]), velocity=res.vs[1],
                                    orientation=res.hs[1],
                                    acceleration=res.accs[1], time_step=i + 1)
        dn_state_list.append(current_state)

    for _, td in obs_traj_data.items():
        assert len(td.prediction_traj_lats) == T - 1
        assert len(td.prediction_traj_longs) == T - 1
        assert len(td.true_lats) == T - 1
        assert len(td.true_longs) == T - 1
        assert len(td.est_lats) == T - 1
        assert len(td.est_longs) == T - 1
        assert len(td.observed_lats) == T - 1
        assert len(td.observed_longs) == T - 1

    return dn_state_list, obs_traj_data


def run():
    pass
    # Ford Escort Config. See Commonroad Vehicle Model Documentation
    # task_config = TaskConfig(time=5,
    #                          dt=0.1,
    #                          x_goal=700.0,
    #                          y_goal=5.0,
    #                          car_width=4.298,
    #                          car_height=1.674,
    #                          v_goal=31.29, # == 70mph
    #                          v_max=45.8,
    #                          acc_max=11.5,
    #                          ang_vel_max=0.4,
    #                          lanes=[-5.0, 0.0, 5.0])
    # obstacles = [
    #     RectObstacle(40.0, 4.5, 4.3, 1.674, 0.0),
    #     RectObstacle(5.0, 4.5, 4.3, 1.674, 0.0)
    # ]
    #
    # res = car_mpc(CarState(0.0, 0.0, 0.0, 0.0), task_config, obstacles)
    #
    # plot_results(res, task_config, obstacles)


if __name__ == "__main__":
    run()
