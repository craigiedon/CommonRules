import time
import typing
from dataclasses import dataclass

from TaskConfig import TaskConfig, CostWeights, KinMPCRes, point_to_kin_res
from cvxPractice.carExample import cvx_mpc, create_cvx_mpc
from typing import List, Any, Tuple, Optional, Dict
import matplotlib.pyplot as plt
import numpy as np

import casadi
import pyro
import torch
from commonroad.scenario.obstacle import Obstacle
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import InitialState, CustomState
from commonroad.scenario.trajectory import State
from matplotlib import patches, transforms
import torch.nn as nn

from Raycasting import ray_intersect, radial_ray_batch, occlusions_from_ray_hits
from immFilter import t_state_pred, sticky_m_trans, AffineModel, StateFeedbackModel, \
    unif_cat_prior, closest_lane_prior, IMMResult, imm_kalman_optional
from utils import angle_diff, rot_mat, obs_long_lats, RecedingHorizonStats, RecHorStat


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
            cw: Optional[CostWeights] = None, prev_solve: KinMPCRes = None) -> KinMPCRes:
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

    # x_span = max(1.0, abs(task.x_goal - start_state.position[0]))
    # y_span = max(1.0, abs(task.y_goal - start_state.position[1]))

    # Distance to destination cost
    # x_progress_cost = casadi.sumsqr((xs - task.x_goal) / x_span)  # / (start_state.x - task.x_goal))
    y_progress_cost = casadi.sumsqr((ys - task.y_goal))  # / (start_state.y - task.y_goal))

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

    # lane_diffs = ((ys - casadi.repmat(lane_params, 1, T).T) ** 2)
    lane_diffs = (ys - casadi.repmat(lane_params, 1, T).T)
    chosen_lanes = lane_selectors * lane_diffs
    # lane_align_cost = casadi.sum2(casadi.sum1(chosen_lanes))
    lane_align_cost = casadi.sumsqr(chosen_lanes)

    vxs = vs * casadi.cos(hs)
    axs = accs * casadi.cos(hs)

    vys = vs * casadi.sin(hs)
    ays = vs * casadi.sin(hs)

    # Obstacle Avoidance (Potential Fields)
    if len(obstacles) > 0:
        d_pots = []
        for obs in obstacles:
            d_xs = obs_pred_longs[obs.obstacle_id][:, 0]
            d_ys = obs_pred_lats[obs.obstacle_id][:, 0]

            # y_left_slack = 50
            # Note: We have a car length of slack as our distance boundary
            d_diffs_x = (0.5 * obs.obstacle_shape.length + 1.5 * task.car_length + (vxs * task.dt) + (
                    axs * task.dt ** 2) + 3) - casadi.fabs(xs - d_xs)

            d_diffs_y = (0.5 * obs.obstacle_shape.width + task.car_width + 1.5 + (vys * task.dt) + (
                    ays * task.dt ** 2)) - casadi.fabs(ys - d_ys)

            sharp = 10
            d_dist_x = casadi.log(1 + casadi.exp(sharp * d_diffs_x)) / sharp
            d_dist_y = casadi.log(1 + casadi.exp(sharp * d_diffs_y)) / sharp

            avoidance_pot = casadi.fmin(d_dist_x, d_dist_y)

            d_pots.append(avoidance_pot)

        d_pots = casadi.vcat(d_pots)
        # obs_cost = casadi.sum2(casadi.sum1(d_pots))
        obs_cost = casadi.sumsqr(d_pots)
    else:
        obs_cost = 0.0

    # No moving faster than left traffic
    # if len(obstacles) > 0:
    #     f_left_costs = []
    #     for obs in obstacles:
    #         d_xs = obs_pred_longs[obs.obstacle_id][:, 0]
    #         d_ys = obs_pred_lats[obs.obstacle_id][:, 0]
    #
    #         d_x_vs = obs_pred_longs[obs.obstacle_id][:, 1]
    #         # d_y_vs = obs_pred_lats[obs.obstacle_id][:, 1]
    #         # d_vs = np.sqrt((d_x_vs ** 2) + (d_y_vs ** 2))
    #
    #         d_diffs_x = (0.5 * obs.obstacle_shape.length + task.car_length) - casadi.fabs(xs - d_xs)
    #         # d_diffs_y = d_ys - ys
    #         # d_diffs_v = vs - d_x_vs
    #         fl_slack = 100
    #         d_diffs_y = (0.5 * obs.obstacle_shape.width + task.car_width + fl_slack) - casadi.fabs(ys - (d_ys - fl_slack))
    #
    #         sharp = 10
    #         xp = (casadi.log(1 + casadi.exp(sharp * d_diffs_x)) / sharp)
    #         yp = (casadi.log(1 + casadi.exp(sharp * d_diffs_y)) / sharp)
    #         # vp = casadi.log(1 + casadi.exp(sharp * d_diffs_v)) / sharp
    #
    #         # f_left_costs.append(casadi.fmin(xp, yp))
    #         # f_left_costs.append(casadi.fmin(casadi.fmin(vp, yp), xp))
    #         lge_meth = -casadi.log(casadi.exp(-xp) + casadi.exp(-yp))
    #         f_left_costs.append(casadi.fmin(xp, yp))
    #         # f_left_costs.append(lge_meth)
    #         # f_left_costs.append(casadi.fmin(vp, xp))
    #
    #     f_lefts_combined = casadi.vcat(f_left_costs)
    #     # f_left_cost = casadi.sumsqr(f_lefts_combined)
    #     f_left_cost = casadi.sum2(casadi.sum1(f_lefts_combined))
    # else:
    f_left_cost = 0.0

    if cw is None:
        cw = CostWeights()

    opti.minimize(  # cw.x_prog * x_progress_cost +
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
    opti.subject_to(opti.bounded(task.y_bounds[0] + task.car_width / 2.0 + 0.4, casadi.vec(ys),
                                 task.y_bounds[1] - task.car_width / 2.0 - 0.4))

    # State Evolution
    opti.subject_to(xs[1:] == xs[:-1] + casadi.cos(hs[:-1]) * vs[:-1] * task.dt)
    opti.subject_to(ys[1:] == ys[:-1] + casadi.sin(hs[:-1]) * vs[:-1] * task.dt)
    opti.subject_to(vs[1:] == vs[:-1] + accs[:-1] * task.dt)
    opti.subject_to(hs[1:] == hs[:-1] + ang_vels[:-1] * task.dt)

    # Lane Selection Simplex
    opti.subject_to(casadi.sum2(lane_selectors) == 1)

    opti.solver('ipopt', {"ipopt.print_level": 0, "print_time": 0, "ipopt.max_iter": 10000})
    # opti.solver("gurobi")
    # opti.solver('gurobi')
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

    res = KinMPCRes(sol.value(xs), sol.value(ys), sol.value(vs), sol.value(hs), sol.value(accs), sol.value(ang_vels))

    return res


def plot_results(res: KinMPCRes, task: TaskConfig, static_obstacles: List[RectObstacle]):
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
        axs_2.add_patch(obs_to_patch(RectObstacle(x, y, task.car_length, task.car_width, h), color='b', alpha=0.2))
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
                    ego_state: CustomState,
                    t: int,
                    prev_long_est: np.ndarray,
                    prev_lat_est: np.ndarray,
                    tru_long_state: np.ndarray,
                    tru_lat_state: np.ndarray,
                    det_pem: nn.Module,
                    reg_pem: nn.Module,
                    norm_mus: torch.Tensor,
                    norm_stds: torch.Tensor,
                    o_viz: float,
                    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    r = np.random.rand()

    # Get the ego vehicle's x,y
    rot_frame = rot_mat(-ego_state.orientation)
    ego_pos = ego_state.position

    s_long, s_lat = rot_frame @ [tru_long_state[0] - ego_pos[0], tru_lat_state[0] - ego_pos[1]]

    s_r = torch.tensor(angle_diff(o.state_at_time(t).orientation, ego_state.orientation), dtype=torch.float)

    s_dims = [o.obstacle_shape.length, o.obstacle_shape.width, 1.7]
    # s_viz = (4 - 1) / 3.0

    state_tensor = torch.Tensor([s_long, s_lat, torch.sin(s_r), torch.cos(s_r), *s_dims, o_viz]).unsqueeze(0)
    state_tensor = (state_tensor - norm_mus) / norm_stds
    state_tensor = state_tensor.cuda()

    p_store = pyro.get_param_store()

    with torch.no_grad():
        pyro.get_param_store().clear()
        det_p = torch.sigmoid(det_pem(state_tensor)[0])

    p_store.clear()

    if r > det_p:
        return None, None

    pem_start = time.time()
    with torch.no_grad():
        pyro.get_param_store().clear()
        noise_loc, noise_var = reg_pem(state_tensor)
        noise_loc = noise_loc.cpu().detach()
        noise_var = noise_var.cpu().detach()

    # Longitudinal observation: Position / Velocity
    observed_long_pos = tru_long_state[0] + np.random.normal(noise_loc[0, 0], noise_var[0, 0])
    observed_long_vel = tru_long_state[1]
    observed_long_state = np.array([observed_long_pos, observed_long_vel])

    # Latitudinal observation: Position
    observed_lat_state = tru_lat_state[0] + np.random.normal(noise_loc[1, 0], noise_var[1, 0])

    return observed_long_state, observed_lat_state


def pem_observation_batch(obs: List[Obstacle],
                          ego_state: CustomState,
                          t: int,
                          tru_long_states: np.ndarray,
                          tru_lat_states: np.ndarray,
                          det_pem: nn.Module,
                          reg_pem: nn.Module,
                          norm_mus: torch.Tensor,
                          norm_stds: torch.Tensor,
                          o_viz: np.ndarray,
                          ) -> Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
    # Get the ego vehicle's x,y
    rot_frame = rot_mat(-ego_state.orientation)
    ego_pos = ego_state.position

    s_longs, s_lats = rot_frame @ np.array([tru_long_states[:, 0] - ego_pos[0], tru_lat_states[:, 0] - ego_pos[1]])

    obs_rots = np.array([o.state_at_time(t).orientation for o in obs])
    s_rs = torch.tensor(angle_diff(obs_rots, ego_state.orientation), dtype=torch.float)

    s_dims = torch.tensor([[o.obstacle_shape.length, o.obstacle_shape.width, 1.7] for o in obs], dtype=torch.float)

    state_tensor = torch.column_stack([torch.tensor(s_longs, dtype=torch.float),
                                       torch.tensor(s_lats, dtype=torch.float),
                                       torch.sin(s_rs),
                                       torch.cos(s_rs),
                                       s_dims,
                                       torch.tensor(o_viz, dtype=torch.float)])
    state_tensor = (state_tensor - norm_mus) / norm_stds
    state_tensor = state_tensor.cuda()

    p_store = pyro.get_param_store()

    with torch.no_grad():
        pyro.get_param_store().clear()
        det_ps = torch.sigmoid(det_pem(state_tensor)[0])
        det_ps = det_ps.cpu().detach().numpy()

    p_store.clear()

    # if r > det_p:
    #     return None, None

    with torch.no_grad():
        pyro.get_param_store().clear()
        noise_locs, noise_vars = reg_pem(state_tensor)
        noise_loc = noise_locs.cpu().detach()
        noise_var = noise_vars.cpu().detach()

    # Longitudinal observation: Position / Velocity
    observed_long_pos = tru_long_states[:, 0] + np.random.normal(noise_loc[0, :], np.sqrt(noise_var[0, :]))
    observed_long_vel = tru_long_states[:, 1]
    observed_long_state = np.array([observed_long_pos, observed_long_vel]).T

    # Latitudinal observation: Position
    observed_lat_state = tru_lat_states[:, 0] + np.random.normal(noise_loc[1, :], noise_var[1, :])

    rands = np.random.rand(len(obs))
    observed_long_state = [s if r < det_p else None for s, r, det_p in zip(observed_long_state, rands, det_ps)]
    observed_lat_state = [s if r < det_p else None for s, r, det_p in zip(observed_lat_state, rands, det_ps)]

    return observed_long_state, observed_lat_state


def get_corners(pos: np.ndarray, rot: float, length: float, width: float) -> np.ndarray:
    rm = rot_mat(rot)
    raw_corners = np.array([
        [-length / 2.0, -width / 2.0],
        [-length / 2.0, width / 2.0],
        [length / 2.0, width / 2.0],
        [length / 2.0, -width / 2.0]
    ])

    rotated_corners = (rm @ raw_corners.T).T
    corners = rotated_corners + pos
    return corners


def car_visibilities_raycast(n_rays: int, current_state: CustomState, i: int, obstacles: List[Obstacle]) -> np.ndarray:
    rays = radial_ray_batch(current_state.position, n_rays)
    obs_corners = {o.obstacle_id: get_corners(o.state_at_time(i).position, o.state_at_time(i).orientation,
                                              o.obstacle_shape.length, o.obstacle_shape.width) for o in obstacles}
    obs_ray_results = {o_id: ray_intersect(rays.origins, rays.dirs, oc) for o_id, oc in obs_corners.items()}

    obs_viz = {}
    visibilities = []
    for target_id, target_res in obs_ray_results.items():
        occs_res = [o_res for occ_id, o_res in obs_ray_results.items() if occ_id != target_id]
        viz, _, _ = occlusions_from_ray_hits(target_res, occs_res)
        obs_viz[target_id] = viz
        visibilities.append(viz)

    return np.array(visibilities)


def initialize_rec_hor_stat(obs: List[Obstacle], long_models: List[AffineModel], lat_models: List[AffineModel],
                            start_step: int, end_step: int, prediction_steps: int) -> RecHorStat:
    obstacle_longs, obstacle_lats = obs_long_lats(obs, start_step, end_step)
    est_long_res = {o.obstacle_id: initial_long(o, obstacle_longs, long_models) for o in obs}
    est_lat_res = {o.obstacle_id: initial_lat(o, obstacle_lats, lat_models) for o in obs}

    obs_traj_data: RecHorStat = RecHorStat(
        true_long={oid: e.fused_mu for oid, e in est_long_res.items()},
        true_lat={oid: e.fused_mu for oid, e in est_lat_res.items()},
        observed_long=None,
        observed_lat=None,
        est_long=est_long_res,
        est_lat=est_lat_res,
        prediction_traj_long={oid: t_state_pred(e.fused_mu, long_models, e.model_ps, prediction_steps) for oid, e in
                              est_long_res.items()},
        prediction_traj_lat={oid: t_state_pred(e.fused_mu, lat_models, e.model_ps, prediction_steps) for oid, e in
                             est_lat_res.items()})

    return obs_traj_data


def kalman_receding_horizon(start_step: int, sim_steps: int, prediction_steps: int, start_state: InitialState,
                            start_ests: Optional[RecHorStat], scenario: Scenario,
                            task_config: TaskConfig, long_models: List[AffineModel],
                            lat_models: List[StateFeedbackModel], observation_func: typing.Callable,
                            cws: CostWeights) -> Tuple[List[CustomState], List[Dict[int, State]], List[RecHorStat]]:
    assert start_step >= 0
    assert sim_steps > 0
    # prediction_steps = int(round(horizon_length / task_config.dt)) - 1
    current_state = start_state

    obstacle_longs, obstacle_lats = obs_long_lats(scenario.obstacles, start_step, start_step + sim_steps)

    # Setup initial state estimation
    dn_state_list = [current_state]
    traj_ob_model_data: List[RecHorStat] = []

    if start_ests is None:
        traj_ob_model_data.append(
            initialize_rec_hor_stat(scenario.obstacles, long_models, lat_models, start_step, start_step + sim_steps, prediction_steps))
    else:
        traj_ob_model_data.append(start_ests)

    cvx_prob_config = create_cvx_mpc(prediction_steps + 1, task_config, scenario.obstacles)

    tru_long_states = np.array(list(obstacle_longs.values()))
    tru_lat_states = np.array(list(obstacle_lats.values()))

    for i in range(1, sim_steps):
        t = start_step + i
        cvx_warm = cvx_mpc(cvx_prob_config, current_state, scenario.obstacles, traj_ob_model_data[-1].prediction_traj_long,
                           traj_ob_model_data[-1].prediction_traj_lat)
        if cvx_warm is not None:
            warm_start = point_to_kin_res(cvx_warm)
        else:
            warm_start = None
        res = car_mpc(prediction_steps + 1, current_state, task_config,
                      scenario.obstacles, traj_ob_model_data[-1].prediction_traj_long, traj_ob_model_data[-1].prediction_traj_lat, cws,
                      warm_start)
        current_state = CustomState(position=np.array([res.xs[1], res.ys[1]]), velocity=res.vs[1],
                                    orientation=res.hs[1],
                                    acceleration=res.accs[1], time_step=t)
        dn_state_list.append(current_state)

        # Calculating Visibilities from Raycasts
        visibilities = car_visibilities_raycast(100, current_state, t, scenario.obstacles)

        z_longs, z_lats = observation_func(scenario.obstacles, current_state, t, tru_long_states[:, i],
                                           tru_lat_states[:, i], visibilities)

        z_longs = {o.obstacle_id: z for o, z in zip(scenario.obstacles, z_longs)}
        z_lats = {o.obstacle_id: z for o, z in zip(scenario.obstacles, z_lats)}

        est_long_res = {oid: imm_kalman_optional(long_models, sticky_m_trans(len(long_models), 0.95), e, z_longs[oid])
                        for oid, e in traj_ob_model_data[-1].est_long.items()}
        est_lat_res = {oid: imm_kalman_optional(lat_models, sticky_m_trans(len(lat_models), 0.95), e, z_lats[oid]) for
                       oid, e in traj_ob_model_data[-1].est_lat.items()}

        # Predict forward using top mode
        pred_longs = {oid: t_state_pred(e.fused_mu, long_models, e.model_ps, prediction_steps) for
                      oid, e in est_long_res.items()}
        pred_lats = {oid: t_state_pred(e.fused_mu, lat_models, e.model_ps, prediction_steps) for oid, e
                     in est_lat_res.items()}

        traj_ob_model_data.append(
            RecHorStat(
                true_long={oid: ols[i] for oid, ols in obstacle_longs.items()},
                true_lat={oid: ols[i] for oid, ols in obstacle_lats.items()},
                observed_long=z_longs,
                observed_lat=z_lats,
                est_long=est_long_res,
                est_lat=est_lat_res,
                prediction_traj_long=pred_longs,
                prediction_traj_lat=pred_lats
            )
        )

    assert len(traj_ob_model_data) == sim_steps
    return dn_state_list, [scenario.obstacle_states_at_time_step(i) for i in range(start_step, start_step + sim_steps)], traj_ob_model_data


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
