from dataclasses import dataclass
from typing import List, Any, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np

import casadi
from commonroad.scenario.obstacle import DynamicObstacle, StaticObstacle
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import InitialState, KSState, CustomState
from commonroad.scenario.trajectory import State
from matplotlib import patches, transforms
from matplotlib.transforms import Affine2D


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


def car_mpc(start_time: float, end_time: float, start_state: State, task: TaskConfig,
            static_obstacles: List[StaticObstacle], dynamic_obstacles: List[DynamicObstacle],
            cw: Optional[CostWeights] = None, prev_solve: CarMPCRes = None) -> CarMPCRes:
    # print(start_state.position)
    # print(start_state.velocity)
    # print(start_state.acceleration)
    opti = casadi.Opti()
    start_step = int(np.round(start_time / task.dt))
    end_step = int(np.round(end_time / task.dt))
    T = end_step - start_step

    xs = opti.variable(T)
    ys = opti.variable(T)
    vs = opti.variable(T)
    hs = opti.variable(T)
    accs = opti.variable(T)
    ang_vels = opti.variable(T)
    lane_selectors = opti.variable(T, len(task.lanes))
    lane_params = opti.parameter(len(task.lanes))
    opti.set_value(lane_params, task.lanes)

    # 4 bs per rectangle (ego + 1st static obstacle 8 == 4 * 2)
    # collision_slacks = opti.variable(T, 8 * len(static_obstacles) + 8 * len(dynamic_obstacles))

    x_span = max(1.0, abs(task.x_goal - start_state.position[0]))
    y_span = max(1.0, abs(task.y_goal - start_state.position[1]))

    # Distance to destination cost
    x_progress_cost = casadi.sumsqr((xs - task.x_goal) / x_span)  # / (start_state.x - task.x_goal))
    y_progress_cost = casadi.sumsqr((ys - task.y_goal) / y_span)  # / (start_state.y - task.y_goal))

    # Track the reference velocity
    vel_track_cost = casadi.sumsqr(vs - task.v_goal * task.dt)

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
    lateral_slack = 0.9
    if len(static_obstacles) > 0:
        s_pots = []
        for s_obs in static_obstacles:
            s_x, s_y = s_obs.initial_state.position
            s_dist_x = ((xs - s_x) / s_obs.obstacle_shape.length) ** 2
            s_dist_y = ((ys - s_y) / (s_obs.obstacle_shape.width * lateral_slack)) ** 2
            s_pots.append(casadi.exp(-task.collision_field_slope * (s_dist_x + s_dist_y)))
        s_pots = casadi.vcat(s_pots)
        s_obs_cost = casadi.sum2(casadi.sum1(s_pots))
    else:
        s_obs_cost = 0.0

    if len(dynamic_obstacles) > 0:
        d_pots = []
        for d_obs in dynamic_obstacles:
            if start_step == 0:
                d_states = [d_obs.initial_state] + d_obs.prediction.trajectory.states_in_time_interval(start_step + 1,
                                                                                                       end_step - 1)
            else:
                d_states = d_obs.prediction.trajectory.states_in_time_interval(start_step, end_step - 1)

            d_xs = np.array([ds.position[0] for ds in d_states])
            d_dist_x = ((xs - d_xs) / d_obs.obstacle_shape.length) ** 2

            d_ys = np.array([ds.position[1] for ds in d_states])
            d_dist_y = ((ys - d_ys) / (d_obs.obstacle_shape.width * lateral_slack)) ** 2
            d_pots.append(casadi.exp(-task.collision_field_slope * (d_dist_x + d_dist_y)))
        d_pots = casadi.vcat(d_pots)
        d_obs_cost = casadi.sum2(casadi.sum1(d_pots))
    else:
        d_obs_cost = 0.0

    if len(dynamic_obstacles) > 0:
        f_left_costs = []
        for d_obs in dynamic_obstacles:
            if start_step == 0:
                d_states = [d_obs.initial_state] + d_obs.prediction.trajectory.states_in_time_interval(start_step + 1,
                                                                                                       end_step - 1)
            else:
                d_states = d_obs.prediction.trajectory.states_in_time_interval(start_step, end_step - 1)

            d_xs = np.array([ds.position[0] for ds in d_states])
            d_ys = np.array([ds.position[1] for ds in d_states])
            d_vs = np.array([ds.velocity for ds in d_states])

            d_dist_x = ((xs - d_xs) / d_obs.obstacle_shape.length) ** 2.0
            x_pot = 10 * casadi.exp(-d_dist_x)

            d_dist_y = casadi.fmax(0.0, (d_ys + d_obs.obstacle_shape.width / 2.0) - (ys - task.car_height / 2.0))
            # d_dist_y = casadi.fmin(d_dist_y, 1.0)

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
                  cw.collision_pot * s_obs_cost +
                  cw.collision_pot * d_obs_cost +
                  cw.faster_left * f_left_cost +
                  cw.braking * braking_cost
                  )

    # Start State Constraints
    opti.subject_to(xs[0] == start_state.position[0])
    opti.subject_to(ys[0] == start_state.position[1])

    opti.subject_to(vs[0] == start_state.velocity)
    opti.subject_to(hs[0] == start_state.orientation)

    # Variable Bounds
    opti.subject_to(opti.bounded(-task.v_max * task.dt, casadi.vec(vs), task.v_max * task.dt))
    opti.subject_to(opti.bounded(-casadi.pi / 2.0, casadi.vec(hs), casadi.pi / 2.0))
    opti.subject_to(opti.bounded(-task.acc_max * task.dt, casadi.vec(accs), task.acc_max * task.dt))
    opti.subject_to(opti.bounded(-task.ang_vel_max * task.dt, casadi.vec(ang_vels), task.ang_vel_max * task.dt))
    opti.subject_to(opti.bounded(0, casadi.vec(lane_selectors), 1))

    # Lane Bounds
    opti.subject_to(opti.bounded(task.y_bounds[0] + task.car_height / 2.0, casadi.vec(ys),
                                 task.y_bounds[1] - task.car_height / 2.0))

    # State Evolution
    opti.subject_to(xs[1:] == xs[:-1] + casadi.cos(hs[:-1]) * vs[:-1])
    opti.subject_to(ys[1:] == ys[:-1] + casadi.sin(hs[:-1]) * vs[:-1])
    opti.subject_to(vs[1:] == vs[:-1] + accs[:-1])
    opti.subject_to(hs[1:] == hs[:-1] + ang_vels[:-1])

    # Lane Selection Simplex
    opti.subject_to(casadi.sum2(lane_selectors) == 1)

    # opti.solver('ipopt', {"ipopt.print_level": 0, "ipopt.max_iter": 10000})
    opti.solver('ipopt', {"ipopt.check_derivatives_for_naninf": "yes", "ipopt.max_iter": 10000})

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
        flat_costs = car_mpc(i * task_config.dt, i * task_config.dt + horizon_length, current_state, task_config,
                             scenario.static_obstacles, scenario.dynamic_obstacles, None)
        res = car_mpc(i * task_config.dt, i * task_config.dt + horizon_length, current_state, task_config,
                      scenario.static_obstacles, scenario.dynamic_obstacles, cws, flat_costs)
        current_state = CustomState(position=np.array([res.xs[1], res.ys[1]]), velocity=res.vs[1],
                                    orientation=res.hs[1],
                                    acceleration=res.accs[1], time_step=i)
        dn_state_list.append(current_state)

    return dn_state_list


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
