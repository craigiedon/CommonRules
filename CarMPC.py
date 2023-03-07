from dataclasses import dataclass
from typing import List, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np

import casadi
from matplotlib import patches, transforms
from matplotlib.transforms import Affine2D

@dataclass
class IntervalConstraint:
    value: Any
    time_interval: Tuple[float, float]

@dataclass
class TaskConfig:
    time: float # Seconds
    dt: float # steps/second
    x_goal: float # Metres
    y_goal: float # Metres
    car_width: float # Metres
    car_height: float # Metres
    v_goal: float # Metres/Sec
    v_max: float # Metres/Sec
    acc_max: float # Metres/Sec**2
    ang_vel_max: float # Metres/Sec
    lanes: List[float] # Metres
    lane_targets: List[IntervalConstraint]

@dataclass
class CarState:
    x: float
    y: float
    v: float
    heading: float

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
    x : float
    y: float
    w : float # Width
    h: float # Height
    rot: float # Radians

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
    r = RectCustom((obs.x - obs.w / 2.0, obs.y - obs.h / 2.0), obs.w, obs.h, np.rad2deg(obs.rot), facecolor=color, alpha=alpha)
    return r

def to_poly_constraints(r: RectObstacle) -> PolyConstraint:
    A = casadi.blockcat([
        [casadi.sin(r.rot), -casadi.cos(r.rot)],
        [-casadi.sin(r.rot), casadi.cos(r.rot)],
        [casadi.cos(r.rot), casadi.sin(r.rot)],
        [-casadi.cos(r.rot), -casadi.sin(r.rot)],
    ])

    b = casadi.vcat([r.h, r.h, r.w, r.w]) / 2.0 + A @ casadi.vcat([r.x, r.y])
    return PolyConstraint(A = A, b=b)

def car_mpc(start_state: CarState, task: TaskConfig, static_obstacles: List[RectObstacle], prev_solve=None) -> CarMPCRes:
    opti = casadi.Opti()
    T = int(task.time/task.dt) # time steps

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
    collision_slacks = opti.variable(T, 8 * len(static_obstacles))

    # Distance to destination cost
    x_progress_cost = casadi.sumsqr((xs - task.x_goal) / (start_state.x - task.x_goal))
    y_progress_cost = casadi.sumsqr((ys - task.y_goal) / (start_state.y - task.y_goal))

    # Track the reference velocity
    vel_track_cost = casadi.sumsqr(vs - task.v_goal * task.dt)

    # Keep velocity low
    acc_cost = casadi.sumsqr(accs)

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

    chosen_lanes = lane_selectors @ lane_params
    lane_align_cost = casadi.sumsqr(ys - chosen_lanes)

    opti.minimize(x_progress_cost +
                  y_progress_cost +
                  vel_track_cost +
                  acc_cost +
                  ang_vel_cost +
                  jerk_cost +
                  road_align_cost +
                  lane_align_cost)

    # Start State Constraints
    opti.subject_to(xs[0] == start_state.x)
    opti.subject_to(ys[0] == start_state.y)
    opti.subject_to(vs[0] == start_state.v)
    opti.subject_to(hs[0] == start_state.heading)

    # Variable Bounds
    opti.subject_to(opti.bounded(-task.v_max * task.dt, casadi.vec(vs), task.v_max * task.dt))
    opti.subject_to(opti.bounded(-casadi.pi / 2.0, casadi.vec(hs), casadi.pi / 2.0))
    opti.subject_to(opti.bounded(-task.acc_max * task.dt, casadi.vec(accs), task.acc_max * task.dt))
    opti.subject_to(opti.bounded(-task.ang_vel_max * task.dt, casadi.vec(ang_vels), task.ang_vel_max * task.dt))
    opti.subject_to(opti.bounded(0.0, casadi.vec(lane_selectors), 1.0))

    # State Evolution
    opti.subject_to(xs[1:] == xs[:-1] + casadi.cos(hs[:-1]) * vs[:-1])
    opti.subject_to(ys[1:] == ys[:-1] + casadi.sin(hs[:-1]) * vs[:-1])
    opti.subject_to(vs[1:] == vs[:-1] + accs[:-1])
    opti.subject_to(hs[1:] == hs[:-1] + ang_vels[:-1])

    # Lane Selection Simplex
    opti.subject_to(casadi.sum2(lane_selectors) == 1.0)

    # Avoid Obstacles
    # TODO: Add *dynamic* obstacles (i.e., find the traj pred class in commonroad)
    if len(static_obstacles) > 0:
        opti.subject_to(casadi.vec(collision_slacks) >= 0.0)
        for t in range(T):
            # Split up the slack variables by timestep, and by obstacle
            c_slacks_t = casadi.horzsplit(collision_slacks[t, :], np.arange(0, len(static_obstacles) * 8 + 1, 8))
            for s_obs, c_slack in zip(static_obstacles, c_slacks_t):
                ego_poly = to_poly_constraints(RectObstacle(xs[t], ys[t], task.car_width, task.car_height, hs[t]))
                obs_poly = to_poly_constraints(s_obs)

                full_A = casadi.vcat([ego_poly.A, obs_poly.A])
                full_b = casadi.vcat([ego_poly.b, obs_poly.b])

                # Use the slack variables to create a constraint
                opti.subject_to(full_A.T @ c_slack.T == 0)
                opti.subject_to(full_b.T @ c_slack.T < 0)


    # opti.solver('ipopt', {"ipopt.print_level": 3})
    opti.solver('ipopt')

    if prev_solve is not None:
        opti.set_initial(prev_solve.value_variables())

    sol = opti.solve()

    return CarMPCRes(sol.value(xs), sol.value(ys), sol.value(vs), sol.value(hs), sol.value(accs), sol.value(ang_vels))

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
        axs_2.add_patch(obs_to_patch(RectObstacle(x,y, task.car_width, task.car_height, h), color='b', alpha=0.2))
    axs_2.scatter(res.xs, res.ys, marker='x')
    axs_2.set_xlabel("X")
    axs_2.set_ylabel("Y")

    plt.show()


def run():
    # Ford Escort Config. See Commonroad Vehicle Model Documentation
    task_config = TaskConfig(time=5,
                             dt=0.1,
                             x_goal=700.0,
                             y_goal=5.0,
                             car_width=4.298,
                             car_height=1.674,
                             v_goal=31.29, # == 70mph
                             v_max=45.8,
                             acc_max=11.5,
                             ang_vel_max=0.4,
                             lanes=[-5.0, 0.0, 5.0])
    obstacles = [
        RectObstacle(40.0, 4.5, 4.3, 1.674, 0.0),
        RectObstacle(5.0, 4.5, 4.3, 1.674, 0.0)
    ]

    res = car_mpc(CarState(0.0, 0.0, 0.0, 0.0), task_config, obstacles)

    plot_results(res, task_config, obstacles)


if __name__ == "__main__":
    run()

