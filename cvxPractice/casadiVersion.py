from dataclasses import dataclass
from typing import List, Any
import matplotlib.pyplot as plt
import numpy as np

import casadi
from matplotlib import patches


@dataclass
class TaskConfig:
    T: int
    x_goal: float
    y_goal: float
    car_width: float
    car_height: float
    v_goal: float
    v_max: float
    acc_max: float
    ang_vel_max: float
    lanes: List[float]

@dataclass
class CarState:
    x: float
    y: float
    v: float
    heading: float

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

def obs_to_patch(obs: RectObstacle) -> patches.Rectangle:
    return patches.Rectangle((obs.x, obs.y), obs.w, obs.h, np.rad2deg(obs.rot), facecolor='r')

def to_poly_constraints(r: RectObstacle) -> PolyConstraint:
    A = casadi.blockcat([
        [casadi.sin(r.rot), -casadi.cos(r.rot)],
        [-casadi.sin(r.rot), casadi.cos(r.rot)],
        [casadi.cos(r.rot), casadi.sin(r.rot)],
        [-casadi.cos(r.rot), -casadi.sin(r.rot)],
    ])

    b = casadi.vcat([r.h, r.h, r.w, r.w]) / 2.0 + A @ casadi.vcat([r.x, r.y])
    return PolyConstraint(A = A, b=b)

def car_mpc(start_state: CarState, task: TaskConfig, static_obstacles: List[RectObstacle]):
    opti = casadi.Opti()

    xs = opti.variable(task.T)
    ys = opti.variable(task.T)
    vs = opti.variable(task.T)
    hs = opti.variable(task.T)
    accs = opti.variable(task.T)
    ang_vels = opti.variable(task.T)
    lane_selectors = opti.variable(task.T, len(task.lanes))

    lane_params = opti.parameter(len(task.lanes))
    opti.set_value(lane_params, task.lanes)

    # 4 bs per rectangle (ego + 1st static obstacle 8 == 4 * 2)
    collision_slacks = opti.variable(task.T, 8)

    # Distance to destination cost
    x_progress_cost = casadi.sumsqr((xs - task.x_goal) / (start_state.x - task.x_goal))
    y_progress_cost = casadi.sumsqr((ys - task.y_goal) / (start_state.y - task.y_goal))

    # Track the reference velocity
    vel_track_cost = casadi.sumsqr(vs - task.v_goal)

    # Keep velocity low
    acc_cost = casadi.sumsqr(accs)

    # Keep angular velocity low
    ang_vel_cost = casadi.sumsqr(ang_vels)

    # Minimize Jerk
    jerk_cost = casadi.sumsqr(ang_vels[1:] - ang_vels[:-1])

    # Align with road curvature
    road_align_cost = casadi.sumsqr(hs)

    # Lane alignment
    chosen_lanes = lane_selectors @ lane_params
    lane_align_cost = casadi.sumsqr(ys - chosen_lanes)

    opti.minimize(x_progress_cost +
                  y_progress_cost +
                  vel_track_cost +
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
    opti.subject_to(opti.bounded(-task.v_max, casadi.vec(vs), task.v_max))
    opti.subject_to(opti.bounded(-casadi.pi / 2.0, casadi.vec(hs), casadi.pi / 2.0))
    opti.subject_to(opti.bounded(-task.acc_max, casadi.vec(accs), task.acc_max))
    opti.subject_to(opti.bounded(-task.ang_vel_max, casadi.vec(ang_vels), task.ang_vel_max))
    opti.subject_to(opti.bounded(0.0, casadi.vec(lane_selectors), 1.0))

    # State Evolution
    opti.subject_to(xs[1:] == xs[:-1] + casadi.cos(hs[:-1]) * vs[:-1])
    opti.subject_to(ys[1:] == ys[:-1] + casadi.sin(hs[:-1]) * vs[:-1])
    opti.subject_to(vs[1:] == vs[:-1] + accs[:-1])
    opti.subject_to(hs[1:] == hs[:-1] + ang_vels[:-1])

    # Lane Selection Simplex
    opti.subject_to(casadi.sum2(lane_selectors) == 1.0)

    # Avoid Obstacles
    # TODO: Add multiple obstacles
    # TODO: Add *dynamic* obstacles (i.e., find the traj pred class in commonroad)
    opti.subject_to(casadi.vec(collision_slacks) >= 0.0)
    for t in range(task.T):
        ego_poly = to_poly_constraints(RectObstacle(xs[t], ys[t], task.car_width, task.car_height, hs[t]))
        obs_poly = to_poly_constraints(static_obstacles[0])

        full_A = casadi.vcat([ego_poly.A, obs_poly.A])
        full_b = casadi.vcat([ego_poly.b, obs_poly.b])
        print()

        # Use the slack variables to create a constraint
        c_slack = collision_slacks[t, :].T

        opti.subject_to(full_A.T @ c_slack == 0)
        opti.subject_to(full_b.T @ c_slack < 0)

        print()
        #opti.subject_to()

    opti.solver('ipopt')
    # opti.set_initial(sol1.value_variables())

    sol = opti.solve()

    n_subplots = 4

    fig, axs = plt.subplots(1, n_subplots)
    axs[0].plot(range(task.T), sol.value(xs))
    axs[0].set_xlabel("t")
    axs[0].set_ylabel("X")

    axs[1].plot(range(task.T), sol.value(ys))
    axs[1].set_ylim(-5, 5)
    axs[1].set_xlabel("t")
    axs[1].set_ylabel("Y")

    axs[2].plot(range(task.T), sol.value(vs))
    axs[2].set_xlabel("t")
    axs[2].set_ylabel("Velocity")

    axs[3].plot(range(task.T), sol.value(hs))
    axs[3].set_ylim(-np.pi, np.pi)
    axs[3].set_xlabel("t")
    axs[3].set_ylabel("$\\theta$")

    fig_2, axs_2 = plt.subplots(figsize=(15, 2))
    axs_2.add_patch(obs_to_patch(static_obstacles[0]))
    axs_2.scatter(sol.value(xs), sol.value(ys), marker='x')
    axs_2.set_xlabel("X")
    axs_2.set_ylabel("Y")

    plt.show()

    return sol

def run():
    task_config = TaskConfig(20, 700.0, 5.0, 2.0, 1.0, 5.0, 30.0, 1.0, 1.0, [-5.0, 0.0, 5.0])
    obstacle = RectObstacle(40.0, 4.5, 2.0, 1.0, 0.0)
    res = car_mpc(CarState(0.0, 0.0, 0.0, 0.0), task_config, [obstacle])

run()

