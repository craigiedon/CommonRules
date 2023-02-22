from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spt
from scipy.optimize import OptimizeResult


@dataclass
class RectObstacle:
    x : float
    y: float
    w : float # Width
    h: float # Height
    rot: float # Radians

@dataclass
class PolyConstraint:
    A: np.ndarray
    b: np.ndarray

def to_poly_constraints(r: RectObstacle) -> PolyConstraint:
    A = np.array([
        [np.sin(r.rot), -np.cos(r.rot)],
        [-np.sin(r.rot), np.cos(r.rot)],
        [np.cos(r.rot), np.sin(r.rot)],
        [-np.cos(r.rot), -np.sin(r.rot)],
    ])

    b = np.array([r.h, r.h, r.w, r.w]) / 2.0 + A @ np.array([r.x, r.y])
    return PolyConstraint(A = A, b=b)

def poly_contains_point(x: np.ndarray, p: PolyConstraint) -> bool:
    return np.all(p.A @ x <= p.b)

car = RectObstacle(50.0, 100.0, 4.0, 3.0, 0.0)
car_poly = to_poly_constraints(car)

containment = poly_contains_point(np.array([50, 100.0]) + np.array([1.9, 1.0]), car_poly)

@dataclass
class CarState:
    x: float
    y: float
    v: float
    heading: float

    def to_array(self):
        return np.array([self.x, self.y, self.v, self.heading])

    def n_dims(self):
        return len(self.to_array())

@dataclass
class CarAction:
    acc: float
    ang_vel: float

    def to_array(self):
        return np.array([self.acc, self.ang_vel])

    def n_dims(self):
        return len(self.to_array())

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
    lanes: np.ndarray

def car_mpc(start_state: CarState, task: TaskConfig, static_obstacles: List[RectObstacle]) -> OptimizeResult:
    s_0 = start_state.to_array()
    state_dims = len(s_0)
    action_dims = CarAction(0.0, 0.0).n_dims()
    lane_dims = len(task.lanes)

    obj_fun = lambda d_vars: interstate_objective(d_vars, start_state, task)

    cons = (
        {"type": "eq", "fun": lambda d_vars: start_constraints(d_vars, task, start_state)},
        {"type": "eq", "fun": lambda d_vars: state_evolution(d_vars, task)},
        {"type": "eq", "fun": lambda d_vars: lane_selection_simplex(d_vars, task)}
    )

    lbs = -np.ones((task.T, state_dims + action_dims + lane_dims)) * np.inf
    ubs = np.ones((task.T, state_dims + action_dims + lane_dims)) * np.inf

    lbs[:, 2] = -task.v_max
    ubs[:, 2] = task.v_max

    lbs[:, 3] = -np.pi / 2.0
    ubs[:, 3] = np.pi / 2.0

    lbs[:, 4] = -task.acc_max
    ubs[:, 4] = task.acc_max

    lbs[:, 5] = -task.ang_vel_max
    ubs[:, 5] = task.ang_vel_max

    lbs[:, 6:9] = 0.0
    ubs[:, 6:9] = 1.0


    state_guess = np.random.random((task.T, state_dims + action_dims + lane_dims)).reshape(-1)

    return spt.minimize(obj_fun, state_guess, constraints=cons, bounds=spt.Bounds(lbs.reshape(-1), ubs.reshape(-1)))


def interstate_objective(d_vars: np.ndarray, start_state: CarState, task: TaskConfig):
    reshaped_vars = d_vars.reshape(task.T, -1)
    xs = reshaped_vars[:, 0]
    ys = reshaped_vars[:, 1]
    vs = reshaped_vars[:, 2]
    hs = reshaped_vars[:, 3]
    accs = reshaped_vars[:, 4]
    ang_vels = reshaped_vars[:, 5]
    lane_selectors = reshaped_vars[:, 6:9]

    # Distance to destination cost
    x_progress_cost = (((xs - task.x_goal) / (start_state.x - task.x_goal)) ** 2).sum()
    y_progress_cost = (((ys - task.y_goal) / (start_state.y - task.y_goal)) ** 2).sum()

    # Track the reference velocity
    vel_track_cost = ((vs - task.v_goal) ** 2).sum()

    # Keep velocity low
    acc_cost = (accs ** 2).sum()

    # Angular Vel Cost
    ang_vel_cost = (ang_vels ** 2).sum()

    # Minimize jerk
    jerk_cost = ((ang_vels[1:] - ang_vels[:-1]) ** 2).sum()

    # Align with road curvature
    road_align_cost = (hs ** 2).sum()

    # Lane alignment
    chosen_lanes = lane_selectors @ task.lanes
    lane_align_cost = ((ys - chosen_lanes) ** 2).sum()

    return x_progress_cost + y_progress_cost + vel_track_cost + acc_cost + ang_vel_cost + jerk_cost + road_align_cost + lane_align_cost

def state_evolution(d_vars, task: TaskConfig):
    reshaped_vars=  d_vars.reshape(task.T, -1)
    xs = reshaped_vars[:, 0]
    ys = reshaped_vars[:, 1]
    vs = reshaped_vars[:, 2]
    hs = reshaped_vars[:, 3]
    accs = reshaped_vars[:, 4]
    ang_vs = reshaped_vars[:, 5]

    return np.concatenate((
        xs[1:] - (xs[:-1] + np.cos(hs[:-1]) * vs[:-1]),
        ys[1:] - (ys[:-1] + np.sin(hs[:-1]) * vs[:-1]),
        vs[1:] - (vs[:-1] + accs[:-1]),
        hs[1:] - (hs[:-1] + ang_vs[:-1])
    ))


def start_constraints(d_vars, task: TaskConfig, start_state: CarState):
    reshaped_vars = d_vars.reshape(task.T, -1)
    state_starts = reshaped_vars[0, 0:start_state.n_dims()]

    return state_starts - start_state.to_array()


def lane_selection_simplex(d_vars, task: TaskConfig):
    reshaped_vars = d_vars.reshape(task.T, -1)
    lane_selectors = reshaped_vars[:, 6:9]
    lane_sums = np.sum(lane_selectors, axis=1)
    return lane_sums - 1.0

task_config = TaskConfig(20, 700.0, 5.0, 2.0, 1.0, 5.0, 30.0, 1.0, 1.0, np.array([-5.0, 0.0, 5.0]))
res = car_mpc(CarState(0.0, 0.0, 0.0, 0.0), task_config, [])
res_vals = res.x.reshape(task_config.T, -1)

print("xs:", res_vals[:, 0])
print("ys:", res_vals[:, 1])

n_subplots = 4

plt.subplot(n_subplots, 1, 1)
plt.plot(range(task_config.T), res_vals[:, 0])
plt.xlabel("t")
plt.ylabel("X")

plt.subplot(n_subplots, 1, 2)
plt.plot(range(task_config.T), res_vals[:, 1])
plt.ylim(-5, 5)
plt.xlabel("t")
plt.ylabel("Y")

plt.subplot(n_subplots, 1, 3)
plt.plot(range(task_config.T), res_vals[:, 2])
plt.xlabel("t")
plt.ylabel("Velocity")

plt.subplot(n_subplots, 1, 4)
plt.plot(range(task_config.T), res_vals[:, 3])
plt.ylim(-np.pi, np.pi)
plt.xlabel("t")
plt.ylabel("$\\theta$")

plt.show()