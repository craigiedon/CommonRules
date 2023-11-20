import time
from dataclasses import dataclass
from typing import List, Dict, Optional

import cvxpy as cp
import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.obstacle import Obstacle
from commonroad.scenario.state import CustomState, InitialState

from utils import obs_long_lats, mpc_result_to_dyn_obj
from TaskConfig import TaskConfig, PointMPCResult
from anim_utils import animate_scenario


@dataclass
class CVXInterstateProblem:
    prob: cp.Problem
    start_params: cp.Parameter
    obs_xs: cp.Parameter
    obs_ys: cp.Parameter
    obs_v_x: cp.Parameter
    xs: cp.Variable
    ys: cp.Variable
    vs_x: cp.Variable
    vs_y: cp.Variable
    as_x: cp.Variable
    as_y: cp.Variable


def create_cvx_mpc(T: int, tc: TaskConfig, obstacles: List[Obstacle]) -> CVXInterstateProblem:
    # Z-State
    x = cp.Variable(T)
    y = cp.Variable(T)
    vx = cp.Variable(T)
    vy = cp.Variable(T)

    # U-Actions
    ax = cp.Variable(T)
    ay = cp.Variable(T)
    us = cp.vstack((ax, ay))

    # Obstacles Predicted locations/velocities
    obs_xs = cp.Parameter((len(obstacles), T))
    obs_ys = cp.Parameter((len(obstacles), T))
    obs_v_x = cp.Parameter((len(obstacles), T))

    # z_dot = Az @ zs + Bz @ us

    # w_gx = 0.0
    w_gy = 10.0
    w_ga = 50.0
    w_j = 10
    w_v = 0.5

    # exp_cost = cp.sum(cp.abs(obs_xs - obs_xs))
    # prog_x = cp.sum_squares(x - tc.x_goal)
    vel_track = cp.sum_squares(vx - tc.v_goal)
    deviation_y = cp.sum_squares(y - tc.lanes[0])
    min_lat_accs = cp.sum_squares(ay)
    min_jerk = cp.sum_squares(cp.diff(ay))

    obj = cp.Minimize(
        w_gy * deviation_y + w_v * vel_track + w_ga * min_lat_accs + w_j * min_jerk)  # + w_ga * min_lat_accs)  # + w_ga * min_lat_accs)
    # obj = cp.Minimize(exp_cost)

    start_params = cp.Parameter(4)

    start_constraint = [
        x[0] == start_params[0],
        y[0] == start_params[1],
        vx[0] == start_params[2],
        vy[0] == start_params[3]
    ]

    f_dyns = [
        # zs[:, 1:] == zs[:, :-1] + tc.dt * z_dot[:, :-1]
        x[1:] == x[:-1] + tc.dt * vx[:-1],
        y[1:] == y[:-1] + tc.dt * vy[:-1],
        vx[1:] == vx[:-1] + tc.dt * ax[:-1],
        vy[1:] == vy[:-1] + tc.dt * ay[:-1]
    ]

    prop_vels = (vx >= 1.5 * cp.abs(vy))

    pos_lims = [
        x >= 0,
        x <= 250,
        y >= 0.0 + tc.car_width / 2.0,
        y <= 7.0 - tc.car_width / 2.0
    ]

    acc_lims = [
        -tc.acc_max <= ax,
        ax <= tc.acc_max,
        -tc.acc_max <= ay,
        ay <= tc.acc_max
    ]

    vel_lims = [
        0 <= vx,
        vx <= tc.v_max,
        -tc.v_max <= vy,
        vy <= tc.v_max
    ]

    jerk_lims = [
        (ax[1:] - ax[:-1]) ** 2 <= (2 * tc.dt),
        (ay[1:] - ay[:-1]) ** 2 <= (2 * tc.dt)
        # axj_min <= 0 <= axj_max
        # ayj_min <= 0 <= ayj_max
    ]

    M = 10000

    obs_avoid_cs = []
    obs_fl_cs = []
    for i, o in enumerate(obstacles):
        # obs_pos = np.array([o.state_at_time(i).position for i in range(T)])
        # obs_xs = obs_pos[:, 0]
        # obs_ys = obs_pos[:, 1]
        # obs_xs = pred_longs[o.obstacle_id][:, 0]
        # obs_ys = pred_lats[o.obstacle_id][:, 0]

        o_l = o.obstacle_shape.length
        o_w = o.obstacle_shape.width

        diag_d = np.sqrt(tc.car_length ** 2 + tc.car_width ** 2) / 2.0

        o_xmin = obs_xs[i, :] - (o_l / 2.0) - diag_d  # tc.car_length / 2.0
        o_xmax = obs_xs[i, :] + (o_l / 2.0) + diag_d  # tc.car_length / 2.0
        o_ymin = obs_ys[i, :] - (o_w / 2.0) - diag_d  # tc.car_width / 2.0
        o_ymax = obs_ys[i, :] + (o_w / 2.0) + diag_d  # tc.car_width / 2.0

        b_col = cp.Variable((T, 4), boolean=True)

        # TODO: Could expand the xmin to take the potentials into account?
        # behind_pot = (vx ** 2) / (-2 * tc.acc_max)
        # front_pot = (obs_v_x[i, :] ** 2) / (-2 * tc.acc_max)
        # safe_dist = behind_pot - front_pot + front_v * reaction_time
        # safe_dist = front_pot - behind_pot + vx * 0.1

        disj_cs = [
            x + tc.dt * vx + tc.dt ** 2 * ax <= o_xmin + M * b_col[:, 0],
            o_xmax + tc.dt * vx + tc.dt ** 2 * ax <= x + M * b_col[:, 1],
            y + tc.dt * vy + tc.dt ** 2 * ay <= o_ymin + M * b_col[:, 2],
            o_ymax + tc.dt * vy + tc.dt ** 2 * ay <= y + M * b_col[:, 3],
            cp.sum(b_col, axis=1) <= 3
        ]

        b_f_left = cp.Variable((T, 4), boolean=True)
        fb_slack = 4

        faster_left_constraint = [
            x <= o_xmin - fb_slack + M * b_f_left[:, 0],
            o_xmax <= x - fb_slack + M * b_f_left[:, 1],
            o_ymax <= y + M * b_f_left[:, 2],
            # obs_ys[i, :] <= y + M * b_f_left[:, 2],
            vx + tc.dt * ax <= obs_v_x[i, :] + M * b_f_left[:, 3],
            # ax <= 0.0 + M * b_f_left[:, 4],
            # x[1:] - x[:-1] <= obs_v_x[i, 1:] - obs_v_x[i, :-1] + M * b_f_left[1:, 3],
            cp.sum(b_f_left, axis=1) <= 3
        ]

        obs_avoid_cs.extend(disj_cs)
        obs_fl_cs.extend(faster_left_constraint)

    constraints = [
        *start_constraint,
        *pos_lims,
        prop_vels,
        *acc_lims,
        # *jerk_lims,
        *vel_lims,
        *f_dyns,
        *obs_avoid_cs,
        # *obs_fl_cs
    ]

    prob = cp.Problem(obj, constraints)

    return CVXInterstateProblem(prob, start_params, obs_xs, obs_ys, obs_v_x, x, y, vx, vy, ax, ay)


def cvx_mpc(prob_config: CVXInterstateProblem, start_state, obstacles, pred_longs, pred_lats, verbose: bool = False) -> Optional[
    PointMPCResult]:
    prob_config.start_params.value = [start_state.position[0],
                                      start_state.position[1],
                                      np.cos(start_state.orientation) * start_state.velocity,
                                      np.sin(start_state.orientation) * start_state.velocity]

    prob_config.obs_xs.value = np.stack([pred_longs[o.obstacle_id][:, 0] for o in obstacles])
    prob_config.obs_ys.value = np.stack([pred_lats[o.obstacle_id][:, 0] for o in obstacles])
    prob_config.obs_v_x.value = np.stack([pred_longs[o.obstacle_id][:, 1] for o in obstacles])

    start_time = time.time()
    # print(cp.installed_solvers())
    prob_config.prob.solve(solver="GUROBI", verbose=verbose)
    # print("Solve Time: ", time.time() - start_time)
    # print(prob_config.prob.solution.status)

    try:
        if prob_config.prob.solution.status == cp.OPTIMAL:
            return PointMPCResult(xs=np.array(prob_config.xs.value), ys=np.array(prob_config.ys.value),
                                  vs_x=np.array(prob_config.vs_x.value),
                                  vs_y=np.array(prob_config.vs_y.value),
                                  as_x=np.array(prob_config.as_x.value), as_y=np.array(prob_config.as_y.value))

        return None
    except Exception as e:
        print("Failed at the CVX Stage (With an exception? I thought I was supposed to catch these!")
        return None



def run():
    file_path = "../scenarios/Complex.xml"
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open()
    all_lane_centres = [scenario.lanelet_network.find_lanelet_by_id(l).center_vertices[0][1] for l in [3, 6, 10]]
    ego_lane_centres = [scenario.lanelet_network.find_lanelet_by_id(l).center_vertices[0][1] for l in [6, 10]]
    lane_widths = np.abs((ego_lane_centres[0] - ego_lane_centres[1]) / 2.0)

    goal_state = planning_problem_set.find_planning_problem_by_id(1).goal.state_list[0].position.center
    end_time = 6.0
    task_config = TaskConfig(dt=0.1,
                             x_goal=goal_state[0],
                             y_goal=goal_state[1],
                             y_bounds=(0.0, 7.0),
                             car_length=4.298,
                             car_width=1.674,
                             v_goal=31.29,  # == 70mph
                             v_max=45.8,
                             acc_max=11.5,
                             ang_vel_max=0.5,
                             lanes=ego_lane_centres,
                             lane_targets=[],
                             collision_field_slope=1.0)

    start_state = InitialState(position=np.array([10.0, ego_lane_centres[0]]), velocity=task_config.v_goal - 15.0,
                               orientation=0, time_step=0)
    T = int(round(end_time / task_config.dt))

    pred_longs, pred_lats = obs_long_lats(scenario.obstacles, 0, T)

    prob_config = create_cvx_mpc(T, task_config, scenario.obstacles)
    res = cvx_mpc(prob_config, start_state, scenario.obstacles, pred_longs, pred_lats)
    dn_state_list = []
    for i in range(T):
        s = CustomState(position=np.array([res.xs[i], res.ys[i]]),
                        velocity=np.sqrt(res.vs_x[i] ** 2 + res.vs_y[i] ** 2),
                        orientation=np.arctan2(res.vs_y[i], res.vs_x[i]),
                        time_step=i)
        dn_state_list.append(s)

    ego_soln_obj = mpc_result_to_dyn_obj(100, dn_state_list, task_config.car_width,
                                         task_config.car_length)
    scenario.add_objects(ego_soln_obj)

    animate_scenario(scenario, T, ego_v=ego_soln_obj, show=True)


if __name__ == "__main__":
    run()
