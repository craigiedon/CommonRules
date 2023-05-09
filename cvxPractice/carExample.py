import time
from typing import List, Dict

import cvxpy as cp
import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.obstacle import Obstacle
from commonroad.scenario.state import CustomState, InitialState

from utils import obs_long_lats, mpc_result_to_dyn_obj
from TaskConfig import TaskConfig, PointMPCResult
from anim_utils import animate_scenario


def cvx_mpc(T: int, tc: TaskConfig, start_state: InitialState, obstacles: List[Obstacle], pred_longs: Dict[int, np.ndarray], pred_lats: Dict[int, np.ndarray]) -> PointMPCResult:
    # Z-State
    x = cp.Variable(T)
    y = cp.Variable(T)
    vx = cp.Variable(T)
    vy = cp.Variable(T)

    # U-Actions
    ax = cp.Variable(T)
    ay = cp.Variable(T)
    us = cp.vstack((ax, ay))

    # z_dot = Az @ zs + Bz @ us

    # w_gx = 0.0
    w_gy = 1.0
    w_ga = 0
    w_v = 1.0

    # prog_x = cp.sum_squares(x - tc.x_goal)
    vel_track = cp.sum_squares(vx - tc.v_goal)
    deviation_y = cp.sum_squares(y - tc.lanes[0])
    min_lat_accs = cp.sum_squares(ay)

    # (Maybe Impossible without Smooth Max): Lane deviation constraint
    # lane_bools = cp.Variable(T, boolean=True)
    # lane_devs = cp.square(cp.vstack((y - tc.lanes[0], y - tc.lanes[1])))
    # ld_items = cp.min(lane_devs, axis=0)
    # lane_dev_cost = cp.sum(cp.log_sum_exp(lane_devs, axis=0))
    # lane_dev_cost = cp.pnorm(lane_devs, 5, axis=0)
    # ld_items = cp.multiply(lane_bools, lane_devs[0, :]) + cp.multiply(lane_bools, lane_devs[1, :])
    # lane_dev_cost = cp.sum(ld_items)

    # TODO: Faster-than left
    # I think...maybe I put the potential fields stuff in here?
    # exp_cost = cp.sum(cp.exp(-vx))

    # TODO: Keep distance from other cars potential field-wise

    obj = cp.Minimize(
        w_gy * deviation_y + w_v * vel_track + w_ga * min_lat_accs)  # + w_ga * min_lat_accs)  # + w_ga * min_lat_accs)

    start_constraint = [
        x[0] == start_state.position[0],
        y[0] == start_state.position[1],
        vx[0] == np.cos(start_state.orientation) * start_state.velocity,
        vy[0] == np.sin(start_state.orientation) * start_state.velocity
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
        y >= -3.5 + tc.car_width / 2.0,
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
        (ax[1:] - ax[:-1]) ** 2 <= 20 * tc.dt,
        (ay[1:] - ay[:-1]) ** 2 <= 20 * tc.dt
        # axj_min <= 0 <= axj_max
        # ayj_min <= 0 <= ayj_max
    ]


    obs_avoid_cs = []
    M = 10000
    for o in obstacles:
        # obs_pos = np.array([o.state_at_time(i).position for i in range(T)])
        # obs_xs = obs_pos[:, 0]
        # obs_ys = obs_pos[:, 1]
        obs_xs = pred_longs[o.obstacle_id][:, 0]
        obs_ys = pred_lats[o.obstacle_id][:, 0]

        obs_l = o.obstacle_shape.length
        obs_w = o.obstacle_shape.width

        diag_d = np.sqrt(tc.car_length ** 2 + tc.car_width ** 2) / 2.0

        obs_xmin = obs_xs - (obs_l / 2.0) - diag_d
        obs_xmax = obs_xs + (obs_l / 2.0) + diag_d
        obs_ymin = obs_ys - (obs_w / 2.0) - diag_d
        obs_ymax = obs_ys + (obs_w / 2.0) + diag_d

        b = cp.Variable((T, 4), boolean=True)

        disj_cs = [
            x <= obs_xmin + M * b[:, 0],
            obs_xmax <= x + M * b[:, 1],
            y <= obs_ymin + M * b[:, 2],
            obs_ymax <= y + M * b[:, 3],
            cp.sum(b, axis=1) <= 3
        ]

        obs_avoid_cs.extend(disj_cs)

    constraints = [
        *start_constraint,
        *pos_lims,
        prop_vels,
        *acc_lims,
        # *jerk_lims,
        *vel_lims,
        *f_dyns,
        # *disj_cs
        *obs_avoid_cs
    ]

    prob = cp.Problem(obj, constraints)

    # print(cp.installed_solvers())

    start_time = time.time()
    # result = prob.solve(solver="SCIP")
    result = prob.solve(solver="SCIP")
    print("Solve Time: ", time.time() - start_time)

    print("Obj func: ", result)
    # print("X Vals:", x.value)
    # print("Y Vals:", y.value)

    # plt.subplot(211)
    # plt.plot(range(T), x.value)
    # plt.xlabel("T")
    # plt.ylabel("X")
    #
    # plt.subplot(212)
    # plt.plot(range(T), y.value)
    # plt.xlabel("T")
    # plt.ylabel("Y")
    #
    # plt.show()

    # plt.scatter(x.value, y.value)
    # o_rect = patches.Rectangle((obs_xs[0] - obs_l / 2.0, obs_ys[0] - obs_w / 2.0), obs_l, obs_w)
    # plt.gca().add_patch(o_rect)
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.show()

    # print("B: ", b.value)
    print(prob.solution.status)
    if prob.solution.status == cp.OPTIMAL:
        return PointMPCResult(xs=np.array(x.value), ys=np.array(y.value), vs_x=np.array(vx.value), vs_y=np.array(vy.value),
                          as_x=np.array(ax.value), as_y=np.array(ay.value))

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

    start_state = InitialState(position=np.array([10.0, ego_lane_centres[0]]), velocity=task_config.v_goal * 0.2,
                               orientation=0, time_step=0)
    T = int(round(end_time / task_config.dt))

    pred_longs, pred_lats = obs_long_lats(scenario.obstacles, 0, T)

    res = cvx_mpc(T, task_config, start_state, scenario.obstacles, pred_longs, pred_lats)
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
