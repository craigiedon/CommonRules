import time
from typing import List

import numpy as np
import torch
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.geometry.shape import Rectangle
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType, StaticObstacle
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import InitialState, KSState, CustomState
from commonroad.scenario.trajectory import State, Trajectory
from commonroad.visualization.mp_renderer import MPRenderer
from matplotlib import pyplot as plt, animation
from matplotlib.animation import FuncAnimation

from CarMPC import TaskConfig, RectObstacle, car_mpc, IntervalConstraint, CostWeights, receding_horizon, \
    kalman_receding_horizon, pem_observation_batch
from KalmanPredictionVisuals import animate_kalman_predictions
from PyroGPClassification import load_gp_classifier
from PyroGPRegression import load_gp_reg
from immFilter import c_vel_long_model, c_acc_long_model, lat_model
from anim_utils import animate_with_predictions, animate_scenario


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


def run():
    file_path = "scenarios/Complex.xml"
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

    # Ford Escort Config. See Commonroad Vehicle Model Documentation
    all_lane_centres = [scenario.lanelet_network.find_lanelet_by_id(l).center_vertices[0][1] for l in [3, 6, 10]]
    ego_lane_centres = [scenario.lanelet_network.find_lanelet_by_id(l).center_vertices[0][1] for l in [6, 10]]

    goal_state = planning_problem_set.find_planning_problem_by_id(1).goal.state_list[0].position.center
    end_time = 5.0
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
                             # collision_field_slope=1e-5)
                             collision_field_slope=0.90)

    # start_state = InitialState(position=np.array([10.0, ego_lane_centres[0]]), velocity=0.0, orientation=0, time_step=0)
    start_state = InitialState(position=np.array([10.0, ego_lane_centres[0]]), velocity=task_config.v_goal * 0.2,
                               orientation=0, time_step=0)

    cws = CostWeights(x_prog=0.01, y_prog=0.1, acc=0.1, ang_v=10, jerk=1, v_track=2, lane_align=1, road_align=50,
                      collision_pot=500,
                      faster_left=1.0, braking=10)
    # cws = CostWeights(x_prog=0.01, y_prog=0.1, jerk=1, v_track=2.0, lane_align=1, road_align=1, collision_pot=1000,
    #                   faster_left=0.0, braking=1.0)

    long_models = [
        c_vel_long_model(task_config.dt, 1.0, 0.1),
        c_acc_long_model(task_config.dt, 1.0, 0.1)
    ]

    lat_models = [
        lat_model(task_config.dt, kd, 7, p_ref, 0.1, 0.1)
        for kd in np.linspace(3.0, 5.0, 3)
        for p_ref in all_lane_centres]

    det_pem = load_gp_classifier("models/nuscenes/vsgp_class", True)
    det_pem.eval()
    reg_pem = load_gp_reg("models/nuscenes/sgp_reg", True)
    reg_pem.eval()
    norm_mus = torch.load("data/nuscenes/inp_mus.pt")
    norm_stds = torch.load("data/nuscenes/inp_stds.pt")
    observation_func = lambda obs, ego_state, t, tlong, tlat, vs: pem_observation_batch(obs, ego_state, t, tlong, tlat,
                                                                                        det_pem, reg_pem, norm_mus,
                                                                                        norm_stds, vs)

    start_time = time.time()
    dn_state_list, prediction_stats = kalman_receding_horizon(end_time, 2.0, start_state, scenario, task_config,
                                                              long_models, lat_models, observation_func, cws)
    print("Receding Horizon Took: ", time.time() - start_time)

    dyn_obs = mpc_result_to_dyn_obj(100, dn_state_list, task_config.car_width, task_config.car_length)
    scenario.add_objects(dyn_obs)

    # plt.plot([s.acceleration for s in dn_state_list])
    # plt.show()

    ###  Show a visual that has the prediction parts too
    animate_with_predictions(scenario, prediction_stats, int(end_time / task_config.dt), show=True)

    animate_scenario(scenario, int(end_time / task_config.dt),
                     ego_v=dyn_obs, show=True)  # , save_path="complexAnim.gif")

    ###

    # scenario_save_path = "scenarios/Complex_Solution.xml"
    # fw = CommonRoadFileWriter(scenario, planning_problem_set, "Craig Innes", "University of Edinburgh")
    # fw.write_to_file(scenario_save_path, OverwriteExistingFile.ALWAYS)


if __name__ == "__main__":
    run()
