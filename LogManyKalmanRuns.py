import time
import torch

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.geometry.shape import Rectangle
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.state import InitialState
from commonroad.scenario.trajectory import Trajectory
from tqdm import tqdm

from CarMPC import TaskConfig, CostWeights, kalman_receding_horizon, pem_observation_batch
from PyroGPClassification import load_gp_classifier
from PyroGPRegression import load_gp_reg
from immFilter import c_vel_long_model, c_acc_long_model, lat_model


def run():
    file_path = "scenarios/Complex.xml"
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open()
    all_lane_centres = [scenario.lanelet_network.find_lanelet_by_id(l).center_vertices[0][1] for l in [3, 6, 10]]
    ego_lane_centres = [scenario.lanelet_network.find_lanelet_by_id(l).center_vertices[0][1] for l in [6, 10]]

    goal_state = planning_problem_set.find_planning_problem_by_id(1).goal.state_list[0].position.center
    end_time = 5.0
    task_config = TaskConfig(dt=0.1,
                             x_goal=goal_state[0],
                             y_goal=goal_state[1],
                             y_bounds=(0.0, 7.0),
                             car_width=4.298,
                             car_height=1.674,
                             v_goal=31.29,  # == 70mph
                             v_max=45.8,
                             acc_max=11.5,
                             ang_vel_max=0.5,
                             lanes=ego_lane_centres,
                             lane_targets=[],
                             # collision_field_slope=1e-5)
                             collision_field_slope=0.90)

    start_state = InitialState(position=np.array([10.0, ego_lane_centres[0]]), velocity=task_config.v_goal * 0.2,
                               orientation=0, time_step=0)

    cws = CostWeights(x_prog=0.01, y_prog=0.1, acc=0.1, ang_v=10, jerk=1, v_track=2, lane_align=1, road_align=50,
                      collision_pot=500,
                      faster_left=1.0, braking=10)

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

    # TODO Run 100 Receding Horizon Kalmans,
    # TODO: Log Results,
    # TODO: and Log STL Monitors on them (Also...time it maybe? We can see if this sampling thing is in any way actually viable...)
    for i in tqdm(range(5)):
        dn_state_list, prediction_stats = kalman_receding_horizon(end_time, 2.0, start_state, scenario, task_config,
                                                                  long_models, lat_models, observation_func,
                                                                  cws)


if __name__ == "__main__":
    run()