import copy
import json
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

from CarMPC import kalman_receding_horizon, pem_observation_batch
from TaskConfig import TaskConfig, CostWeights
from utils import mpc_result_to_dyn_obj
from PyroGPClassification import load_gp_classifier
from PyroGPRegression import load_gp_reg
from immFilter import c_vel_long_model, c_acc_long_model, lat_model
from monitorScenario import InterstateRulesConfig, gen_interstate_rules
from stl import stl_rob
from anim_utils import animate_with_predictions, animate_scenario


def run():
    file_path = "scenarios/Complex.xml"
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open()
    all_lane_centres = [scenario.lanelet_network.find_lanelet_by_id(l).center_vertices[0][1] for l in [3, 6, 10]]
    ego_lane_centres = [scenario.lanelet_network.find_lanelet_by_id(l).center_vertices[0][1] for l in [6, 10]]
    lane_widths = np.abs((ego_lane_centres[0] - ego_lane_centres[1]) / 2.0)

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
                             collision_field_slope=1.0)

    start_state = InitialState(position=np.array([10.0, ego_lane_centres[0]]), velocity=task_config.v_goal * 0.3,
                               orientation=0, acceleration=0.0, time_step=0)

    with open("config/cost_weights.json", 'r') as f:
        cws = CostWeights(**json.load(f))

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

    def observation_func(obs, ego_state, t, tlong, tlat, vs):
        return pem_observation_batch(obs, ego_state, t, tlong, tlat,
                                     det_pem, reg_pem, norm_mus,
                                     norm_stds, vs)

    with open("config/interstate_rule_config.json", 'r') as f:
        irc = InterstateRulesConfig(**json.load(f))

    # TODO Run 100 Receding Horizon Kalmans,
    # TODO: Log Results,
    # TODO: and Log STL Monitors on them (Also...time it maybe? We can see if this sampling thing is in any way actually viable...)
    for i in range(20):
    # for i in tqdm(range(5)):
        kalman_start = time.time()
        dn_state_list, prediction_stats = kalman_receding_horizon(end_time, 1.5, start_state, scenario, task_config,
                                                                  long_models, lat_models, observation_func,
                                                                  cws)
        print(f"Took {time.time() - kalman_start}s")
        solution_scenario = copy.deepcopy(scenario)
        ego_soln_obj = mpc_result_to_dyn_obj(100, dn_state_list, task_config.car_width,
                                             task_config.car_length)
        solution_scenario.add_objects(ego_soln_obj)

        solution_state_dict = [solution_scenario.obstacle_states_at_time_step(i) for i in range(len(dn_state_list))]

        rules = gen_interstate_rules(100, solution_scenario, all_lane_centres, lane_widths, ego_lane_centres,
                                     all_lane_centres[0:1], irc)

        # rule_name = "rg_2"
        # rob_val = stl_rob(rules["rg_2"], solution_state_dict, 0)
        # print(f"{rule_name}:\t {rob_val}")

        rob_vals = []
        for rule_name, rule in rules.items():
            rob_val = stl_rob(rule, solution_state_dict, 0)
            rob_vals.append(rob_val)
            print(f"{rule_name}:\t {rob_val}")
        # if np.any(np.array(rob_vals) < 0):
        animate_with_predictions(solution_scenario, prediction_stats, int(end_time / task_config.dt), show=True)


        ## Then...calc and print out the results here!


if __name__ == "__main__":
    run()
