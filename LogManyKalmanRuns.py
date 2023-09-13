import copy
import datetime
import json
import os
import pickle
import time
import torch

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.file_writer import CommonRoadFileWriter
from commonroad.common.writer.file_writer_interface import OverwriteExistingFile
from commonroad.geometry.shape import Rectangle
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.state import InitialState
from commonroad.scenario.trajectory import Trajectory
from matplotlib import pyplot as plt
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
    end_time = 4.0
    with open("config/task_config.json") as f:
        task_config = TaskConfig(**json.load(f))

    start_state = InitialState(position=np.array([0.0 + task_config.car_length / 2.0, ego_lane_centres[0]]), velocity=task_config.v_max - 15,
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

    results_folder_path = f"results/kal_mpc_res_{datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')}"
    os.mkdir(results_folder_path)
    rep_rob_vals = []

    for i in tqdm(range(100000)):
    # for i in tqdm(range(5)):

        try:
            # kalman_start = time.time()
            # prediction_steps = int(round(horizon_length / task_config.dt)) - 1
            sim_steps = int(round(end_time / task_config.dt))
            start_ests = None
            dn_state_list, prediction_stats = kalman_receding_horizon(0, sim_steps, int(round(2.0 / task_config.dt)) - 1,
                                                                      start_state, start_ests, scenario, task_config,
                                                                      long_models, lat_models, observation_func, cws)
            # print(f"Took {time.time() - kalman_start}s")
        except Exception as e:
            print("Exception", e)
            print("Failed to solve, skipping")
            continue

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
            if rule_name == "rg_4" and rob_val < 0.0:
                print("Got an rg_4 failure!: ", rob_val)
            # print(f"{rule_name}:\t {rob_val}")

        rep_rob_vals.append(rob_vals)

        np.savetxt(os.path.join(results_folder_path, "rule_rob_vals.txt"), rep_rob_vals, fmt="%.4f")
        with open(os.path.join(results_folder_path, f"prediction_stats_{i}.pkl"), 'wb') as f:
            pickle.dump(prediction_stats, f)

        scenario_save_path = os.path.join(results_folder_path, f"kal_mpc_{i}.xml")
        fw = CommonRoadFileWriter(solution_scenario, planning_problem_set, "Craig Innes", "University of Edinburgh")
        fw.write_to_file(scenario_save_path, OverwriteExistingFile.ALWAYS)

        # if np.any(np.array(rob_vals) < 0):
        # animate_with_predictions(solution_scenario, prediction_stats, int(end_time / task_config.dt), show=True)
        # plt.plot([s.acceleration for s in dn_state_list])
        # plt.show()

        ## Then...calc and print out the results here!


if __name__ == "__main__":
    run()
