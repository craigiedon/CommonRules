# TODO: Specify an N via args? Use the args parse stuff
import argparse
import copy
import json
import os
from datetime import datetime
from os.path import join

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.file_writer import CommonRoadFileWriter
from commonroad.common.writer.file_writer_interface import OverwriteExistingFile
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.state import InitialState

from CarMPC import pem_observation_batch, initialize_rec_hor_stat, kalman_receding_horizon
from PyroGPClassification import load_gp_classifier
from PyroGPRegression import load_gp_reg
from ScenarioImportanceSampling import convert_PEM_traj, dets_and_noise_from_stats, dets_and_noise_from_stats_new
from TaskConfig import TaskConfig, CostWeights
from adaptiveMultilevelSplit import raw_MC_prob
from immFilter import c_vel_long_model, c_acc_long_model, lat_model
from monitorScenario import InterstateRulesConfig, gen_interstate_rules
from stl import stl_rob
from utils import mpc_result_to_dyn_obj, RecHorStat, EnhancedJSONEncoder
import torch


def run(num_sims: int, exp_name: str, save_root: str):
    print(f"Num Sims: {num_sims}, Exp Name: {exp_name}")

    file_path = "scenarios/Complex.xml"
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

    with open("config/task_config.json") as f:
        task_config = TaskConfig(**json.load(f))

    all_lane_centres = [scenario.lanelet_network.find_lanelet_by_id(l).center_vertices[0][1] for l in [3, 6, 10]]
    ego_lane_centres = [scenario.lanelet_network.find_lanelet_by_id(l).center_vertices[0][1] for l in [6, 10]]
    lane_widths = np.abs((ego_lane_centres[0] - ego_lane_centres[1]) / 2.0)

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

    with open("config/cost_weights.json", 'r') as f:
        cws = CostWeights(**json.load(f))

    sim_T = 40

    start_state = InitialState(position=np.array([15.0 + task_config.car_length / 2.0, ego_lane_centres[0]]),
                               velocity=task_config.v_max - 15,
                               orientation=0, acceleration=0.0, time_step=0)
    start_est: RecHorStat = initialize_rec_hor_stat(scenario.obstacles, long_models, lat_models, 0, 0 + sim_T, 20)

    def try_kalman_rh(start_step, sim_steps, st_st, st_es):
        max_retries = 100
        for i in range(max_retries):
            try:
                return kalman_receding_horizon(start_step, sim_steps, 20, st_st, st_es, scenario, task_config,
                                               long_models, lat_models, observation_func, cws)
            except Exception as e:
                print("Exception", e)
                print(f"Failed to solve (Try attempt: {i} , retrying")

        raise ValueError(f"Max Retries {max_retries} attempted and failed!")

    sim_func = try_kalman_rh

    with open("config/interstate_rule_config.json", 'r') as f:
        irc = InterstateRulesConfig(**json.load(f))

    results_folder = join(save_root, f"results/{exp_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_N{num_sims}")
    os.makedirs(results_folder, exist_ok=True)
    ego_car = DynamicObstacle(100, ObstacleType.CAR,
                              Rectangle(width=task_config.car_width, length=task_config.car_length), start_state, None)

    rules = gen_interstate_rules(ego_car, scenario.dynamic_obstacles, all_lane_centres, lane_widths, ego_lane_centres, all_lane_centres[0:1], irc)

    # raw_mc_fail_prob = raw_MC_prob(sim_func, start_state, spec, sim_T, num_sims, 0, scenario, task_config)

    ssds = []
    for n in range(num_sims):
        ego_states, obs_state_dicts, obs_ests = sim_func(0, sim_T, start_state, None)
        print(f"Raw MC Sim: {n}")
        solution_scenario = copy.deepcopy(scenario)
        ego_soln_obj = mpc_result_to_dyn_obj(100, ego_states, task_config.car_width,
                                             task_config.car_length)
        solution_scenario.add_objects(ego_soln_obj)

        # Note this conversion function for using alongside robustness calculation
        solution_state_dict = [solution_scenario.obstacle_states_at_time_step(i) for i in range(len(ego_states))]
        fw = CommonRoadFileWriter(solution_scenario, planning_problem_set, "Craig Innes", "University of Edinburgh")
        fw.write_to_file(os.path.join(results_folder, f"solution-{n}.xml"), OverwriteExistingFile.ALWAYS)
        with open(os.path.join(results_folder, f"noise_estimates-{n}.json"), 'w') as f:
            json.dump(obs_ests, f, cls=EnhancedJSONEncoder)

        pem_states = convert_PEM_traj(sim_T, 100, solution_scenario, norm_mus, norm_stds)[1:]
        pem_dets, pem_long_noises, pem_lat_noises = dets_and_noise_from_stats_new(obs_ests)
        torch.save(pem_states, os.path.join(results_folder, f"pem_states-{n}.pt"))
        torch.save(pem_dets, os.path.join(results_folder, f"pem_dets-{n}.pt"))
        torch.save(pem_long_noises, os.path.join(results_folder, f"pem_long_noise-{n}.pt"))
        torch.save(pem_lat_noises, os.path.join(results_folder, f"pem_lat_noise-{n}.pt"))

        ssds.append(solution_state_dict)

    for rule_name, spec in rules.items():
        rob_vals = np.array([stl_rob(spec, ssd, 0) for ssd in ssds])
        np.savetxt(os.path.join(results_folder, f"{rule_name}-vals.txt"), rob_vals)

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("N", help="Number of Simulations to Run", type=int)
    parser.add_argument("exp_name", help="Name of Experiment Run", type=str)
    parser.add_argument("save_root", help="Root folder to save results in", type=str)
    args = parser.parse_args()
    run(args.N, args.exp_name, args.save_root)
