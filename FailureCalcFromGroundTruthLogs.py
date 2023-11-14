import json
import os

import numpy as np
from glob import glob

import torch
from os.path import join
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.state import InitialState

from PyroGPClassification import load_gp_classifier
from PyroGPRegression import load_gp_reg
from ScenarioImportanceSampling import log_probs_scenario_traj
from TaskConfig import TaskConfig
from monitorScenario import gen_interstate_rules, InterstateRulesConfig
from stl import stl_rob


def run():
    file_path = "scenarios/Complex.xml"
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

    with open("config/task_config.json") as f:
        task_config = TaskConfig(**json.load(f))

    with open("config/interstate_rule_config.json", 'r') as f:
        irc = InterstateRulesConfig(**json.load(f))
        
    det_pem = load_gp_classifier("models/nuscenes/vsgp_class", True)
    det_pem.eval()
    reg_pem = load_gp_reg("models/nuscenes/sgp_reg", True)
    reg_pem.eval()
    norm_mus = torch.load("data/nuscenes/inp_mus.pt")
    norm_stds = torch.load("data/nuscenes/inp_stds.pt")

    all_lane_centres = [scenario.lanelet_network.find_lanelet_by_id(l).center_vertices[0][1] for l in [3, 6, 10]]
    ego_lane_centres = [scenario.lanelet_network.find_lanelet_by_id(l).center_vertices[0][1] for l in [6, 10]]
    lane_widths = np.abs((ego_lane_centres[0] - ego_lane_centres[1]) / 2.0)

    start_state = InitialState(position=np.array([15.0 + task_config.car_length / 2.0, ego_lane_centres[0]]),
                               velocity=task_config.v_max - 15,
                               orientation=0, acceleration=0.0, time_step=0)

    ego_car = DynamicObstacle(100, ObstacleType.CAR,
                              Rectangle(width=task_config.car_width, length=task_config.car_length), start_state, None)

    rules = gen_interstate_rules(ego_car, scenario.dynamic_obstacles, all_lane_centres, lane_widths, ego_lane_centres,
                             all_lane_centres[0:1], irc)

    results_folder ="results/gt-mini-2023-11-14-12-25-53_N10"
    # print(os.listdir(results_folder))
    solution_files = sorted(glob("solution-*.xml", root_dir=results_folder))
    pem_states = [torch.load(join(results_folder, ps)) for ps in sorted(glob("pem_states-*.pt", root_dir=results_folder))]
    pem_dets = [torch.load(join(results_folder, ps)) for ps in sorted(glob("pem_dets-*.pt", root_dir=results_folder))]
    pem_long_ns = [torch.load(join(results_folder, ps)) for ps in sorted(glob("pem_long_noise-*.pt", root_dir=results_folder))]
    pem_lat_ns = [torch.load(join(results_folder, ps)) for ps in sorted(glob("pem_lat_noise-*.pt", root_dir=results_folder))]
    solution_scenarios = [CommonRoadFileReader(join(results_folder, sf)).open()[0] for sf in solution_files]
    ssds = []
    time_steps = 40
    for ss in solution_scenarios:
        ssd = [ss.obstacle_states_at_time_step(i) for i in range(time_steps)]
        ssds.append(ssd)

    # Compute rule robustnesses from SSDS (this is the slow way of doing it! See next bit for the cached)
    for rule_name, spec in rules.items():
        print(rule_name)
        rob_vals = np.array([stl_rob(spec, ssd, 0) for ssd in ssds])
        print(rob_vals)
        failing_vals = rob_vals[rob_vals <= 0]
        fail_prob = len(failing_vals) / len(ssds)
        print(len(failing_vals))
        print("Raw MC Prob:", fail_prob)

    for rule_name, _ in rules.items():
        print(rule_name)
        rvs = np.loadtxt(join(results_folder, f"{rule_name}-vals.txt"))
        print(rvs)
        failing_vals = rob_vals[rob_vals <= 0]
        fail_prob = len(failing_vals) / len(ssds)
        print(len(failing_vals))
        print("Raw MC Prob:", fail_prob)

    for p_state, p_det, p_long_n, p_lat_n in zip(pem_states, pem_dets, pem_long_ns, pem_lat_ns):
        pem_log_likelihood = log_probs_scenario_traj(p_state, p_det, p_long_n, p_lat_n, det_pem, reg_pem)
        
        print(pem_log_likelihood)


if __name__ == "__main__":
    run()