import copy
import json
import argparse
import os
import time
from dataclasses import dataclass
from datetime import datetime
from os.path import join

import torch
import random
from typing import Callable, List, Any, Tuple

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.file_writer import CommonRoadFileWriter
from commonroad.common.writer.file_writer_interface import OverwriteExistingFile
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import InitialState

from CarMPC import kalman_receding_horizon, pem_observation_batch, initialize_rec_hor_stat
from PyroGPClassification import load_gp_classifier
from PyroGPRegression import load_gp_reg
from ScenarioImportanceSampling import convert_PEM_traj, dets_and_noise_from_stats_new
from TaskConfig import TaskConfig, CostWeights
from anim_utils import animate_with_predictions
from immFilter import c_vel_long_model, c_acc_long_model, lat_model
from monitorScenario import InterstateRulesConfig, gen_interstate_rules
from onlineMonitor import gen_hor_map, init_wmap, update_work_list, update_work_list_pure
from stl import STLExp, G, GEQ0, stl_rob
from trafficRules import no_unnecessary_braking_rule
from utils import RecHorStat, mpc_result_to_dyn_obj, EnhancedJSONEncoder


def first_pred(xs: List[Any], pred: Callable[[Any], bool]) -> int:
    for i, x in enumerate(xs):
        if pred(x):
            return i
    raise ValueError("No element in list satisfies given predicate")


def level_partition(xs: np.ndarray, level: float) -> Tuple[np.ndarray, np.ndarray]:
    idxs = np.arange(len(xs))
    val_mask = xs < level
    saved_idxs, discard_idxs = idxs[val_mask], idxs[~val_mask]
    assert len(saved_idxs) + len(discard_idxs) == len(idxs)
    return saved_idxs, discard_idxs

def level_partition_strict(xs: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray]:
    sort_id_vs = sorted(enumerate(xs), key=lambda kv: kv[1], reverse=True)
    sort_ids = np.array([i for i, x in sort_id_vs])
    discard_idxs, saved_idxs = sort_ids[:K], sort_ids[K:]

    assert len(discard_idxs) == K
    assert len(saved_idxs) == len(xs) - K

    return saved_idxs, discard_idxs



@dataclass(frozen=True)
class AMSResult:
    failure_prob: float
    level_vals: List[int]
    discards_per_level: List[int]
    rob_vals_per_level: np.ndarray

    # samples per level, discards per level, robustness values per level (Levels_Reps X N)


def adaptive_multi_split(start_state, sim_func: Callable, spec: STLExp, sim_T: int, sample_size: int, num_discard: int,
                         final_level: float) -> Tuple[List, AMSResult]:
    stage_trajs = []
    # stage_rob_hists = []
    stage_wl_maps_hist = []
    h_map = gen_hor_map(spec)

    for i in range(sample_size):
        print(f"Initial Sim: {i}")
        # traj = [start_state]
        # update_work_list(wl_map, h_map, spec, start_state, 0)
        # wmh.append(wl_map)

        wmh = []
        wl_map = init_wmap(spec)
        ego_states, obs_state_dicts, obs_ests = sim_func(0, sim_T, start_state, None)
        for t in range(len(ego_states)):
            wl_map = update_work_list_pure(wl_map, h_map, spec, {100: ego_states[t]} | obs_state_dicts[t], t)
            wmh.append(wl_map)

        assert len(wmh) == sim_T
        assert len(wmh[-1][spec].vs) == sim_T
        assert len(ego_states) == sim_T

        stage_wl_maps_hist.append(wmh)
        stage_trajs.append((ego_states, obs_ests))

    # Lower is better (as that means negative robustness)
    # Each round, we want to dicard <num_discard> of the *highest* robustness ones
    final_rob_vals = np.array([wmh[-1][spec].vs[0] for wmh in stage_wl_maps_hist])
    current_level = sorted(final_rob_vals, reverse=True)[num_discard]
    levels = [current_level]
    rob_vals_per_level = [final_rob_vals]

    stage_discards = []
    while current_level > final_level:
        saved_idxs, discard_idxs = level_partition(final_rob_vals, current_level)
        if len(saved_idxs) == 0:
            print(f"Extinction at level: {current_level}")
            break
        # saved_idxs, discard_idxs = level_partition_strict(final_rob_vals, num_discard)
        stage_discards.append(len(discard_idxs))
        print(f"Current Level: {current_level}, Final Level: {final_level}, Discards: {len(discard_idxs)}")

        resampled_trajs = []
        resampled_wl_maps = []
        for d_idx in discard_idxs:
            # Choose random index in I_m
            clone_idx = random.choice(saved_idxs)

            # Clone corresponding trajectory until first time enters L
            cloned_egos, cloned_ests = copy.deepcopy(stage_trajs[clone_idx])
            cloned_maps = copy.deepcopy(stage_wl_maps_hist[clone_idx])

            level_entry = first_pred(cloned_maps, lambda x: x[spec].vs[0] < current_level)
            # level_entry = first_pred(cloned_maps, lambda x: x[spec].vs[0] <= current_level) # Note - this is for the fixed discards regime to work

            # current_resample_traj = cloned_traj[level_entry]
            # current_wl_map = cloned_maps[level_entry]

            reentry_ego, reentry_est = cloned_egos[level_entry], cloned_ests[level_entry]
            if level_entry < sim_T - 1:
                # sim_func(0, sim_T, start_state, None)
                resamp_ego, resamp_obs_sd, resamp_est = sim_func(level_entry, sim_T - level_entry, reentry_ego,
                                                                 reentry_est)
                assert len(resamp_ego) == sim_T - level_entry
                for t in range(1, len(resamp_ego)):
                    # cloned_traj[level_entry + t] = resampled_tail[t]
                    cloned_maps[level_entry + t] = update_work_list_pure(cloned_maps[level_entry + t - 1], h_map, spec,
                                                                         {100: resamp_ego[t]} | resamp_obs_sd[t],
                                                                         level_entry + t)

                # cloned_traj[level_entry + 1:] = resampled_tail[1:]
                cloned_egos[level_entry + 1:] = resamp_ego[1:]
                cloned_ests[level_entry + 1:] = resamp_est[1:]

            resampled_trajs.append((cloned_egos, cloned_ests))
            resampled_wl_maps.append(cloned_maps)
            assert len(cloned_maps) == sim_T
            assert len(cloned_maps[-1][spec].vs) == sim_T
            assert len(cloned_egos) == sim_T
            assert len(cloned_ests) == sim_T

        stage_trajs = [stage_trajs[i] for i in saved_idxs] + resampled_trajs
        stage_wl_maps_hist = [stage_wl_maps_hist[i] for i in saved_idxs] + resampled_wl_maps

        final_rob_vals = np.array([wmh[-1][spec].vs[0] for wmh in stage_wl_maps_hist])
        rob_vals_per_level.append(final_rob_vals)
        current_level = sorted(final_rob_vals, reverse=True)[num_discard]
        levels.append(current_level)

    failure_probability = np.prod([(sample_size - K_m) / sample_size for K_m in stage_discards]) * (
            1 / sample_size) * len(final_rob_vals[final_rob_vals <= final_level])

    return stage_trajs, AMSResult(failure_probability, levels, stage_discards,
                                  rob_vals_per_level=np.array(rob_vals_per_level))


def simple_sim(start_state: int, T: int) -> List[int]:
    traj = [start_state]
    for t in range(1, T):
        r = random.random()
        if r < 0.8:
            traj.append(traj[-1])
        else:
            traj.append(traj[-1] - 1)
    return traj


def kal_run(sims_per_stage: int, discard_prop: float, rule_name: str, exp_name: str, save_root: str):
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
    # sample_size = 1000
    # num_discard = int(np.ceil(0.1 * sample_size))
    final_level = 0

    # start_state = InitialState(position=np.array([0.0 + task_config.car_length / 2.0, ego_lane_centres[0]]),
    #                            velocity=task_config.v_max - 15,
    #                            orientation=0, acceleration=0.0, time_step=0)
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
    num_discard = int(discard_prop * sims_per_stage)

    with open("config/interstate_rule_config.json", 'r') as f:
        irc = InterstateRulesConfig(**json.load(f))

    results_folder = join(save_root,
                          f"results/{exp_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_N{sims_per_stage}_K{num_discard}_L{final_level}")
    ego_car = DynamicObstacle(100, ObstacleType.CAR,
                              Rectangle(width=task_config.car_width, length=task_config.car_length), start_state, None)

    rules = gen_interstate_rules(ego_car, scenario.dynamic_obstacles, all_lane_centres, lane_widths, ego_lane_centres,
                                 all_lane_centres[0:1], irc)

    spec = rules[rule_name]

    print("Rule: ", rule_name)

    final_stage_trajs, ams_results = adaptive_multi_split(start_state, sim_func, spec, sim_T, sims_per_stage,
                                                          num_discard, final_level)
    print("Failure Probability:", ams_results.failure_prob)

    rule_folder = os.path.join(results_folder, rule_name)
    scenarios_folder = os.path.join(rule_folder, "final_scenarios")
    stats_folder = os.path.join(rule_folder, "stats")
    os.makedirs(rule_folder, exist_ok=True)
    os.makedirs(scenarios_folder, exist_ok=True)
    os.makedirs(stats_folder, exist_ok=True)

    np.savetxt(os.path.join(stats_folder, "levels.txt"), ams_results.level_vals)
    np.savetxt(os.path.join(stats_folder, "discards.txt"), ams_results.discards_per_level)
    np.savetxt(os.path.join(stats_folder, "rob_vals.txt"), ams_results.rob_vals_per_level)
    np.savetxt(os.path.join(stats_folder, "failure_prob.txt"), [ams_results.failure_prob])


    for i, (traj, pred_stat) in enumerate(final_stage_trajs):
        solution_scenario = copy.deepcopy(scenario)
        ego_soln_obj = mpc_result_to_dyn_obj(100, traj, task_config.car_width, task_config.car_length)
        solution_scenario.add_objects(ego_soln_obj)

        scenario_save_path = os.path.join(scenarios_folder, f"soln_{i}.xml")
        pred_save_path = os.path.join(scenarios_folder, f"predStats_{i}.json")

        fw = CommonRoadFileWriter(solution_scenario, planning_problem_set, "Craig Innes", "University of Edinburgh")
        fw.write_to_file(scenario_save_path, OverwriteExistingFile.ALWAYS)

        with open(pred_save_path, 'w') as f:
            json.dump(pred_stat, f, cls=EnhancedJSONEncoder)

        pem_states = convert_PEM_traj(sim_T, 100, solution_scenario, norm_mus, norm_stds)[1:]
        pem_dets, pem_long_noises, pem_lat_noises = dets_and_noise_from_stats_new(pred_stat)
        torch.save(pem_states, os.path.join(scenarios_folder, f"pem_states-{i}.pt"))
        torch.save(pem_dets, os.path.join(scenarios_folder, f"pem_dets-{i}.pt"))
        torch.save(pem_long_noises, os.path.join(scenarios_folder, f"pem_long_noise-{i}.pt"))
        torch.save(pem_lat_noises, os.path.join(scenarios_folder, f"pem_lat_noise-{i}.pt"))

    # animate_with_predictions(solution_scenario, pred_stat, sim_T, show=True)


def toy_run():
    start_state = 0
    sim_func = simple_sim
    spec = G(GEQ0(lambda x: x), 0, np.inf)
    sim_T = 10
    sample_size = 1000
    num_discard = int(0.10 * sample_size)
    final_level = -9

    fail_probs = []
    for _ in range(1):
        failure_prob = adaptive_multi_split(start_state, sim_func, spec, sim_T, sample_size, num_discard, final_level)
        fail_probs.append(failure_prob)
    print("Mean: ", np.array(fail_probs).mean(), "STD: ", np.array(fail_probs).std())

    print("AMS Failure Prob:", failure_prob)

    # Raw monte carlo version
    N = 1000000
    trajectories = []
    for n in range(N):
        trajectories.append(sim_func(start_state, sim_T))

    rob_vals = np.array([stl_rob(spec, traj, 0) for traj in trajectories])
    failing_vals = rob_vals[rob_vals <= final_level]
    print(len(failing_vals))
    print("Raw MC Prob:", len(failing_vals) / N)


def raw_MC_prob(sim_func: Callable, start_state, spec: STLExp, sim_T: int, N: int, final_level: float,
                scenario: Scenario, task_config) -> float:
    # Raw monte carlo version
    # trajectories = []
    ssds = []
    for n in range(N):
        ego_states, obs_state_dicts, obs_ests = sim_func(0, sim_T, start_state, None)
        print(f"Raw MC Sim: {n}")
        solution_scenario = copy.deepcopy(scenario)
        ego_soln_obj = mpc_result_to_dyn_obj(100, ego_states, task_config.car_width,
                                             task_config.car_length)
        solution_scenario.add_objects(ego_soln_obj)

        solution_state_dict = [solution_scenario.obstacle_states_at_time_step(i) for i in range(len(ego_states))]
        # trajectories.append(ego_states)
        ssds.append(solution_state_dict)

    rob_vals = np.array([stl_rob(spec, ssd, 0) for ssd in ssds])

    failing_vals = rob_vals[rob_vals <= final_level]
    fail_prob = len(failing_vals) / N
    print(len(failing_vals))
    print("Raw MC Prob:", fail_prob)
    return fail_prob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("N", help="Number of Sims Per Split Level", type=int)
    parser.add_argument("discard_prop", help="proportion to discard each stage (range 0-1)", type=float)
    parser.add_argument("rule_name", help="Name of rule to split against", type=str)
    parser.add_argument("exp_name", help="Name of Experiment Run", type=str)
    parser.add_argument("save_root", help="Root folder to save results in", type=str)
    args = parser.parse_args()
    kal_run(args.N, args.discard_prop, args.rule_name, args.exp_name, args.save_root)
