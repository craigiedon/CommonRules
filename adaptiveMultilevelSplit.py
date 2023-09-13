import copy
import json
import torch
import random
from typing import Callable, List, Any, Tuple

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader

from CarMPC import kalman_receding_horizon, pem_observation_batch
from PyroGPClassification import load_gp_classifier
from PyroGPRegression import load_gp_reg
from TaskConfig import TaskConfig, CostWeights
from immFilter import c_vel_long_model, c_acc_long_model, lat_model
from onlineMonitor import gen_hor_map, init_wmap, update_work_list, update_work_list_pure
from stl import STLExp, G, GEQ0, stl_rob


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


def adaptive_multi_split(start_state, sim_func: Callable, spec: STLExp, sim_T: int, sample_size: int, num_discard: int,
                         final_level: float) -> float:
    stage_trajs = []
    # stage_rob_hists = []
    stage_wl_maps_hist = []
    h_map = gen_hor_map(spec)

    for i in range(sample_size):
        # traj = [start_state]
        # update_work_list(wl_map, h_map, spec, start_state, 0)
        # wmh.append(wl_map)

        wmh = []
        wl_map = init_wmap(spec)
        traj = sim_func(start_state, sim_T)
        for t, x in enumerate(traj):
            wl_map = update_work_list_pure(wl_map, h_map, spec, x, t)
            wmh.append(wl_map)

        # for t in range(1, sim_T):
        #     next_state = sim_func(traj[-1])
        #     traj.append(next_state)
        #     new_wl_map = update_work_list_pure(wmh[-1], h_map, spec, next_state, t)
        #     wmh.append(new_wl_map)
        assert len(wmh) == sim_T
        assert len(wmh[-1][spec].vs) == sim_T
        assert len(traj) == sim_T

        stage_wl_maps_hist.append(wmh)
        stage_trajs.append(traj)

    # Lower is better (as that means negative robustness)
    # Each round, we want to dicard <num_discard> of the *highest* robustness ones
    final_rob_vals = np.array([wmh[-1][spec].vs[0] for wmh in stage_wl_maps_hist])
    # shuffle_sorted = sorted(random.sample(list(enumerate(final_rob_vals)), k=len(final_rob_vals)), key=lambda x: x[1], reverse=True)
    # current_level = shuffle_sorted[num_discard][1]
    current_level = sorted(final_rob_vals, reverse=True)[num_discard]

    stage_discards = []
    while current_level > final_level:
        saved_idxs, discard_idxs = level_partition(final_rob_vals, current_level)
        if len(saved_idxs) == 0:
            print(f"Extinction at level: {current_level}")
            break
        # sorted_ids = [i for i, x in shuffle_sorted]
        # discard_idxs, saved_idxs = sorted_ids[:num_discard], sorted_ids[num_discard:]
        # assert len(discard_idxs) == num_discard
        # assert len(saved_idxs) == sample_size - num_discard
        stage_discards.append(len(discard_idxs))
        print(f"Current Level: {current_level}, Final Level: {final_level}, Discards: {len(discard_idxs)}")

        resampled_trajs = []
        resampled_wl_maps = []
        for d_idx in discard_idxs:
            # Choose random index in I_m
            clone_idx = random.choice(saved_idxs)

            # Clone corresponding trajectory until first time enters L
            cloned_traj = copy.deepcopy(stage_trajs[clone_idx])
            cloned_maps = copy.deepcopy(stage_wl_maps_hist[clone_idx])

            level_entry = first_pred(cloned_maps, lambda x: x[spec].vs[0] < current_level)

            # current_resample_traj = cloned_traj[level_entry]
            # current_wl_map = cloned_maps[level_entry]

            resample_traj_entry = cloned_traj[level_entry]
            if level_entry < sim_T - 1:
                resampled_tail = sim_func(resample_traj_entry, sim_T - level_entry)
                assert len(resampled_tail) == sim_T - level_entry
                for t in range(1, len(resampled_tail)):
                    cloned_traj[level_entry + t] = resampled_tail[t]
                    cloned_maps[level_entry + t] = update_work_list_pure(cloned_maps[level_entry + t - 1], h_map, spec, resampled_tail[t], level_entry + t)

                cloned_traj[level_entry + 1:] = resampled_tail[1:]

            resampled_trajs.append(cloned_traj)
            resampled_wl_maps.append(cloned_maps)
            assert len(cloned_maps) == sim_T
            assert len(cloned_maps[-1][spec].vs) == sim_T
            assert len(cloned_traj) == sim_T

        stage_trajs = [stage_trajs[i] for i in saved_idxs] + resampled_trajs
        stage_wl_maps_hist = [stage_wl_maps_hist[i] for i in saved_idxs] + resampled_wl_maps

        final_rob_vals = np.array([wmh[-1][spec].vs[0] for wmh in stage_wl_maps_hist])
        current_level = sorted(final_rob_vals, reverse=True)[num_discard]

    failure_probability = np.prod([(sample_size - K_m) / sample_size for K_m in stage_discards]) * (
            1 / sample_size) * len(final_rob_vals[final_rob_vals <= final_level])
    return failure_probability


def simple_sim(start_state: int, T: int) -> List[int]:
    traj = [start_state]
    for t in range(1, T):
        r = random.random()
        if r < 0.8:
            traj.append(traj[-1])
        else:
            traj.append(traj[-1] - 1)
    return traj


def kal_run():

    file_path = "scenarios/Complex.xml"
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

    with open("config/task_config.json") as f:
        task_config = TaskConfig(**json.load(f))

    all_lane_centres = [scenario.lanelet_network.find_lanelet_by_id(l).center_vertices[0][1] for l in [3, 6, 10]]

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

    # TODO: Find the form of the "estimates" needed to continue, and put it in as an argument?
    # TODO: Assess what still doesn't work from here.
    # TODO: Assess what to print so that you can monitor success...

    start_state = None
    start_est = None

    sim_func = lambda ss, T: kalman_receding_horizon(T, 2.0, ss, scenario, task_config, long_models, lat_models, observation_func, cws)

    failure_prob = adaptive_multi_split(start_state, kalman_receding_horizon)

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


if __name__ == "__main__":
    kal_run()
    # toy_run()
