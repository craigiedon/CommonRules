import copy
import random
from typing import Callable, List, Any, Tuple

import numpy as np

from onlineMonitor import gen_hor_map, init_wmap, update_work_list, update_work_list_pure
from stl import STLExp, G, GEQ0, stl_rob


def first_pred(xs: List[Any], pred: Callable[[Any], bool]) -> int:
    for i, x in enumerate(xs):
        if pred(x):
            return i
    raise ValueError("No element in list satisfies given predicate")


def level_partition(xs: np.ndarray, level: float) -> Tuple[np.ndarray, np.ndarray]:
    idxs = np.arange(len(xs))
    val_mask = xs <= level
    saved_idxs, discard_idxs = idxs[val_mask], idxs[~val_mask]
    assert len(saved_idxs) + len(discard_idxs) == len(idxs)
    return saved_idxs, discard_idxs


def adaptive_multi_split(start_state, sim_func: Callable, spec: STLExp, sample_size: int, num_discard: int,
                         final_level: float) -> float:
    stage_trajs = []
    # stage_rob_hists = []
    stage_wl_maps_hist = []
    sim_T = 10
    h_map = gen_hor_map(spec)

    for i in range(sample_size):
        traj = [start_state]
        wmh = []
        wl_map = init_wmap(spec)
        update_work_list(wl_map, h_map, spec, start_state, 0)
        wmh.append(wl_map)

        for t in range(1, sim_T):
            next_state = sim_func(traj[-1])
            traj.append(next_state)
            new_wl_map = update_work_list_pure(wmh[-1], h_map, spec, next_state, t)
            wmh.append(new_wl_map)
        assert len(wmh) == sim_T
        assert len(wmh[-1][spec].vs) == sim_T
        assert len(traj) == sim_T

        stage_wl_maps_hist.append(wmh)
        stage_trajs.append(traj)

    # Lower is better (as that means negative robustness)
    # Each round, we want to dicard <num_discard> of the *highest* robustness ones
    final_rob_vals = np.array([wmh[-1][spec].vs[0] for wmh in stage_wl_maps_hist])
    current_level = sorted(final_rob_vals, reverse=True)[num_discard]

    stage_discards = []
    while current_level > final_level:
        saved_idxs, discard_idxs = level_partition(final_rob_vals, current_level)
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

            level_entry = first_pred(cloned_maps, lambda x: x[spec].vs[0] <= current_level)

            current_resample_traj = cloned_traj[level_entry]
            current_wl_map = cloned_maps[level_entry]
            for t in range(level_entry + 1, len(stage_trajs[clone_idx])):
                old_resample_traj = current_resample_traj
                current_resample_traj = sim_func(old_resample_traj)
                current_wl_map = update_work_list_pure(current_wl_map, h_map, spec, current_resample_traj, t)
                cloned_traj[t] = current_resample_traj
                cloned_maps[t] = current_wl_map

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


def simple_sim(state: int):
    r = random.random()
    if r < 0.8:
        return state
    else:
        return state - 1


def run():
    start_state = 0
    sim_func = simple_sim
    spec= G(GEQ0(lambda x: x), 0, np.inf)
    sample_size = 1000
    num_discard = int(0.9 * sample_size)
    final_level = -8
    failure_prob = adaptive_multi_split(start_state, sim_func, spec, sample_size, num_discard, final_level)
    print("AMS Failure Prob:", failure_prob)

    # Raw monte carlo version
    N = 1000000
    sim_T = 10
    trajectories = []
    for n in range(N):
        traj = [start_state]
        for t in range(1, sim_T):
            traj.append(sim_func(traj[-1]))
        trajectories.append(traj)
    rob_vals = np.array([stl_rob(spec, traj, 0) for traj in trajectories])
    failing_vals = rob_vals[rob_vals <= final_level]
    print(len(failing_vals))
    print("Raw MC Prob:", len(failing_vals) / N)



if __name__ == "__main__":
    run()
