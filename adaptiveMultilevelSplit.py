import copy
import random

import numpy as np


def adaptive_multi_split(start_state, sim_func, spec_monitor, sample_size, num_discard, final_level) -> float:
    stage_trajs = []
    stage_rob_hists = []
    sim_T = 10

    # TODO: Look up online_run in online monitor to see
    # TODO: 1. What setup needs to be done in terms of horizon structures, initial worklists etc
    # TODO: 2. What updates need to be done in each timestep, and what information needs to be saved/cloned in each part for this to work in a "stage to stage" way within the while also
    for i in range(sample_size):
        traj = [start_state]
        rob_hist = [spec_monitor(traj, worklist)]
        for t in range(sim_T):
            traj.append(sim_func(traj[-1]))
            rob_val = spec_monitor(traj, worklist)
            rob_hist.append(rob_val)

        stage_trajs.append(traj)
        rob_hist.append(rob_hist)

    # Lower is better (as that means negative robustness)
    # Each round, we want to dicard <num_discard> of the *highest* robustness ones
    final_rob_vals = np.array(stage_rob_hists)[:, -1]
    sorted_vals = sorted(enumerate(final_rob_vals), key=lambda x: x[1], reverse=True)
    current_level = sorted_vals[num_discard]

    stage_discards = []
    while current_level > final_level:
        # Discard all trajs for which M_i <= L
        # Let K_m be the number of such trajectories (K_m >= K)
        # Get the final index where
        idxs = np.arange(sample_size)
        val_mask = final_rob_vals <= current_level
        saved_idxs, discard_idxs = idxs[val_mask], idxs[~val_mask]
        stage_discards.append(len(discard_idxs))
        assert len(saved_idxs) + len(discard_idxs) == idxs

        resampled_trajs = []
        resampled_rob_hists = []
        for d_idx in discard_idxs:
            # Choose random index in I_m
            clone_idx = random.choice(saved_idxs)

            # Clone corresponding trajectory until first time enters L
            cloned_traj = copy.deepcopy(stage_trajs[clone_idx])
            cloned_rob_hist = copy.deepcopy(stage_rob_hists[clone_idx])

            level_entry = 0
            while cloned_rob_hist[level_entry] > current_level:
                level_entry += 1

            current_resample_traj = cloned_traj[level_entry]
            current_rob_hist = cloned_rob_hist[level_entry]
            for t in range(level_entry + 1, len(stage_trajs[clone_idx])):
                old_resample_traj = current_resample_traj
                old_rob_hist = current_rob_hist
                current_resample_traj, current_rob_hist = sim_func(old_resample_traj, spec_monitor)
                cloned_traj[t] = current_resample_traj
                cloned_rob_hist[t] = current_rob_hist

            resampled_trajs.append(cloned_traj)
            resampled_rob_hists.append(cloned_rob_hist)

        stage_trajs = [stage_trajs[i] for i in saved_idxs] + resampled_trajs
        stage_rob_hists = [stage_rob_hists[i] for i in saved_idxs] + resampled_rob_hists

        # Set M_i = rob_val[i]
        final_rob_vals = np.array(stage_rob_hists)[:, -1]
        sorted_vals = sorted(enumerate(final_rob_vals), key=lambda x: x[1], reverse=True)
        current_level = sorted_vals[num_discard]

    failure_probability = np.prod([(sample_size - K_m) / sample_size for K_m in stage_discards]) * (
                1 / sample_size) * len(final_rob_vals[final_rob_vals <= final_level])
    return failure_probability


def simple_sim(state: int):
    r = random.random()
    if r < 0.8:
        return state + 1
    else:
        return state - 1


def run():
    start_state = 0
    sim_func = simple_sim
    spec_monitor = None
    sample_size = 100
    num_discard = 50
    final_level = -20
    failure_prob = adaptive_multi_split(start_state, sim_func, spec_monitor, sample_size, num_discard, final_level)
    print("Failure Prob:", failure_prob)


if __name__ == "__main__":
    run()
