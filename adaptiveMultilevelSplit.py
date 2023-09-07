import random

import numpy as np


def adaptive_multi_split(start_state, sim_func, spec_monitor, sample_size, num_discard, final_level) -> float:
    m = 0
    trajectories = []
    rob_hists = []
    for i in range(sample_size):
        traj, rob_hist = sim_func(start_state, spec_monitor)
        trajectories.append(traj)
        rob_hist.append(rob_hist)

    # Lower is better (as that means negative robustness)
    # Each round, we want to dicard <num_discard> of the *highest* robustness ones
    final_rob_vals = np.array(rob_hists)[:, -1]
    sorted_vals = sorted(enumerate(final_rob_vals), key=lambda x: x[1], reverse=True)
    current_level = sorted_vals[num_discard]

    while current_level > final_level:
        m = m + 1
        # Discard all trajs for which M_i <= L
        # Let K_m be the number of such trajectories (K_m >= K)
        # Get the final index where
        idxs = np.arange(sample_size)
        val_mask = final_rob_vals <= current_level
        saved_idxs, discard_idxs = idxs[val_mask], idxs[~val_mask]
        assert len(saved_idxs) + len(discard_idxs) == idxs

        # Define I_m as st of indices of remaining trajectories
        # discard_idxs, saved_idxs =

        for d_idx in discard_idxs:
            # Choose random index in I_m
            clone_idx = random.choice(saved_idxs)

            # Clone corresponding trajectory until first time enters L

            # From that time, simulate the cloned trajectory up to its end time t_i
            # Replace trajectory with index i by new traj
            # Set M_i = rob_val[i]
        sorted_vals = sorted(rob_vals)
        current_level = sorted_vals[num_discard]

    total_iterations = m
    failure_probability = 0
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