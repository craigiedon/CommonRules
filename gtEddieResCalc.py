import os
import glob
from collections import defaultdict
import scipy.stats as scp

import numpy as np
from pathlib import Path


def run():
    rule_names = ["rg_1", "rg_2", "rg_4", "ri_2"]
    results_root = "/media/craig/Extreme Pro/amsResults/groundTruthRG"

    rule_vals = defaultdict(list)
    for rule in rule_names:
        for path in Path(results_root).rglob(f"{rule}-vals.txt"):
            rule_vals[rule].append(np.loadtxt(path.as_posix()))
            # print(path.as_posix())

    # Full Stats
    print("Ground Truths")
    for rule_name, rv_arr in rule_vals.items():
        stacked_rvs = np.stack(rv_arr).reshape(-1)
        fails = stacked_rvs[stacked_rvs <= 0]
        num_fails = fails.size
        total_rvs = stacked_rvs.size
        print(f"{rule_name}: {num_fails / total_rvs:.2e} ({num_fails} / {total_rvs})")

    print("Naive 100")
    for rule_name in rule_vals.keys():
        chunked_stats(rule_vals, rule_name, 100)

    print("Naive 1000")
    for rule_name in rule_vals.keys():
        chunked_stats(rule_vals, rule_name, 1000)


def chunked_stats(rule_vals, rule_name: str, chunked_size: int):
    stacked_rvs = np.stack(rule_vals[rule_name])
    per_chunk = stacked_rvs.reshape(-1, chunked_size)
    fail_props = np.sum(per_chunk <= 0, axis=1) / chunked_size
    # print("Num zeros: ", len(fail_props[fail_props == 0]))
    fail_subset = fail_props[0:5]
    fail_mu = np.mean(fail_subset)
    # fail_sem = scp.sem(fail_subset)
    fail_std = np.std(fail_subset)
    print(f"{rule_name}: {fail_mu:.2e} (+- {fail_std:.2e})")


if __name__ == "__main__":
    run()
