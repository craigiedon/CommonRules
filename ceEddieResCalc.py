import os
import glob
from collections import defaultdict
import scipy.stats as scp

import numpy as np
from pathlib import Path

from matplotlib import pyplot as plt
from scipy.special import logsumexp


def run():
    rule_names = ["rg_1", "rg_2", "rg_4", "ri_2"]
    # Ks = [2, 12, 25, 125, 225]
    results_root = "/media/craig/Extreme Pro/amsResults/ceRG"

    rv_log_fail_probs = defaultdict(list)
    rv_levels = defaultdict(list)
    rv_stage_fails = {}
    for rule in rule_names:
        for path in Path(results_root).rglob(f"*{rule}*/log_failure_prob.txt"):
            rv_log_fail_probs[rule].append(np.loadtxt(path.as_posix()))
        for path in Path(results_root).rglob(f"*{rule}*/levels.txt"):
            rv_levels[rule].append(np.loadtxt(path.as_posix()))

        stage_vs = []
        for stage in range(10):
            vs = []
            for path in Path(results_root).rglob(f"*{rule}*/s-{stage}/{rule}-vals.txt"):
                rvs = np.loadtxt(path.as_posix())
                fail_rvs = len(rvs[rvs <= 0])
                vs.append(fail_rvs)
            if len(vs) > 0:
                stage_vs.append(np.mean(vs))
            else:
                stage_vs.append(-1)
        rv_stage_fails[rule] = stage_vs



    print("Done")

    # Chart the average "levels" of each CE
    for rule, levels in rv_levels.items():
        avg_d_levels = np.mean(levels, axis=0)
        plt.plot(avg_d_levels, label=rule)
    plt.legend()
    plt.show()

    # Chart the log failure probabilities as levels proceed
    for rule, lfps in rv_log_fail_probs.items():
        # avg_d_levels = np.mean(levels, axis=0)
        avg_lfps = logsumexp(lfps, axis=0)
        plt.plot(avg_lfps, label=rule)
    plt.legend()
    plt.show()

    for rule, stage_fails in rv_stage_fails.items():
        plt.plot(stage_fails, label=rule)
    plt.legend()
    plt.show()

    # Chart the...

    # for rule_name, fps in rv_fail_probs.items():
    #     nfps = np.array(fps)
    #     nonzero_fps = nfps[nfps > 0]
    #     print(f"{rule_name}: {np.mean(nonzero_fps):.2e} (+- {np.std(nonzero_fps):.2e}) \t {nonzero_fps.size}")
        # print(f"{rule_name}: {np.mean(fps):.2e} (+- {np.std(fps):.2e})\t {len(fps)}" )
        # print(np.mean(fps))
        # print(fps)
        # print(rv_fail_probs)


if __name__ == "__main__":
    run()
