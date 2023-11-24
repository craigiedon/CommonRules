import os
import glob
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib import rcParams
import scipy.stats as scp

import numpy as np
from pathlib import Path


def run():
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42
    rule_names = ["rg_1", "rg_2", "rg_4", "ri_2"]
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()['color']
    rule_colors = {rn: c for rn, c in zip(rule_names, colors)}
    print(colors)

    # Ks = [1, 5, 10, 50, 90]
    # N = 100
    # results_root = "/media/craig/Extreme Pro/amsResults/ams100RG"

    Ks = [2, 12, 25, 125, 225]
    N = 250
    results_root = "/media/craig/Extreme Pro/amsResults/amsRG"

    rv_fail_probs = {rule_name: defaultdict(list) for rule_name in rule_names}
    rv_levels = {rule_name: defaultdict(list) for rule_name in rule_names}
    for rule in rule_names:
        for k in Ks:
            for path in Path(results_root).rglob(f"*N{N}_K{k}_*/{rule}/stats/failure_prob.txt"):
                rv_fail_probs[rule][k].append(np.loadtxt(path.as_posix()))
            for path in Path(results_root).rglob(f"*N{N}_K{k}_*/{rule}/stats/levels.txt"):
                rv_levels[rule][k].append(np.loadtxt(path.as_posix()))

    print(rv_fail_probs)
    # rv_levels_avg = {rule_name: defaultdict(list) for rule_name in rule_names}
    for k in Ks:
        fig, ax = plt.subplots()
        max_l = max([ls[0] for ls in rv_levels[rule][k] for rule in rule_names])
        min_l = min([ls[-1] for ls in rv_levels[rule][k] for rule in rule_names])
        for rule in rule_names:
            levels = rv_levels[rule][k]
            for i, level in enumerate(levels):
                if i == 0:
                    ax.plot(level, color=rule_colors[rule], label=rule, alpha=0.5)
                else:
                    ax.plot(level, color=rule_colors[rule], alpha=0.5)
        ax.set_ylim(bottom=-0.0)
        ax.set_title(f"K={k}")
        ax.set_xlabel("Stage")
        ax.set_ylabel("$\gamma_k$")
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_bounds(0.0, max_l)
        ax.spines['bottom'].set_bounds(0.0, ax.get_xlim()[1])
        ax.tick_params(direction='out')
        plt.show()

            # rv_levels_avg[rule][k] = np.mean(rv_levels[rule][k], axis=0)

    for k in Ks:
        print(f"K: {k}")
        for rn in rule_names:
            print(f"{rn}: {np.mean(rv_fail_probs[rn][k]):.2e} (+- {np.std(rv_fail_probs[rn][k]):.2e}) \t {len(rv_fail_probs[rn][k])}")

    published_K = 25
    pub_mus = {rn: np.mean(rv_fail_probs[rn][published_K]) for rn in rule_names}
    pub_stds = {rn: np.std(rv_fail_probs[rn][published_K]) for rn in rule_names}
    row_output = []
    for rn in rule_names:
        mu, std = pub_mus[rn], pub_stds[rn]
        rv_string = f"${mu:.2e}$ $(\\pm {std:.2e})$".replace("e-", "\\text{e-}")
        row_output.append(rv_string)
    print(" & ".join(row_output), "\\\\")

    # for k in Ks:
    #     print(k)
    #     rv_fail_probs = defaultdict(list)
    #     for rule in rule_names:
    #         for path in Path(results_root).rglob(f"*N{N}_K{k}_*/{rule}/stats/failure_prob.txt"):
    #             rv_fail_probs[rule].append(np.loadtxt(path.as_posix()))
    #             # print(path.as_posix())
    #
    #     for rule_name, fps in rv_fail_probs.items():
    #         nfps = np.array(fps)
    #         nonzero_fps = nfps[nfps > 0]
    #         print(f"{rule_name}: {np.mean(nonzero_fps):.2e} (+- {np.std(nonzero_fps):.2e}) \t {nonzero_fps.size}")
    #         # print(f"{rule_name}: {np.mean(fps):.2e} (+- {np.std(fps):.2e})\t {len(fps)}" )
    #         # print(np.mean(fps))
    #         # print(fps)
    #         # print(rv_fail_probs)


if __name__ == "__main__":
    run()