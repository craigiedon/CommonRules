import os
import glob
from collections import defaultdict
import scipy.stats as scp

import numpy as np
from pathlib import Path

from matplotlib import pyplot as plt
from scipy.special import logsumexp


def tufte_plot(value_map, x_label, y_label, ax_title):
    fig, ax = plt.subplots()
    all_vals = np.array(list(value_map.values()))
    min_val = np.amin(all_vals[np.isfinite(all_vals)])
    max_val = np.amax(all_vals[np.isfinite(all_vals)])
    v_range = max_val - min_val
    ax.set_ylim([min_val - 0.1 * v_range, max_val + 0.1 * v_range])
    for leg, vals in value_map.items():
        converted_vals = np.nan_to_num(vals, neginf=min_val - 0.12 * v_range)
        ax.plot(converted_vals, linestyle='-', zorder=1)
        ax.scatter(range(len(vals)), converted_vals, label=leg, s=15, zorder=3)
        ax.scatter(range(len(vals)), converted_vals, color='white', s=75, zorder=2)

    ax.set_title(ax_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_bounds([0, len(vals) - 1])
    ax.spines['left'].set_bounds([min_val, max_val])
    ax.tick_params(direction='in')
    # plt.tight_layout()
    plt.show()


def run():
    # plt.rcParams["font.family"] = 'serif'
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    # TODO: Add in the "true type" thing for plots
    rule_names = ["rg_1", "rg_2", "rg_4", "ri_2"]
    # Ks = [2, 12, 25, 125, 225]
    results_root = "/media/craig/Extreme Pro/amsResults/cePemRG"

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

    # Chart the average "levels" of each CE
    avg_level_map = {rule: np.mean(levels, axis=0) for rule, levels in rv_levels.items()}
    # tufte_plot(avg_level_map, "Stage", "$\gamma_{k}$", "Robustness Threshold per Stage")

    # Chart the log failure probabilities as levels proceed
    avg_fps_map = {rule: logsumexp(lfps, axis=0) - np.log(len(lfps)) for rule, lfps in rv_log_fail_probs.items()}
    probs_map = {rule: np.exp(lfps) for rule, lfps in rv_log_fail_probs.items()}
    # tufte_plot(avg_fps_map, "Stage", "Log Failure Probability", "Log Failure Probability Estimate per Stage")

    # Chart Number of Failures Per Stage
    # tufte_plot(rv_stage_fails, "Stage", "Failures", "Number of True Failures Per Stage")

    final_mu_map = {rule: np.mean(probs, axis=0)[-1] for rule, probs in probs_map.items()}
    final_std_map = {rule: np.std(probs, axis=0)[-1] for rule, probs in probs_map.items()}
    for rule, probs in probs_map.items():
        print(len(probs))
        print(f"Rule: {rule} Fail Prob: {final_mu_map[rule]:.2e} (+- {final_std_map[rule]:.2e})")

    # For latex format
    row_output = []
    for rule in probs_map.keys():
        mu, std = final_mu_map[rule], final_std_map[rule]
        rv_string = f"${mu:.2e}$ $(\\pm {std:.2e})$".replace("e-", "\\text{e-}")
        row_output.append(rv_string)
    print(" & ".join(row_output), "\\\\")

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
