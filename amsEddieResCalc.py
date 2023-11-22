import os
import glob
from collections import defaultdict
import scipy.stats as scp

import numpy as np
from pathlib import Path


def run():
    rule_names = ["rg_1", "rg_2", "rg_4", "ri_2"]
    Ks = [2, 12, 25, 125, 225]
    results_root = "/media/craig/Extreme Pro/amsResults/amsRG"

    for k in Ks:
        print(k)
        rv_fail_probs = defaultdict(list)
        for rule in rule_names:
            for path in Path(results_root).rglob(f"*N250_K{k}_*/{rule}/stats/failure_prob.txt"):
                rv_fail_probs[rule].append(np.loadtxt(path.as_posix()))
                # print(path.as_posix())

        for rule_name, fps in rv_fail_probs.items():
            nfps = np.array(fps)
            nonzero_fps = nfps[nfps > 0]
            print(f"{rule_name}: {np.mean(nonzero_fps):.2e} (+- {np.std(nonzero_fps):.2e}) \t {nonzero_fps.size}")
            # print(f"{rule_name}: {np.mean(fps):.2e} (+- {np.std(fps):.2e})\t {len(fps)}" )
            # print(np.mean(fps))
            # print(fps)
            # print(rv_fail_probs)


if __name__ == "__main__":
    run()