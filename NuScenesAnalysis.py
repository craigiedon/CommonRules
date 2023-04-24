import itertools
from typing import Callable, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader, Dataset

from PlotSalientData import train_model, SimpleClassPEM, plot_roc


def detections_by_viz(xs, labels):
    occs = xs[:, 6]
    dets = labels[:, 0]

    for i in range(1, 5):
        o_dets = dets[occs == i]
        hits = o_dets[o_dets == 1]
        misses = o_dets[o_dets == 0]

        miss_rate = len(misses) / (len(hits) + len(misses))
        print(f"Viz {i}, total: {len(o_dets)}\t misses: {100 * miss_rate}%")
        # print(f"Occs : {len(misses)} \t hits: {len(hits)})")


def rotations_histogram(rots: np.ndarray):
    rots = rots % (2 * np.pi)
    n_numbers = 100
    bins_number = 8
    bins = np.linspace(0.0, 2 * np.pi, bins_number + 1)
    n, _, _ = plt.hist(rots, bins)

    plt.clf()
    width = 2 * np.pi / bins_number
    ax = plt.subplot(1, 1, 1, projection='polar')
    bars = ax.bar(bins[:bins_number], n, width=width, bottom=0.0)
    for bar in bars:
        bar.set_alpha(0.5)
    plt.show()


def distance_histogram(xs, ys):
    plt.hist2d(xs.numpy(), ys.numpy(), range=[[-100, 100], [-100, 100]], bins=30)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Instances by Position")
    plt.colorbar()
    plt.show()


def visibility_by_dist_histogram(s_inp, s_labels):
    xs = s_inp[:, 0]
    ys = s_inp[:, 1]
    dets = s_labels[:, 0]

    dists = torch.sqrt(xs ** 2 + ys ** 2)

    bin_totals, bs = np.histogram(dists, bins=20)
    det_totals, _ = np.histogram(dists[dets == 1], bins=bs)

    det_percentages = (det_totals / bin_totals) * 100.0

    plt.bar(range(len(det_percentages)), det_percentages)  # , edgecolor='black')
    plt.xticks(range(len(det_percentages)), [f"$\leq${bs[i + 1]:.1f}" for i in range(len(bs) - 1)])
    plt.xlabel("Distance")
    plt.ylabel("Detection Percentage (%)")
    plt.title("Detection Percentage by Distance")
    plt.show()


def load_nuscenes_salient(inp_path, label_path) -> Tuple[Tensor, Tensor]:
    s_inp = torch.from_numpy(np.loadtxt(inp_path)).to(dtype=torch.float)
    s_label = torch.from_numpy(np.loadtxt(label_path)).to(dtype=torch.float)

    # Setup input data columns
    s_xs = s_inp[:, 0]
    s_ys = s_inp[:, 1]

    s_r = s_inp[:, 2]
    s_dims = s_inp[:, 3:6]

    s_viz = (s_inp[:, 6] - 1) / 3.0
    X = torch.column_stack((s_xs, s_ys, torch.sin(s_r), torch.cos(s_r), s_dims, s_viz))

    # s_det = s_label[:, 0]

    return X, s_label


def norm_saved(X: Tensor, mus_path: str, stds_path: str) -> Tensor:
    mus = torch.load(mus_path)
    stds = torch.load(stds_path)

    normed_X = X.clone()
    normed_X[:, 0:len(mus)] = (normed_X[:, 0:len(mus)] - mus) / stds

    return normed_X


# def norm_nusc_ins(X: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
#     # Last 4 are one-hot and do not require normalization
#     mus = X[:, :-4].mean(dim=0)
#     stds = X[:, :-4].std(dim=0)
#     norm_X = X.clone()
#     norm_X[:, :-4] = (norm_X[:, :-4] - mus) / stds
#     return norm_X, mus, stds

def norm_nusc_ins_no_cats(X: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    mus = X.mean(dim=0)
    stds = X.std(dim=0)
    return (X - mus) / stds, mus, stds


def run():
    inp_path = "data/nuscenes/salient_inputs.txt"
    label_path = "data/nuscenes/salient_labels.txt"

    X, labels = load_nuscenes_salient(inp_path, label_path)
    X, mus, stds = norm_nusc_ins_no_cats(X)
    train_idx, val_idx = next(StratifiedShuffleSplit(n_splits=1, random_state=1).split(X, labels))
    X, labels = X.cuda(), labels.cuda()

    train_data = TensorDataset(X[train_idx], labels[train_idx])
    val_data = TensorDataset(X[val_idx], labels[val_idx])

    batch_size = 1024
    epochs = [50]
    # capacities = [256, 512, 1024, 2048]
    # capacities = [256, 512, 1024]
    capacities = [512]
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    configs = list(itertools.product(capacities, epochs))
    pems = [SimpleClassPEM(X.shape[1], conf[0]).cuda() for conf in configs]

    trained_models = []
    for (c, e), pm in zip(configs, pems):
        trained_models.append(train_model(train_loader, val_loader, pm, loss_fn, e, plot=True))

    for (c, e), pm in zip(configs, pems):
        with torch.no_grad():
            pem_preds = torch.sigmoid(pm(val_data.tensors[0]))
        plot_roc(val_loader.dataset.tensors[1][:, 0].cpu(), pem_preds.cpu(), f"C:{c} E:{e}")

    plt.legend()
    plt.show()

    # Save model and params
    for (c, e), pm in zip(configs, pems):
        torch.save(pm.state_dict(), f"models/nuscenes/pem_c{c}_e{e}.pt")
    torch.save(mus, f"data/nuscenes/inp_mus.pt")
    torch.save(stds, f"data/nuscenes/inp_stds.pt")


if __name__ == "__main__":
    run()
