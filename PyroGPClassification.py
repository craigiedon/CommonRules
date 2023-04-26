import os
import time

import matplotlib.pyplot as plt
import torch
import numpy as np

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist

from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sns
from pyro.contrib.gp.kernels import Product
from pyro.infer import TraceMeanField_ELBO, JitTraceMeanField_ELBO, Trace_ELBO
from pyro.infer.util import torch_backward, torch_item
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.utils.data import TensorDataset

from NuScenesAnalysis import load_nuscenes_salient, norm_saved
from PlotSalientData import plot_roc


def plot_loss(loss):
    plt.plot(loss)
    plt.xlabel("iterations")
    _ = plt.ylabel("Loss")


def train_gp(vgsp, val_data, num_steps, optimizer, scheduler_step=None):
    loss_fn = TraceMeanField_ELBO().differentiable_loss

    if scheduler_step is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, scheduler_step, 0.5)
    else:
        scheduler = None

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(vgsp.model, vgsp.guide)
        torch_backward(loss)
        return loss

    losses = []
    val_losses = []
    for i in range(num_steps):
        loss = optimizer.step(closure)
        if scheduler is not None:
            scheduler.step()
            print(f"{i}: {loss.item():.3f}\t (lr={scheduler.get_last_lr()})")
        else:
            print(f"{i}: {loss.item():.3f}")
        losses.append(torch_item(loss))

        with torch.no_grad():
            gp_v_preds, gp_v_vars = vgsp(val_data.tensors[0], full_cov=False)
            val_loss = torch.nn.functional.binary_cross_entropy_with_logits(gp_v_preds, val_data.tensors[1][:, 0])
            print(f"V: {val_loss.item()}")
            val_losses.append(torch_item(val_loss))

    return losses, val_losses


def save_gp_classifier(folder_path: str, vsgp: gp.models.VariationalSparseGP):
    # Make folder on folder path (re-create if already there)
    os.makedirs(folder_path, exist_ok=True)

    # Save important model details
    torch.save(vsgp.state_dict(), os.path.join(folder_path, "state_dict.pt"))
    torch.save(vsgp.X, os.path.join(folder_path, "train_ins.pt"))
    torch.save(vsgp.y, os.path.join(folder_path, "train_labels.pt"))
    torch.save(vsgp.Xu, os.path.join(folder_path, "inducing_points.pt"))
    torch.save(vsgp.kernel, os.path.join(folder_path, "kernel.pt"))


def load_gp_classifier(folder_path):
    train_ins = torch.load(os.path.join(folder_path, "train_ins.pt"))
    train_labels = torch.load(os.path.join(folder_path, "train_labels.pt"))
    Xu = torch.load(os.path.join(folder_path, "inducing_points.pt"))
    kernel = torch.load(os.path.join(folder_path, "kernel.pt"))

    vsgp = gp.models.VariationalSparseGP(train_ins, train_labels, kernel, Xu, gp.likelihoods.Binary(),
                                         jitter=1e-03).cuda()
    vsgp.load_state_dict(torch.load(os.path.join(folder_path, "state_dict.pt")))
    return vsgp


def run():
    norm_mus = torch.load("data/nuscenes/inp_mus.pt")
    norm_stds = torch.load("data/nuscenes/inp_stds.pt")

    X, labels = load_nuscenes_salient("data/nuscenes/salient_inputs.txt", "data/nuscenes/salient_labels.txt")
    X = norm_saved(X, "data/nuscenes/inp_mus.pt", "data/nuscenes/inp_stds.pt")

    train_idx, val_idx = next(StratifiedShuffleSplit(n_splits=1, random_state=1).split(X, labels[:, 0]))
    X, labels = X.cuda(), labels.cuda()

    train_data = TensorDataset(X[train_idx], labels[train_idx])
    val_data = TensorDataset(X[val_idx], labels[val_idx])

    kernel = gp.kernels.RationalQuadratic(X.shape[1])

    likelihood = gp.likelihoods.Binary()

    subset = len(train_data.tensors[0])

    t_lows = torch.min(train_data.tensors[0], dim=0)[0]
    t_highs = torch.max(train_data.tensors[0], dim=0)[0]

    Xu = train_data.tensors[0][::100]
    # Xu = torch.rand((590, 8))

    vsgp = gp.models.VariationalSparseGP(train_data.tensors[0][:subset],
                                         train_data.tensors[1][:subset, 0],
                                         kernel,
                                         Xu=Xu,
                                         likelihood=likelihood,
                                         # whiten=False,
                                         jitter=1e-03).cuda()

    elbo_losses, val_losses = train_gp(vsgp, val_data, num_steps=500,
                                       optimizer=torch.optim.AdamW(vsgp.parameters(), lr=0.1),
                                       scheduler_step=200)

    plot_loss(elbo_losses)
    plt.show()

    plot_loss(val_losses)
    plt.show()

    # torch.save(vsgp.state_dict(), f"models/nuscenes/vsgp_class_i{len(Xu)}.pt")
    # vsgp.load_state_dict(torch.load(f"models/nuscenes/vsgp_class_i{len(Xu)}.pt"))

    save_gp_classifier(f"models/nuscenes/vsgp_class", vsgp)

    vsgp = load_gp_classifier(f"models/nuscenes/vsgp_class")

    gp_v_preds, gp_v_vars = vsgp(val_data.tensors[0], full_cov=False)
    plot_roc(val_data.tensors[1][:, 0].cpu().detach(), torch.sigmoid(gp_v_preds.cpu().detach()), "GP")
    plt.show()

    # plot(model=vgsp, plot_observed_data=True, plot_predictions=True)
    # plt.show()

    num_x, num_y = 100, 100
    viz_range = 110
    ps = torch.stack(
        torch.meshgrid([torch.linspace(-viz_range, viz_range, num_x), torch.linspace(-viz_range, viz_range, num_y)]),
        dim=2).reshape(-1, 2)
    for o in range(4):
        plt.subplot(2, 2, o + 1)
        plt.title(f"Viz: {o + 1}")

        ev_rot = torch.full((len(ps),), np.pi / 2.0)
        ev_dims = torch.tensor([4.6, 1.9, 1.7]).repeat(len(ps), 1)
        ev_occ = torch.tensor(o).repeat(len(ps)) / 3.0

        eval_xs = torch.column_stack((ps, torch.sin(ev_rot), torch.cos(ev_rot), ev_dims, ev_occ))
        eval_xs = norm_saved(eval_xs, "data/nuscenes/inp_mus.pt", "data/nuscenes/inp_stds.pt")

        z_preds, _ = vsgp(eval_xs.cuda(), full_cov=True)
        z_preds = torch.sigmoid(z_preds.reshape(num_x, num_y))
        print(f"Viz: {o}", z_preds.sum())

        p_grid = ps.reshape(num_x, num_y, 2)

        plt.pcolormesh(p_grid[:, :, 0], p_grid[:, :, 1], z_preds.cpu().detach(), vmin=0.0, vmax=1.0)
        plt.colorbar()

    plt.show()


if __name__ == "__main__":
    run()
