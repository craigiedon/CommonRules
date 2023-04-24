# Load up Nuscenes Model
import time
from typing import Tuple, Callable

from matplotlib import pyplot as plt
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.kernel_ridge import KernelRidge
from torch.utils.data import Dataset, TensorDataset
from sklearn import linear_model
from sklearn.gaussian_process import GaussianProcessClassifier

from NuScenesAnalysis import load_nuscenes_salient, norm_saved
from PlotSalientData import SimpleClassPEM, plot_roc
import torch
import torch.nn.functional as F
import numpy as np
import GPy


def run():
    norm_mus = torch.load("data/nuscenes/inp_mus.pt")
    norm_stds = torch.load("data/nuscenes/inp_stds.pt")

    in_dims = len(norm_mus)

    pem = SimpleClassPEM(in_dims, 512).cuda()
    pem.load_state_dict(torch.load("models/nuscenes/pem_c512_e50.pt"))
    pem.eval()

    X, labels = load_nuscenes_salient("data/nuscenes/salient_inputs.txt", "data/nuscenes/salient_labels.txt")
    X = norm_saved(X, "data/nuscenes/inp_mus.pt", "data/nuscenes/inp_stds.pt")

    train_idx, val_idx = next(StratifiedShuffleSplit(n_splits=1, random_state=1).split(X, labels))
    X, labels = X.cuda(), labels.cuda()

    train_data = TensorDataset(X[train_idx], labels[train_idx])
    val_data = TensorDataset(X[val_idx], labels[val_idx])

    print("Fitting Linear Model...")
    lr = linear_model.LogisticRegression(max_iter=1000)
    lr.fit(train_data.tensors[0].cpu(), train_data.tensors[1].cpu())
    print("Done")
    #

    print("Fitting GP")
    # kernel = 1.0 * RBF()
    # kernel = Matern()
    # gpc = GaussianProcessClassifier(kernel=kernel, random_state=0).fit(train_data.tensors[0].cpu()[:100].reshape(-1 ,6),
    #                                                                    train_data.tensors[1].cpu()[:100])

    t_np = train_data.tensors[0].cpu().numpy().reshape(-1, in_dims)
    l_np = train_data.tensors[1].cpu().numpy().reshape(-1, 1)
    start_time = time.time()
    kernel = GPy.kern.RBF(input_dim=t_np.shape[1])
    gpc = GPy.models.SparseGPClassification(t_np[::5], l_np[::5], kernel=kernel, num_inducing=20)
    gpc.optimize()
    print("Time taken:", time.time() - start_time)
    # gpc.plot()

    # print("Fitting SVM Model...")
    # svm = SVC(probability=True, verbose=True)
    # svm.fit(train_data.tensors[0].cpu(), train_data.tensors[1].cpu())
    print("Done")

    with torch.no_grad():
        pem_preds = torch.sigmoid(pem(val_data.tensors[0]))
    plot_roc(val_data.tensors[1][:, 0].cpu(), pem_preds.cpu(), "NN")

    plot_roc(val_data.tensors[1][:, 0].cpu(), lr.predict_proba(val_data.tensors[0].cpu())[:, 1], "LR")

    val_x_np = val_data.tensors[0].cpu().numpy().reshape(-1, in_dims)
    gpc_preds, _ = gpc.predict_noiseless(val_x_np)
    gpc_lik = GPy.likelihoods.Bernoulli()
    p = gpc_lik.gp_link.transf(gpc_preds)
    plot_roc(val_data.tensors[1].cpu(), p, "GP")
    # plot_roc(val_data.tensors[1].cpu(), svm.predict_proba(val_data.tensors[0].cpu())[:, 1], "SVM")
    plt.legend()
    plt.show()

    num_x, num_y = 50, 50
    ps = torch.stack(torch.meshgrid([torch.linspace(-100, 100, num_x), torch.linspace(-100, 100, num_y)]), dim=2).reshape(-1, 2)
    for o in range(4):
        plt.subplot(2, 2, o + 1)
        plt.title(f"Viz: {o+1}")
        dists = torch.sqrt(ps[:, 0] ** 2 + ps[:, 1] ** 2)
        polars = torch.atan2(ps[:, 1], ps[:, 0])

        ev_rot = torch.full((len(ps),), np.pi / 2.0)
        ev_dims = torch.tensor([4.6, 1.9, 1.7]).repeat(len(ps), 1)
        # ev_occ = F.one_hot(torch.tensor(o), num_classes=4).repeat(len(ps), 1)
        ev_occ = torch.tensor(o).repeat(len(ps)) / 3.0
        # ev_occ = torch.tensor(o).repeat(len(ps))

        eval_xs = torch.column_stack((ps, torch.sin(ev_rot), torch.cos(ev_rot), ev_dims, ev_occ))
        # eval_xs = torch.column_stack((ps, torch.sin(ev_rot), torch.cos(ev_rot), ev_dims))
        # eval_xs = torch.column_stack((ps, torch.sin(ev_rot), torch.cos(ev_rot)))
        # eval_xs = torch.column_stack((dists, torch.sin(polars), torch.cos(polars), torch.sin(ev_rot), torch.cos(ev_rot), ev_dims, ev_occ))
        # eval_xs = torch.column_stack((dists, torch.sin(polars), torch.cos(polars), torch.sin(ev_rot), torch.cos(ev_rot), ev_occ))

        eval_xs = norm_saved(eval_xs, "data/nuscenes/inp_mus.pt", "data/nuscenes/inp_stds.pt")

        # with torch.no_grad():
        #     pem.eval()
        #     z_preds = torch.sigmoid(pem(eval_xs.cuda())).cpu().detach().reshape(num_x, num_y)
        #     print(f"Viz: {o}", z_preds.sum())

        z_preds, _ = gpc.predict_noiseless(eval_xs.cpu().numpy()) # .reshape(num_x, num_y)
        z_preds = gpc_lik.gp_link.transf(z_preds).reshape(num_x, num_y)
        print(f"Viz: {o}", z_preds.sum())

        p_grid = ps.reshape(num_x, num_y, 2)

        plt.pcolormesh(p_grid[:, :, 0], p_grid[:, :, 1], z_preds, vmin=0.0, vmax=1.0)
        plt.colorbar()

    plt.show()



if __name__ == "__main__":
    run()
