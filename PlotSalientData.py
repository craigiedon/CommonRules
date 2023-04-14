import matplotlib.pyplot as plt
import numpy as np
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_curve, roc_auc_score
from torchvision.ops import sigmoid_focal_loss


class SimpleClassPEM(nn.Module):
    def __init__(self, x_dim: int, h_dim: int):
        super().__init__()
        self.ff_nn = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.BatchNorm1d(h_dim),
            nn.Dropout(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.BatchNorm1d(h_dim),
            nn.Dropout(),
            nn.Linear(h_dim, 1)
        )

    def forward(self, x):
        logits = self.ff_nn(x)
        return logits


def detections_by_occlusion(xs, labels):
    occs = xs[:, 5]
    dets = labels[:, 0]

    for i in range(3):
        o_dets = dets[occs == i]
        hits = o_dets[o_dets == 1]
        misses = o_dets[o_dets == 0]

        miss_rate = len(misses) / (len(hits) + len(misses))
        print(f"Occ {i}, total: {len(o_dets)} misses: {100 * miss_rate}%")
        # print(f"Occs : {len(misses)} \t hits: {len(hits)})")


def mus_by_occlusion(xs, labels):
    occs = xs[:, 5]
    dets = labels[:, 0]

    return torch.FloatTensor([dets[occs == i].mean() for i in range(3)])


def guess_mu_by_occ(x, occ_mus):
    occ_hot = x[:, 3:6].detach()
    mu_choices = torch.matmul(occ_hot, occ_mus)
    # mu_choices = occ_hot @ occ_mus
    return mu_choices.unsqueeze(1)


def plot_roc(v_data, model, model_name):
    with torch.no_grad():
        final_val_preds = torch.sigmoid(model(v_data.tensors[0]))
    fpr, tpr, threshs = roc_curve(v_data.tensors[1].cpu(), final_val_preds.cpu(), pos_label=1)
    print(f"ROC AUC: {model_name} --- {roc_auc_score(v_data.tensors[1].cpu(), final_val_preds.cpu())}")
    plt.plot(fpr, tpr, label=model_name)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot(np.linspace(0, 1), np.linspace(0, 1), '--', color='black', alpha=0.4)


def train_model(train_loader, val_loader, pem, loss_fn, epochs) -> nn.Module:
    avg_train_losses = []
    avg_val_losses = []

    # optim = torch.optim.Adam(pem.parameters())
    optim = torch.optim.AdamW(pem.parameters())
    # optim = torch.optim.SGD(pem.parameters(), 1e-3, 0.9)

    for i in range(epochs):

        pem.train()
        train_losses = []
        # guess_one_losses = []
        # guess_mu_losses = []
        # guess_om_losses = []
        for x, label in train_loader:
            pred = pem(x)
            loss = loss_fn(pred, label.unsqueeze(1))

            # pred_nans = pred[torch.isnan(pred)]
            # x_nans = x[torch.isnan(x)]

            # g1_loss = pure_loss_fn(torch.ones_like(pred), label.unsqueeze(1))
            # g_mu_loss = pure_loss_fn(torch.full_like(pred, g_mu), label.unsqueeze(1))

            # g_om = guess_mu_by_occ(x.detach(), occ_mus.detach()).detach()
            # F.binary_cross_entropy(g_om, label.unsqueeze(1).detach())
            # g_om_loss = F.binary_cross_entropy(g_om, label.unsqueeze(1).detach())
            # with torch.no_grad():
            #     g_om_loss = F.binary_cross_entropy(g_om, torch.ones_like(g_om))
            # g_om_loss = pure_loss_fn(g_om, label.unsqueeze(1).detach())

            optim.zero_grad()
            loss.backward()
            optim.step()

            train_losses.append(loss.item())
            # guess_one_losses.append(g1_loss.item())
            # guess_mu_losses.append(g_mu_loss.item())
            # guess_om_losses.append(g_om_loss.item())

        avg_train_loss = np.mean(train_losses)
        # avg_g1_loss = np.mean(guess_one_losses)
        # avg_gmu_loss = np.mean(guess_mu_losses)
        # avg_gom_loss = np.mean(guess_om_losses)

        pem.eval()
        val_losses = []
        for x, label in val_loader:
            with torch.no_grad():
                pred = pem(x)
                val_loss = loss_fn(pred, label.unsqueeze(1))
                val_losses.append(val_loss.item())

        avg_val_loss = np.mean(val_losses)

        print(f"{i}: Avg Loss T - {avg_train_loss} \t V - {avg_val_loss}")
        avg_train_losses.append(avg_train_loss)
        avg_val_losses.append(avg_val_loss)

    plt.plot(range(epochs), avg_train_losses, label='train')
    plt.plot(range(epochs), avg_val_losses, label='val')
    plt.legend(loc='best')
    plt.show()

    # if i % checkpoint_interval == 0:
    #     plot_roc(val_data, pem, i)

    return pem


def run():
    inp_path = "data/salient_inputs.txt"
    label_path = "data/salient_labels.txt"

    s_inp = torch.from_numpy(np.loadtxt(inp_path)).to(dtype=torch.float)
    s_label = torch.from_numpy(np.loadtxt(label_path)).to(dtype=torch.float)

    detections_by_occlusion(s_inp, s_label)
    occ_mus = mus_by_occlusion(s_inp, s_label).cuda()

    tru_dets = s_label[s_label[:, 0] == 0]
    print(len(tru_dets))

    s_xs = s_inp[:, 0]
    s_zs = s_inp[:, 1]
    s_rys = s_inp[:, 2]
    s_lens = s_inp[:, 3]
    s_ws = s_inp[:, 4]
    # X = np.stack((s_xs, s_zs, s_rys, s_lens, s_ws), axis=1)
    # X = np.stack((s_xs, s_zs, s_rys), axis=1)
    X = torch.column_stack((s_xs, s_zs, torch.sin(s_rys), torch.cos(s_rys), s_lens, s_ws))
    mu = X.mean(dim=0)
    std = X.std(dim=0)
    X = (X - X.mean(dim=0)) / X.std(dim=0)

    s_occs = s_inp[:, 5].to(dtype=torch.long)
    one_hot_occs = F.one_hot(s_occs)

    X = torch.column_stack((X, one_hot_occs))
    # X = one_hot_occs.to(dtype=torch.float)
    # X = (s_occs / 2.0).unsqueeze(1)

    s_det = s_label[:, 0]
    g_mu = torch.mean(s_det)

    train_idx, val_idx = next(StratifiedShuffleSplit(n_splits=1, random_state=1).split(X, s_det))
    X = X.cuda()
    s_det = s_det.cuda()

    train_data = TensorDataset(X[train_idx], s_det[train_idx])
    val_data = TensorDataset(X[val_idx], s_det[val_idx])

    batch_size = 512
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    bce_logit_loss_fn = nn.BCEWithLogitsLoss()
    focal_loss_fn = lambda inps, targs: sigmoid_focal_loss(inps, targs, reduction='mean', alpha=0.6)

    # Batches from dataloader
    epochs = [1000]
    capacities = [256]
    loss_fns = [bce_logit_loss_fn]

    configs = list(itertools.product(capacities, epochs, loss_fns))
    pems = [SimpleClassPEM(X.shape[1], conf[0]).cuda() for conf in configs]

    trained_models = []
    for (c, e, l_fn), pm in zip(configs, pems):
        trained_models.append(train_model(train_loader, val_loader, pm, l_fn, e))

    for (c, e, l_fn), pm in zip(configs, pems):
        plot_roc(val_loader.dataset, pm, f"C:{c} E:{e} L:{l_fn}")

    plt.legend()
    plt.show()

    num_x, num_y = 50, 50
    ps = torch.stack(torch.meshgrid([torch.linspace(-100, 100, num_x), torch.linspace(-100, 100, num_y)]), dim=2).reshape(
        -1, 2)

    # ps = (ps - mu[0:2]) / std[0:2]

    for o in range(3):
        # Displaying them in reverse order here just to line up with a presentation
        plt.subplot(1, 3, 3 - o)

        # eval_xs = torch.column_stack((ps, ev_rot, ev_occ))
        ev_rot = torch.full((len(ps),), -np.pi / 2.0)  # -np.pi / 2.0)
        ev_l = torch.full((len(ps),), 4.0)  # -np.pi / 2.0)
        ev_w = torch.full((len(ps),), 1.6)  # -np.pi / 2.0)
        # ev_rot = (ev_rot - mu[2]) / std[2]

        eval_xs = torch.column_stack((ps, torch.sin(ev_rot), torch.cos(ev_rot), ev_l, ev_w))
        eval_xs = (eval_xs - mu) / std

        ev_occ = F.one_hot(torch.tensor(o), num_classes=3).repeat(len(ps), 1)
        eval_xs = torch.column_stack((eval_xs, ev_occ))
        m = pems[0]

        with torch.no_grad():
            m.eval()
            z_preds = torch.sigmoid(m(eval_xs.cuda())).cpu().detach().reshape(num_x, num_y)

        p_grid = ps.reshape(num_x, num_y, 2)

        plt.pcolormesh(p_grid[:, :, 0], p_grid[:, :, 1], z_preds)
        plt.colorbar()
        # plt.tight_layout()

    # plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    run()
