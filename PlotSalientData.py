import matplotlib.pyplot as plt
import numpy as np
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_curve
from torchvision.ops import sigmoid_focal_loss


class SimpleClassPEM(nn.Module):
    def __init__(self, x_dim: int, h_dim: int):
        super().__init__()
        self.ff_nn = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            # nn.Dropout(),
            nn.BatchNorm1d(h_dim),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.BatchNorm1d(h_dim),
            # nn.Dropout(),
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
    plt.plot(fpr, tpr, label=model_name)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot(np.linspace(0, 1), np.linspace(0, 1), '--', color='black', alpha=0.4)


def run():
    inp_path = "data/salient_inputs.txt"
    label_path = "data/salient_labels.txt"

    s_inp = np.loadtxt(inp_path)
    s_label = np.loadtxt(label_path)

    detections_by_occlusion(s_inp, s_label)
    occ_mus = mus_by_occlusion(s_inp, s_label).cuda()

    tru_dets = s_label[s_label[:, 0] == 0]
    print(len(tru_dets))

    s_xs = s_inp[:, 0]
    s_zs = s_inp[:, 1]
    s_rys = s_inp[:, 2]
    s_lens = s_inp[:, 3]
    s_ws = s_inp[:, 4]
    X = np.stack((s_xs, s_zs, s_rys, s_lens, s_ws), axis=1)
    # X = np.stack((s_xs, s_zs, s_rys), axis=1)
    X = torch.from_numpy(X).to(dtype=torch.float)

    mus = X.mean(dim=0)
    stds = X.std(dim=0)
    X = (X - X.mean(dim=0)) / X.std(dim=0)

    s_occs = torch.from_numpy(s_inp[:, 5]).to(dtype=torch.long)
    one_hot_occs = F.one_hot(s_occs)

    X = torch.column_stack((X, one_hot_occs))
    # X = one_hot_occs.to(dtype=torch.float)
    # X = (s_occs / 2.0).unsqueeze(1)

    s_det = s_label[:, 0]
    s_det = torch.from_numpy(s_det).to(dtype=torch.float)
    g_mu = torch.mean(s_det)

    # plt.hist2d(s_xs, s_zs, bins=30)
    # plt.colorbar()
    # plt.show()

    # clf = MLPClassifier(hidden_layer_sizes=(5, 2), alpha=1e-5, random_state=1)
    # clf.fit(X, s_det)

    pem = SimpleClassPEM(X.shape[1], 128).cuda()
    # pem = nn.Linear(X.shape[1], 1).cuda()
    # Loss
    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = lambda inps, targs: sigmoid_focal_loss(inps, targs, reduction='mean', alpha=0.6)

    pure_loss_fn = nn.BCELoss()
    optim = torch.optim.Adam(pem.parameters())
    # optim = torch.optim.SGD(pem.parameters(), lr=1e-2)

    train_idx, val_idx = next(StratifiedShuffleSplit(n_splits=1, random_state=1).split(X, s_det))
    X = X.cuda()
    s_det = s_det.cuda()
    train_data = TensorDataset(X[train_idx], s_det[train_idx])
    val_data = TensorDataset(X[val_idx], s_det[val_idx])

    train_loader = DataLoader(train_data, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1024, shuffle=False)

    # Batches from dataloader
    epochs = 250
    # epochs = 10
    # checkpoint_interval = 100
    # fig, axs = plt.subplots(1,2)
    # loss_ax, roc_ax = axs

    avg_train_losses = []
    avg_val_losses = []

    for i in range(epochs):

        pem.train()
        train_losses = []
        guess_one_losses = []
        guess_mu_losses = []
        guess_om_losses = []
        for x, label in train_loader:
            pred = pem(x)
            loss = loss_fn(pred, label.unsqueeze(1))

            # pred_nans = pred[torch.isnan(pred)]
            # x_nans = x[torch.isnan(x)]

            g1_loss = pure_loss_fn(torch.ones_like(pred), label.unsqueeze(1))
            g_mu_loss = pure_loss_fn(torch.full_like(pred, g_mu), label.unsqueeze(1))

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
            guess_one_losses.append(g1_loss.item())
            guess_mu_losses.append(g_mu_loss.item())
            # guess_om_losses.append(g_om_loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_g1_loss = np.mean(guess_one_losses)
        avg_gmu_loss = np.mean(guess_mu_losses)
        # avg_gom_loss = np.mean(guess_om_losses)

        pem.eval()
        val_losses = []
        for x, label in val_loader:
            with torch.no_grad():
                pred = pem(x)
                val_loss = loss_fn(pred, label.unsqueeze(1))
                val_losses.append(val_loss.item())

        avg_val_loss = np.mean(val_losses)

        print(f"{i}: Avg Loss T - {avg_train_loss} \t V - {avg_val_loss} \t G_1 - {avg_g1_loss} \t G_mu {avg_gmu_loss}")
        avg_train_losses.append(avg_train_loss)
        avg_val_losses.append(avg_val_loss)

        # if i % checkpoint_interval == 0:
        #     plot_roc(val_data, pem, i)

    plot_roc(val_data, pem, epochs)
    # plot_roc(val_data, lambda x: torch.full((x.shape[0], 1), torch.logit(g_mu)), "g_mu")
    plt.legend()
    plt.show()

    plt.plot(range(epochs), avg_train_losses, label='train')
    plt.plot(range(epochs), avg_val_losses, label='val')
    plt.legend()
    plt.show()

    # p_xs = np.linspace(-40, 40)
    # p_ys = np.linspace(0, 40)
    # ev = np.array(list(itertools.product(p_xs, p_ys)))

    # preds = clf.predict(ev)
    # ranges = (-40, 40), (0, 80)

    # Load in the salient data
    # Stick x/ys on some sort of histogram
    # Display


if __name__ == "__main__":
    run()
