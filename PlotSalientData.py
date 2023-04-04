import matplotlib.pyplot as plt
import numpy as np
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit


class SimpleClassPEM(nn.Module):
    def __init__(self, x_dim: int, h_dim: int):
        super().__init__()
        self.ff_nn = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1)
        )

    def forward(self, x):
        logits = self.ff_nn(x)
        return logits


inp_path = "data/salient_inputs.txt"
label_path = "data/salient_labels.txt"

s_inp = np.loadtxt(inp_path)
s_label = np.loadtxt(label_path)

tru_dets = s_label[s_label[:, 0] == 0]
print(len(tru_dets))


s_xs = s_inp[:, 0]
s_zs = s_inp[:, 1]
s_rys = s_inp[:, 2]
X = np.stack((s_xs, s_zs, s_rys), axis=1)
X = torch.from_numpy(X).to(dtype=torch.float)
X = (X - X.mean(dim=0)) / X.std(dim=0)

s_occs = torch.from_numpy(s_inp[:, 3]).to(dtype=torch.long)
one_hot_occs = F.one_hot(s_occs)

X = torch.column_stack((X, one_hot_occs))

s_det = s_label[:, 0]
s_det = torch.from_numpy(s_det).to(dtype=torch.float)


# plt.hist2d(s_xs, s_zs, bins=30)
# plt.colorbar()
# plt.show()

# clf = MLPClassifier(hidden_layer_sizes=(5, 2), alpha=1e-5, random_state=1)
# clf.fit(X, s_det)

pem = SimpleClassPEM(X.shape[1], 20).cuda()
# pem = nn.Linear(2,1)
# Loss
logits_loss_fn = nn.BCEWithLogitsLoss()
optim = torch.optim.Adam(pem.parameters(), lr=1e-2)
# optim = torch.optim.SGD(pem.parameters(), lr=1e-2)

train_idx, val_idx = next(StratifiedShuffleSplit(n_splits=1, random_state=1).split(X, s_det))
X = X.cuda()
s_det=  s_det.cuda()
train_data = TensorDataset(X[train_idx], s_det[train_idx])
val_data = TensorDataset(X[val_idx], s_det[val_idx])

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=1024, shuffle=False)

# Batches from dataloader
epochs = 5000

avg_train_losses = []
avg_val_losses = []

for i in range(epochs):

    pem.train()
    train_losses = []
    for x, label in train_loader:

        pred = pem(x)
        loss = logits_loss_fn(pred, label.unsqueeze(1))

        optim.zero_grad()
        loss.backward()
        optim.step()

        train_losses.append(loss.item())

    avg_train_loss = np.mean(train_losses)

    pem.eval()
    val_losses = []
    for x, label in val_loader:
        with torch.no_grad():
            pred = pem(x)
            val_loss = logits_loss_fn(pred, label.unsqueeze(1))
            val_losses.append(val_loss.item())

    avg_val_loss = np.mean(val_losses)

    print(f"{i}: Avg Loss T - {np.mean(train_losses)} \t V - {np.mean(val_losses)}")

    avg_train_losses.append(avg_train_loss)
    avg_val_losses.append(avg_val_loss)

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