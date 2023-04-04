import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
import itertools
import torch


inp_path = "data/salient_inputs.txt"
label_path = "data/salient_labels.txt"

s_inp = np.loadtxt(inp_path)
s_label = np.loadtxt(label_path)


s_xs = s_inp[:, 0]
s_zs = s_inp[:, 1]
X = np.stack((s_xs, s_zs), axis=1)
s_det = s_label[:, 0]

# plt.hist2d(s_xs, s_zs, bins=30)
# plt.colorbar()
# plt.show()

clf = MLPClassifier(hidden_layer_sizes=(5, 2), alpha=1e-5, random_state=1)
clf.fit(X, s_det)

p_xs = np.linspace(-40, 40)
p_ys = np.linspace(0, 40)
ev = np.array(list(itertools.product(p_xs, p_ys)))

preds = clf.predict(ev)

ranges = (-40, 40), (0, 80)




# Load in the salient data
# Stick x/ys on some sort of histogram
# Display