import os
import matplotlib.pyplot as plt
import torch
import numpy as np

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist

from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

smoke_test = "CI" in os.environ  # ignore; used to check code integrity in the Pyro repo
assert pyro.__version__.startswith('1.8.4')
pyro.set_rng_seed(0)


# note that this helper function does three different things:
# (i) plots the observed data;
# (ii) plots the predictions from the learned GP after conditioning on data;
# (iii) plots samples from the GP prior (with no conditioning on observed data)


def plot(
        plot_observed_data=False,
        plot_predictions=False,
        n_prior_samples=0,
        model=None,
        kernel=None,
        n_test=500,
        ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    if plot_observed_data:
        ax.plot(X.numpy(), y.numpy(), "kx")
    if plot_predictions:
        Xtest = torch.linspace(-0.5, 5.5, n_test)  # test inputs
        # compute predictive mean and variance
        with torch.no_grad():
            if type(model) == gp.models.VariationalSparseGP:
                mean, cov = model(Xtest, full_cov=True)
            else:
                mean, cov = model(Xtest, full_cov=True, noiseless=False)
        sd = cov.diag().sqrt()  # standard deviation at each input point x
        ax.plot(Xtest.numpy(), mean.numpy(), "r", lw=2)  # plot the mean
        ax.fill_between(
            Xtest.numpy(),  # plot the two-sigma uncertainty about the mean
            (mean - 2.0 * sd).numpy(),
            (mean + 2.0 * sd).numpy(),
            color="C0",
            alpha=0.3,
        )
    if n_prior_samples > 0:  # plot samples from the GP prior
        Xtest = torch.linspace(-0.5, 5.5, n_test)  # test inputs
        noise = (
            model.noise
            if type(model) != gp.models.VariationalSparseGP
            else model.likelihood.variance
        )
        cov = kernel.forward(Xtest) + noise.expand(n_test).diag()
        samples = dist.MultivariateNormal(
            torch.zeros(n_test), covariance_matrix=cov
        ).sample(sample_shape=(n_prior_samples,))
        ax.plot(Xtest.numpy(), samples.numpy().T, lw=2, alpha=0.4)

    ax.set_xlim(-0.5, 5.5)


# N = 20
# X = dist.Uniform(0.0, 5.0).sample(sample_shape=(N,))
# y = 0.5 * torch.sin(3 * X) + dist.Normal(0.0, 0.2).sample(sample_shape=(N,))

# plot(plot_observed_data=True)
# plt.show()

# kernel = gp.kernels.RBF(input_dim=1, variance=torch.tensor(5.0), lengthscale=torch.tensor(10.0))
# gpr = gp.models.GPRegression(X, y, kernel, noise=torch.tensor(0.1))
#
# # Setting Priors
# gpr.kernel.lengthscale = pyro.nn.PyroSample(dist.LogNormal(0.0, 1.0))
# gpr.kernel.variance = pyro.nn.PyroSample(dist.LogNormal(0.0, 1.0))

# plot(model=gpr, kernel=kernel, n_prior_samples=2)
# _ = plt.ylim((-8, 8))
# plt.show()

# kernel2 = gp.kernels.RBF(
#     input_dim=1, variance=torch.tensor(6.0), lengthscale=torch.tensor(1)
# )
# gpr2 = gp.models.GPRegression(X, y, kernel2, noise=torch.tensor(0.1))
# plot(model=gpr2, kernel=kernel2, n_prior_samples=2)
# _ = plt.ylim((-8, 8))
# plt.show()

# optimizer = torch.optim.Adam(gpr.parameters(), lr=0.005)
# loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
# losses = []
# variances = []
# lengthscales = []
# noises = []
# num_steps = 2000
#
# for i in range(num_steps):
#     variances.append(gpr.kernel.variance.item())
#     noises.append(gpr.noise.item())
#     lengthscales.append(gpr.kernel.lengthscale.item())
#     optimizer.zero_grad()
#     loss = loss_fn(gpr.model, gpr.guide)
#     loss.backward()
#     optimizer.step()
#     losses.append(loss.item())


def plot_loss(loss):
    plt.plot(loss)
    plt.xlabel("iterations")
    _ = plt.ylabel("Loss")


# plot_loss(losses)
# plt.show()

# plot(model=gpr, plot_observed_data=True, plot_predictions=True)
# plt.show()

N = 1000
X = dist.Uniform(0.0, 5.0).sample(sample_shape=(N,))
y = 0.5 * torch.sin(3 * X) + dist.Normal(0.0, 0.2).sample(sample_shape=(N,))
Xu = torch.arange(10.0) / 2.0


def plot_inducing_points(Xu, ax=None):
    for xu in Xu:
        g = ax.axvline(xu, color="red", linestyle="-.", alpha=0.5)
    ax.legend(
        handles=[g],
        labels=["Inducing Point Locations"],
        bbox_to_anchor=(0.5, 1.15),
        loc="upper center",
    )


# plot_inducing_points(Xu, plt.gca())
# plt.show()

# kernel = gp.kernels.RBF(input_dim=1)
# likelihood = gp.likelihoods.Gaussian()
# vsgp = gp.models.VariationalSparseGP(
#     X, y, kernel, Xu=Xu, likelihood=likelihood, whiten=True
# )
# 
# num_steps = 1500
# losses = gp.util.train(vsgp, num_steps=num_steps)
# plot_loss(losses)
# plt.show()
# 
# plot(model=vsgp, plot_observed_data=True, plot_predictions=True)
# plt.show()

df = sns.load_dataset("iris")
df.head()

X = torch.from_numpy(
    df[df.columns[2:4]].values.astype("float32"),
)
df["species"] = df["species"].astype("category")
# encode the species as 0, 1, 2
y = torch.from_numpy(df["species"].cat.codes.values.copy())

# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors=(0, 0, 0))
# plt.xlabel("Feature 1 (Petal length)")
# _ = plt.ylabel("Feature 2 (Petal width)")
# plt.show()

kernel = gp.kernels.RBF(input_dim=2)
pyro.clear_param_store()
likelihood = gp.likelihoods.MultiClass(num_classes=3)

model = gp.models.VariationalGP(
    X,
    y,
    kernel,
    likelihood=likelihood,
    whiten=True,
    jitter=1e-03,
    latent_shape=torch.Size([3]),
)

num_steps = 1000
loss = gp.util.train(model, num_steps=num_steps)

# plot_loss(loss)
# plt.show()

mean, var = model(X)
y_hat = model.likelihood(mean, var)
print(f"Accuracy: {(y_hat == y).sum() * 100 / (len(y)) :0.2f}%")

xs = torch.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, steps=100)
ys = torch.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, steps=100)
xx, yy = torch.meshgrid(xs, ys, indexing="xy")

with torch.no_grad():
    mean, var = model(torch.vstack((xx.ravel(), yy.ravel())).t())
    Z = model.likelihood(mean, var)


def plot_pred_2d(arr, xx, yy, contour=False, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    image = ax.imshow(
        arr,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        aspect="equal",
        origin="lower",
        cmap=plt.cm.PuOr_r,
    )
    if contour:
        contours = ax.contour(
            xx,
            yy,
            torch.sigmoid(mean).reshape(xx.shape),
            levels=[0.5],
            linewidths=2,
            colors=["k"],
        )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    ax.get_figure().colorbar(image, cax=cax)
    if title:
        ax.set_title(title)


# fig, ax = plt.subplots(ncols=3, figsize=(16, 4))
# for cl in [0, 1, 2]:
#     plot_pred_2d(
#         mean[cl, :].reshape(xx.shape), xx, yy, ax=ax[cl], title=f"f (class {cl})"
#     )

p_class = torch.nn.functional.softmax(mean, dim=0)

fig, ax = plt.subplots(ncols=3, figsize=(16, 4))
for cl in [0, 1, 2]:
    plot_pred_2d(
        p_class[cl, :].reshape(xx.shape), xx, yy, ax=ax[cl], title=f" p(class {cl})"
    )


plt.show()
