# Set up a basic classification
import math
import torch
import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import LinearKernel, ScaleKernel, RBFKernel
from gpytorch.models import AbstractVariationalGP, ApproximateGP, ExactGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from matplotlib import pyplot as plt
from gpytorch.mlls.variational_elbo import VariationalELBO

train_x = torch.linspace(0, 1, 10)
train_y = torch.sign(torch.cos(train_x * (4 * math.pi)))

from gpytorch.models import AbstractVariationalGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy


class GPClassificationModel(ApproximateGP):
    def __init__(self, train_x, train_y, likelihood):
        # variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_distribution = CholeskyVariationalDistribution(10)
        variational_strategy = VariationalStrategy(self, train_x, variational_distribution)
        # variational_strategy = VariationalStrategy(self, train_x, variational_distribution)
        # super(GPClassificationModel, self).__init__(variational_strategy)
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


# Initialize model and likelihood
likelihood = gpytorch.likelihoods.BernoulliLikelihood()
model = GPClassificationModel(train_x, train_y, likelihood)

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# optimizer = torch.optim.LBFGS(model.parameters())

# "Loss" for GPs - the marginal log likelihood
# num_data refers to the amount of training data
mll = VariationalELBO(likelihood, model, train_y.numel())
# mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)


training_iter = 50000
for i in range(training_iter):

    def closure():
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        return loss
    l = optimizer.step(closure)
    # # Zero backpropped gradients from previous iteration
    # optimizer.zero_grad()
    # # Get predictive output
    # output = model(train_x)
    # # Calc loss and backprop gradients
    # loss = -mll(output, train_y)
    # loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, l.item()))
    # optimizer.step()

# Go into eval mode
model.eval()
likelihood.eval()

with torch.no_grad():
    # Test x are regularly spaced by 0.01 0,1 inclusive
    test_x = torch.linspace(0, 1, 101)
    # Get classification predictions
    observed_pred = likelihood(model(test_x))

    # Initialize fig and axes for plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
    # Get the predicted labels (probabilites of belonging to the positive class)
    # Transform these probabilities to be 0/1 labels
    pred_labels = observed_pred.mean.ge(0.5).float().mul(2).sub(1)
    ax.plot(test_x.numpy(), pred_labels.numpy(), 'b')
    ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])

plt.show()
