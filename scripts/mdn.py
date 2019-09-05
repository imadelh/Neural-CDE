import numpy as np
import scipy as sp

import torch
import torch.nn as nn


class MDN_network(nn.Module):
    """
    nn module that take x as input and outputs (w, sigma, mu) parameters of 
    the mixture model for CDE
    """
    def __init__(self, n_hidden, n_layers, n_gaussians, dropout):
        super().__init__()
        self.h = nn.Sequential(nn.Linear(1, n_hidden), nn.Tanh())

        hidden_layers = list()
        for i in range(n_layers):
            hidden_layers.append(nn.Linear(n_hidden, n_hidden))
            hidden_layers.append(nn.Dropout(dropout))
            hidden_layers.append(nn.ReLU())

        self.hidden_layers = nn.Sequential(*hidden_layers)

        self.z_w = nn.Linear(n_hidden, n_gaussians)
        self.z_sigma = nn.Linear(n_hidden, n_gaussians)
        self.z_mu = nn.Linear(n_hidden, n_gaussians)

    def forward(self, x):
        x = self.h(x)
        x = self.hidden_layers(x)

        w = nn.functional.softmax(self.z_w(x), -1)
        sigma = torch.exp(self.z_sigma(x))
        mu = self.z_mu(x)
        return w, sigma, mu


# Define Loss function

cte = 1.0 / np.sqrt(2.0 * np.pi)


def gaussian_distribution(y, mu, sigma):

    result = (y.expand_as(mu) - mu) * torch.reciprocal(sigma)
    result = -0.5 * (result * result)
    return (torch.exp(result) * torch.reciprocal(sigma)) * cte


def mdn_loss_fn(w, sigma, mu, y):

    result = gaussian_distribution(y, mu, sigma) * w
    result = torch.sum(result, dim=1)
    result = -torch.log(result)
    return torch.mean(result)


# Train function for 1Epoch


def train_epoch(model, data, optimizer):
    model.train()

    # get data
    x, y = data[0], data[1]

    optimizer.zero_grad()

    w_variable, sigma_variable, mu_variable = model(x)

    loss = mdn_loss_fn(w_variable, sigma_variable, mu_variable, y)

    loss.backward()

    optimizer.step()

    # Validate : Skipped because of computation cost in live version
    # with torch.no_grad():
    #  x_val, y_val = data_val
    #  w_variable, sigma_variable, mu_variable = model(x)
    #  loss_val = mdn_loss_fn(w_variable, sigma_variable, mu_variable, y)

    return loss.item()


def simulate_condprob_trained(X, model, ys):

    """
    Sampling estimated cond density
    """

    model.eval()
    x_test_data = torch.from_numpy(np.float32([X])).reshape(1, 1)

    w, sigma, mu = model(x_test_data)

    n_kernels = sigma.shape[1]

    w_n = w.data.numpy()
    sigma_n = sigma.data.numpy()
    mu_n = mu.data.numpy()

    # construct k componenets
    k_densities = []

    for i in range(n_kernels):
        k_densities.append(
            sp.stats.multivariate_normal(mean=mu_n[0][i], cov=sigma_n[0][i])
        )

    P_y = np.stack(
        [k_densities[i].pdf(ys) for i in range(n_kernels)], axis=1
    )

    cond_prob = np.sum(np.multiply(w_n, P_y), axis=1)

    return cond_prob
