from typing import Callable

import cvxpy as cp
import numpy as np
import torch
from scipy import optimize
from scipy.sparse import csr_array


def _optim_minimize(loss: Callable, n_classes: int, method="SLSQP"):
    """
    Searches for the optimal prevalence values, i.e., an `n_classes`-dimensional vector of the (`n_classes`-1)-simplex
    that yields the smallest lost. This optimization is carried out by means of a constrained search using scipy's
    SLSQP routine.

    :param loss: (callable) the function to minimize
    :param n_classes: (int) the number of classes, i.e., the dimensionality of the prevalence vector
    :param method: (str) the method used by scipy.optimize to minimize the loss; default="SLSQP"
    :return: (ndarray) the best prevalence vector found
    """
    # the initial point is set as the uniform distribution
    uniform_distribution = np.full(fill_value=1 / n_classes, shape=(n_classes,))

    bounds = tuple((0, 1) for _ in range(n_classes))
    constraints = {"type": "eq", "fun": lambda x: 1 - sum(x)}

    # solutions are bounded to those contained in the unit-simplex
    r = optimize.minimize(loss, x0=uniform_distribution, method=method, bounds=bounds, constraints=constraints)
    return r.x


def _optim_cvxpy(A: csr_array | np.ndarray, b: np.ndarray):
    x = cp.Variable(A.shape[1])

    objective = cp.Minimize(0.5 * cp.sum_squares(A @ x - b))
    constraints = [x >= 0, x <= 1, cp.sum(x) == 1]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP)

    return x.value


def _optim_lsq_linear(A, b):
    # we impose the bounds of 0 <= x <= 1 for x
    bounds = (0, 1)

    r = optimize.lsq_linear(A, b, bounds=bounds)
    x = r.x

    # if x.sum() !=1 we normalize through L1 norm
    if not np.isclose(x.sum(), 1):
        x /= x.sum()

    return x


def _optim_Adam(A, b):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    A = torch.tensor(A, device=device)
    b = torch.tensor(b, device=device)
    n = A.shape[1]

    x = torch.full((n,), 1 / n, dtype=torch.double, device=device, requires_grad=True)
    lambda_lagrange = torch.nn.Parameter(torch.tensor(0.0, device=device))

    # hyperparameters
    lr_x = 5e-3
    lr_lambda = 1e-1
    num_iters = 5000

    x_opt = torch.optim.Adam([x], lr=lr_x)
    lambda_opt = torch.optim.SGD([lambda_lagrange], lr=lr_lambda)

    for i in range(num_iters):
        x_opt.zero_grad()
        lambda_opt.zero_grad()
        # loss
        loss = torch.norm(A @ x - b) ** 2
        # penalty to enforce x.sum() == 1 with lagrangian
        loss += loss + (lambda_lagrange.data * (x.sum() - 1))
        # step
        loss.backward()
        x_opt.step()
        lambda_opt.step()
        # apply bounds
        with torch.no_grad():
            x.clamp_(0, 1)

    x.data /= x.data.sum()

    return x.to("cpu").detach().numpy()


def _optim_Adam_batched(A, bs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    A = torch.tensor(A, device=device)
    bs = torch.tensor(bs, device=device)
    k, n = bs.shape[0], A.shape[1]

    xs = torch.full((k, n), 1 / n, dtype=torch.double, device=device, requires_grad=True)

    lambda_lagrange = torch.nn.Parameter(torch.zeros(k, device=device))

    # hyperparameters
    lr_xs = 5e-3
    lr_lambda = 1e-1
    num_iters = 5000

    xs_opt = torch.optim.Adam([xs], lr=lr_xs)
    lambda_opt = torch.optim.SGD([lambda_lagrange], lr=lr_lambda)

    for i in range(num_iters):
        xs_opt.zero_grad()
        lambda_opt.zero_grad()
        # losses for all batches
        Ax = torch.matmul(xs, A.T)
        losses = torch.norm(Ax - bs, dim=1) ** 2
        # constraint
        constraints = xs.sum(dim=1) - 1
        # cumulative loss
        loss = losses.sum() + (lambda_lagrange.data * constraints).sum()
        # step
        loss.backward()
        xs_opt.step()
        lambda_opt.step()
        # apply bounds
        with torch.no_grad():
            xs.clamp_(0, 1)

    out = xs.clone().detach()
    out = xs / torch.sum(torch.abs(xs), dim=1, keepdim=True)

    return out.to("cpu").detach().numpy()
