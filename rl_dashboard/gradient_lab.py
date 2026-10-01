"""The "Math Lab" gradient descent engine.

A real PPO policy network has thousands of parameters — its loss surface
can't be drawn. So instead of faking that, this module runs *actual*
gradient descent on a problem small enough to plot exactly: fitting a line
y = w*x + b to noisy synthetic data by minimizing mean-squared error.

Every number returned here (the data points, the loss surface, the descent
path) comes from real arithmetic, not a scripted animation — the same
`grad_w, grad_b -> w -= lr * grad_w` update that trains any RL value/policy
network, just in 2 dimensions instead of thousands.
"""

from __future__ import annotations

import numpy as np


def generate_data(n_points: int = 30, true_w: float = 2.0, true_b: float = -1.0, noise: float = 1.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-5, 5, size=n_points)
    y = true_w * x + true_b + rng.normal(0, noise, size=n_points)
    return x, y


def mse_loss(w: float, b: float, x: np.ndarray, y: np.ndarray) -> float:
    pred = w * x + b
    return float(np.mean((pred - y) ** 2))


def gradients(w: float, b: float, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """dL/dw and dL/db for MSE — the exact quantities gradient descent needs."""
    n = len(x)
    pred = w * x + b
    error = pred - y
    grad_w = float((2.0 / n) * np.sum(error * x))
    grad_b = float((2.0 / n) * np.sum(error))
    return grad_w, grad_b


def loss_surface(x: np.ndarray, y: np.ndarray, w_range=(-6, 6), b_range=(-6, 6), resolution: int = 40):
    ws = np.linspace(*w_range, resolution)
    bs = np.linspace(*b_range, resolution)
    grid = np.zeros((resolution, resolution))
    for i, b in enumerate(bs):
        for j, w in enumerate(ws):
            grid[i, j] = mse_loss(w, b, x, y)
    return ws.tolist(), bs.tolist(), grid.tolist()


def run_gradient_descent(
    learning_rate: float = 0.05,
    steps: int = 40,
    start_w: float = -4.0,
    start_b: float = 4.0,
    n_points: int = 30,
    true_w: float = 2.0,
    true_b: float = -1.0,
    noise: float = 1.0,
    seed: int = 0,
) -> dict:
    """Run real gradient descent and return everything needed to animate
    it in the browser: the data, the loss surface, and the exact path.
    """
    x, y = generate_data(n_points, true_w, true_b, noise, seed)

    w, b = start_w, start_b
    path = [{"step": 0, "w": w, "b": b, "loss": mse_loss(w, b, x, y)}]
    for step in range(1, steps + 1):
        grad_w, grad_b = gradients(w, b, x, y)
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b
        path.append({"step": step, "w": w, "b": b, "loss": mse_loss(w, b, x, y)})

    ws, bs, grid = loss_surface(x, y)

    return {
        "data": {"x": x.tolist(), "y": y.tolist()},
        "true_params": {"w": true_w, "b": true_b},
        "learned_params": {"w": w, "b": b},
        "surface": {"w": ws, "b": bs, "loss": grid},
        "path": path,
        "learning_rate": learning_rate,
    }


def run_1d_descent(learning_rate: float = 0.1, steps: int = 30, start_x: float = 8.0):
    """The simplest possible picture of gradient descent: minimizing
    f(x) = (x - 3)^2 by rolling downhill. f'(x) = 2(x - 3).
    """
    def f(x):
        return (x - 3.0) ** 2

    def fprime(x):
        return 2.0 * (x - 3.0)

    x = start_x
    path = [{"step": 0, "x": x, "loss": f(x)}]
    for step in range(1, steps + 1):
        x -= learning_rate * fprime(x)
        path.append({"step": step, "x": x, "loss": f(x)})

    xs = np.linspace(-2, 10, 200)
    curve = [{"x": float(xv), "loss": float(f(xv))} for xv in xs]

    return {"curve": curve, "path": path, "learning_rate": learning_rate}
