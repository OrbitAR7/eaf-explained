"""Checks of the EAF's linear algebra against the relations in the preprint.

    python -m pytest tests
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eaf.filter import chol, delta_star, min_volume_lambda, schweppe_member  # noqa: E402

rng = np.random.default_rng(0)


def random_problem(n=4, m=2, scale=1.0):
    A = rng.standard_normal((n, n))
    P = A @ A.T + n * np.eye(n)
    H = rng.standard_normal((m, n))
    c = rng.standard_normal(n)
    r = np.full(m, 0.5)
    y = H @ c + scale * rng.standard_normal(m)
    L = chol(P)
    return c, P, L, H, r, y, (H @ L) / r[:, None], (y - H @ c) / r


def test_kalman_form():
    """Corollary 3: the centre is a Kalman update with R replaced by R / kappa, kappa = (1 - lam) / lam,
    and the shape is ((1 - delta) / lam) (I - K H) P."""
    c, P, L, H, r, y, Ht, nu = random_problem()
    lam = 0.7
    cu, Pu = schweppe_member(c, L, Ht, nu, lam)
    R = np.diag(r**2)
    kappa = (1 - lam) / lam
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R / kappa)
    S = lam * np.eye(2) + (1 - lam) * Ht @ Ht.T
    d = lam * (1 - lam) * nu @ np.linalg.solve(S, nu)
    assert np.allclose(cu, c + K @ (y - H @ c))
    assert np.allclose(Pu, (1 - d) / lam * (np.eye(4) - K @ H) @ P)


def test_member_contains_intersection():
    """Points in both the prior support and the cut lie in every Schweppe member."""
    c, P, L, H, r, y, Ht, nu = random_problem(scale=0.3)
    X = c + (L @ rng.uniform(-1, 1, (4, 200000))).T
    inP = np.einsum("ki,ij,kj->k", X - c, np.linalg.inv(P), X - c) <= 1
    inC = np.sum(((y - X @ H.T) / r) ** 2, axis=1) <= 1
    both = X[inP & inC]
    assert both.shape[0] > 10
    for lam in (0.1, 0.5, 0.9, min_volume_lambda(Ht, nu, 0.996)):
        cu, Pu = schweppe_member(c, L, Ht, nu, lam)
        q = np.einsum("ki,ij,kj->k", both - cu, np.linalg.inv(Pu), both - cu)
        assert q.max() <= 1 + 1e-9


def test_emptiness():
    """delta* > 1 exactly when the support and the cut do not meet (checked by sampling the cut)."""
    for scale in (0.2, 5.0, 20.0):
        c, P, L, H, r, y, Ht, nu = random_problem(scale=scale)
        # smallest prior Mahalanobis distance over the cut: minimise (x-c)'P^-1(x-c) s.t. ||(y-Hx)/r|| <= 1
        X = c + (L @ rng.standard_normal((4, 400000)) * 3).T
        inC = np.sum(((y - X @ H.T) / r) ** 2, axis=1) <= 1
        hit = inC.any() and np.einsum("ki,ij,kj->k", X[inC] - c, np.linalg.inv(P), X[inC] - c).min() <= 1
        if delta_star(Ht, nu) > 1.05:
            assert not hit
        elif hit:
            assert delta_star(Ht, nu) <= 1.0
