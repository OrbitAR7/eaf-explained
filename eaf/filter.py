"""The Epistemic Abductive Filter (EAF), extended form for nonlinear models.

Our implementation of Algorithm 1 of M. K. Jah, "The Epistemic Abductive Filter:
Closed-Form Possibilistic Estimation, Its Optimality, and Its Kalman Form",
Preprints 202610.2608 (2026). Equation numbers are the preprint's. It is not the
author's code, which is not public.

The filter does not carry a probability distribution. It carries a support
ellipsoid E(c, P) = {x : (x - c)' P^-1 (x - c) <= 1}, the set of states that
are still possible given three declarations made by the user:

    D1  the initial state lies in E(c0, P0);
    D2  the unmodelled motion over each interval lies in an ellipsoid Q_bound;
    D3  each measurement error component is at most rho * r_i.

One epoch is

    1. prediction (eq. 4): c- = f(c), P- = (1+p) F P F' + (1+1/p) Q_bound,
       an outer bound of the Minkowski sum, with 1 + p = (1 + eps)^2;
    2. whitening (eq. 5): P- = L L', H~ = R^-1/2 H L, nu = R^-1/2 (y - h(c-)),
       with R = rho^2 diag(r)^2;
    3. the Schweppe family (eq. 6-7) of ellipsoids that contain the intersection
       of the predicted support with the measurement cut (eq. 3)
       C = {x : (y - Hx)' R^-1 (y - Hx) <= 1}. Its emptiness statistic
       delta* = sup_lambda delta_lambda tells whether that intersection is empty;
    4. if delta* > 1 a declaration is false. The filter explains the conflict
       (the "abductive" step) by a velocity impulse that could have acted since
       the last good epoch, and widens the support until delta* <= 1;
    5. the update is the least-volume member over lambda in [lambda_eps, 1]
       (eq. 8). Its centre is the estimate;
    6. an adaptive gate on rho: a miss (the predicted centre's residual leaves
       the cut) grows rho by g, a hit shrinks it slowly towards 1.

Where the preprint leaves a choice open, the choice made here is noted in the code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.optimize import minimize_scalar

Array = np.ndarray


@dataclass
class Model:
    """What the filters need to know about the system.

    propagate(x, t0, t1)      -> x at t1, for one state (n,) or a batch (N, n)
    propagate_stm(x, t0, t1)  -> (x at t1, state transition matrix)
    h(x, t)                   -> predicted measurement, for one state or a batch
    H(x, t)                   -> measurement Jacobian at x
    residual(z, zhat)         -> z - zhat (handles angle wrap)
    """

    n: int
    propagate: Callable
    propagate_stm: Callable
    h: Callable
    H: Callable
    residual: Callable = lambda z, zhat: np.asarray(z) - np.asarray(zhat)


def sym(A: Array) -> Array:
    return 0.5 * (A + A.T)


def chol(P: Array) -> Array:
    """Lower Cholesky factor, computed on the correlation matrix so that mixed
    units (km and km/s) do not hurt, with jitter as a fallback."""
    P = sym(np.asarray(P, float))
    d = np.sqrt(np.maximum(np.diag(P), 1e-300))
    C = P / np.outer(d, d)
    jit = 0.0
    for _ in range(8):
        try:
            return d[:, None] * np.linalg.cholesky(C + jit * np.eye(len(d)))
        except np.linalg.LinAlgError:
            jit = 1e-14 if jit == 0.0 else 100.0 * jit
    w, V = np.linalg.eigh(C)
    return d[:, None] * np.linalg.cholesky(sym((V * np.maximum(w, 1e-12)) @ V.T))


# --------------------------------------------------------------- Schweppe family
def _sigmoid(u):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(u, float)))


def _svd_coords(Ht: Array, nu: Array) -> tuple[Array, Array]:
    """Coordinates of nu in the left singular basis of H~, and the squared
    singular values padded with zeros to the measurement dimension."""
    U, s, _ = np.linalg.svd(Ht, full_matrices=True)
    s2 = np.zeros(Ht.shape[0])
    s2[: s.size] = s**2
    return U.T @ nu, s2


def _delta(lam, one_m, b, s2):
    """delta_lambda = lambda (1 - lambda) nu' S_lambda^-1 nu, vectorised over lambda."""
    lam, one_m = np.atleast_1d(lam), np.atleast_1d(one_m)
    return lam * one_m * np.sum(b**2 / (lam[:, None] + one_m[:, None] * s2[None, :]), axis=1)


def delta_star(Ht: Array, nu: Array, n_grid: int = 1500) -> float:
    """Emptiness statistic: sup over lambda in (0, 1) of delta_lambda.

    The predicted support and the measurement cut intersect if and only if
    delta* <= 1. delta_lambda is concave in lambda, so a grid in logit(lambda)
    followed by a bounded search in the best bracket finds the sup.
    """
    b, s2 = _svd_coords(np.atleast_2d(Ht), np.asarray(nu, float))
    u = np.linspace(-25.0, 25.0, n_grid)
    lam, one_m = _sigmoid(u), _sigmoid(-u)
    d = _delta(lam, one_m, b, s2)
    i = int(np.argmax(d))
    if i in (0, n_grid - 1):
        return float(d[i])
    res = minimize_scalar(lambda v: -float(_delta(_sigmoid(v), _sigmoid(-v), b, s2)[0]),
                          bounds=(u[i - 1], u[i + 1]), method="bounded", options={"xatol": 1e-9})
    return max(float(d[i]), -float(res.fun))


def schweppe_member(c: Array, L: Array, Ht: Array, nu: Array, lam: float) -> tuple[Array, Array]:
    """Member lambda of the Schweppe family (eq. 6-7): an ellipsoid that contains
    the intersection of E(c, L L') and the cut. lambda = 1 returns the prior."""
    n = L.shape[1]
    M = lam * np.eye(n) + (1.0 - lam) * Ht.T @ Ht
    Minv = np.linalg.inv(sym(M))
    u = (1.0 - lam) * Minv @ Ht.T @ nu
    S = lam * np.eye(Ht.shape[0]) + (1.0 - lam) * Ht @ Ht.T
    dl = min(lam * (1.0 - lam) * nu @ np.linalg.solve(S, nu), 1.0 - 1e-12)
    return c + L @ u, sym((1.0 - dl) * L @ Minv @ L.T)


def min_volume_lambda(Ht: Array, nu: Array, lam_eps: float, n_grid: int = 400) -> float:
    """argmin over lambda in [lam_eps, 1] of n log(1 - delta_lambda) - log det M_lambda (eq. 8)."""
    n = Ht.shape[1]
    b, s2 = _svd_coords(Ht, nu)
    k = min(s2.size, n)

    def J(lam):
        lam = np.atleast_1d(lam)
        dl = _delta(lam, 1.0 - lam, b, s2)
        logdet = np.sum(np.log(lam[:, None] + (1.0 - lam)[:, None] * s2[None, :k]), axis=1) + (n - k) * np.log(lam)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = n * np.log1p(-dl) - logdet
        return np.where(dl < 1.0, out, np.inf)

    grid = np.linspace(lam_eps, 1.0, n_grid)
    Jg = J(grid)
    i = int(np.argmin(Jg))
    if not np.isfinite(Jg[i]):
        return 1.0
    if 0 < i < n_grid - 1:
        res = minimize_scalar(lambda x: float(J(x)[0]), bounds=(grid[i - 1], grid[i + 1]),
                              method="bounded", options={"xatol": 1e-12})
        if np.isfinite(res.fun) and res.fun < Jg[i]:
            return float(res.x)
    return float(grid[i])


# -------------------------------------------------------------------- filter
class EAF:
    """Extended Epistemic Abductive Filter.

    Parameters
    ----------
    model : Model
    resolution : (m,) sensor resolution r per measurement component (D3).
    q_bound : callable (t0, t1) -> (n, n) shape of the D2 ellipsoid over [t0, t1].
    vel_idx : indices of the velocity components (used by the widening impulse).
    eps, alpha, gate_gain, rho0 : preprint constants (tolerance 0.002, miss budget
        0.05, gate gain 1.2, initial cut multiple 3).
    dv0, gw, dv_max : widening impulse start (5 cm/s), growth per repeat (1.2)
        and cap (20 m/s), in the velocity units of the state.
    """

    def __init__(self, model: Model, resolution, q_bound: Callable, vel_idx=(3, 4, 5),
                 eps: float = 0.002, alpha: float = 0.05, gate_gain: float = 1.2, rho0: float = 3.0,
                 dv0: float = 5e-5, gw: float = 1.2, dv_max: float = 0.02):
        self.model = model
        self.r = np.asarray(resolution, float)
        self.q_bound = q_bound
        self.vel_idx = list(vel_idx)
        self.alpha, self.g, self.rho0 = alpha, gate_gain, rho0
        self.dv0, self.gw, self.dv_max = dv0, gw, dv_max
        self.p = (1.0 + eps) ** 2 - 1.0
        self.lam_eps = (1.0 + eps) ** -2
        self.diag: dict = {}

    def initialize(self, c0: Array, P0: Array, t0: float) -> None:
        """Start from the declared support E(c0, P0) (declaration D1)."""
        self.c, self.P, self.t = np.array(c0, float), sym(np.array(P0, float)), float(t0)
        self.rho = self.rho0
        self.Phi = np.eye(self.model.n)  # transition matrix since the last hit
        self.n_rep = 0  # widening repeats, carried across consecutive conflicts

    def predict(self, t: float) -> None:
        if t == self.t:
            return
        c, F = self.model.propagate_stm(self.c, self.t, t)
        self.P = sym((1.0 + self.p) * F @ self.P @ F.T + (1.0 + 1.0 / self.p) * self.q_bound(self.t, t))
        self.c, self.Phi, self.t = c, F @ self.Phi, float(t)

    def _whiten(self, P, H, res):
        L = chol(P)
        s = 1.0 / (self.rho * self.r)
        return L, (H @ L) * s[:, None], res * s

    def update(self, z: Array) -> None:
        m, t = self.model, self.t
        H = m.H(self.c, t)
        res = m.residual(z, m.h(self.c, t))
        L, Ht, nu = self._whiten(self.P, H, res)
        ds = delta_star(Ht, nu)
        miss = bool(np.linalg.norm(nu) > 1.0)
        widened = ds > 1.0

        if widened:
            # The prior and the measurement cannot both be right. The preprint explains
            # the conflict by an impulse; it does not say how the impulse enters the
            # support. Here: an extra velocity uncertainty of size dv at the last hit,
            # mapped to now by the transition matrix since then.
            Phi_v = self.Phi[:, self.vel_idx]
            A = (H @ Phi_v) / self.r[:, None]
            dv_implied = float(np.linalg.norm(np.linalg.lstsq(A, res / self.r, rcond=1e-10)[0]))
            dv = min(self.dv_max, max(self.dv0, 2.0 * dv_implied) * self.gw**self.n_rep)
            P0 = self.P
            while True:
                P = sym(P0 + (1.0 + 1.0 / self.p) * dv**2 * Phi_v @ Phi_v.T)
                L, Ht, nu = self._whiten(P, H, res)
                ds = delta_star(Ht, nu)
                self.n_rep += 1
                if ds <= 1.0 or dv >= self.dv_max:
                    break
                dv = min(self.dv_max, dv * self.gw)
            # Last resort (not in the preprint): if even the largest impulse does not
            # explain the data, inflate the support until it does.
            for _ in range(60):
                if ds <= 1.0:
                    break
                P = 2.0 * P
                L, Ht, nu = self._whiten(P, H, res)
                ds = delta_star(Ht, nu)
        else:
            self.n_rep = 0

        lam = min_volume_lambda(Ht, nu, self.lam_eps)
        self.c, self.P = schweppe_member(self.c, L, Ht, nu, lam)

        if widened:
            pass  # gate held
        elif miss:
            self.rho *= self.g
        else:
            self.rho = max(self.rho / self.g ** (self.alpha / (1.0 - self.alpha)), 1.0)
            self.Phi = np.eye(m.n)
        self.diag = {"delta_star": ds, "widened": widened, "miss": miss, "rho": self.rho, "lambda": lam}

    def region(self) -> tuple[Array, Array]:
        """Centre and shape of the support."""
        return self.c.copy(), self.P.copy()
