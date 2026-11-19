"""Unscented Kalman filter (scaled sigma points, additive noise), the baseline."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .filter import Model, chol, sym


class UKF:
    def __init__(self, model: Model, R: np.ndarray, Q: Callable, alpha: float = 1.0, beta: float = 2.0,
                 kappa: float = 0.0):
        self.model, self.R, self.Q = model, np.asarray(R, float), Q
        n = model.n
        self.lam = alpha**2 * (n + kappa) - n
        self.Wm = np.full(2 * n + 1, 0.5 / (n + self.lam))
        self.Wc = self.Wm.copy()
        self.Wm[0] = self.lam / (n + self.lam)
        self.Wc[0] = self.Wm[0] + 1.0 - alpha**2 + beta

    def initialize(self, x0, P0, t0) -> None:
        self.x, self.P, self.t = np.array(x0, float), sym(np.array(P0, float)), float(t0)

    def _sigma(self):
        S = np.sqrt(self.model.n + self.lam) * chol(self.P)
        return np.vstack([self.x, self.x + S.T, self.x - S.T])

    def predict(self, t: float) -> None:
        if t == self.t:
            return
        X = self.model.propagate(self._sigma(), self.t, t)
        self.x = self.Wm @ X
        d = X - self.x
        self.P = sym((self.Wc[:, None] * d).T @ d + self.Q(self.t, t))
        self.t = float(t)

    def update(self, z) -> None:
        X = self._sigma()
        Z = self.model.h(X, self.t)
        # mean of the predicted measurement, relative to the centre point (angle wrap)
        dZ = self.model.residual(Z, Z[0])
        zhat_off = self.Wm @ dZ
        dz = dZ - zhat_off
        S = sym((self.Wc[:, None] * dz).T @ dz + self.R)
        Pxz = (self.Wc[:, None] * (X - self.x)).T @ dz
        K = np.linalg.solve(S, Pxz.T).T
        y = self.model.residual(z, Z[0]) - zhat_off
        self.x = self.x + K @ y
        self.P = sym(self.P - K @ S @ K.T)
        self.nis = float(y @ np.linalg.solve(S, y))

    def region(self):
        return self.x.copy(), self.P.copy()
