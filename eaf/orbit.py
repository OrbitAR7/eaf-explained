"""Two-body plus J2 dynamics, with the state transition matrix.

Units are km, km/s and s throughout. The state is x = (r, v) in an Earth-centred
inertial frame.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

MU = 398600.4418  # km^3/s^2
RE = 6378.137  # km
J2 = 1.08262668e-3
OMEGA_E = 7.292115e-5  # rad/s

_RTOL, _ATOL = 1e-11, 1e-12


def accel(r: np.ndarray) -> np.ndarray:
    """Acceleration of two-body gravity plus J2. r has shape (..., 3)."""
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    r2 = x * x + y * y + z * z
    rn = np.sqrt(r2)
    k = -MU / (r2 * rn)
    f = 1.5 * J2 * MU * RE**2 / (r2 * r2 * rn)
    zz = 5.0 * z * z / r2
    a = np.empty_like(r)
    a[..., 0] = k * x + f * x * (zz - 1.0)
    a[..., 1] = k * y + f * y * (zz - 1.0)
    a[..., 2] = k * z + f * z * (zz - 3.0)
    return a


def accel_jacobian(r: np.ndarray) -> np.ndarray:
    """d accel / d r for a single position, by central differences (3 x 3)."""
    h = 1e-6 * np.maximum(1.0, np.abs(r))
    D = np.diag(h)
    A = accel(np.vstack([r + D, r - D]))
    return (A[:3] - A[3:]).T / (2.0 * h)


def propagate(x: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """Propagate one state (6,) or a batch (N, 6) from t0 to t1."""
    x = np.asarray(x, dtype=float)
    if t1 == t0:
        return x.copy()
    shape = x.shape
    X = x.reshape(-1, 6)

    def rhs(_, y):
        Y = y.reshape(-1, 6)
        out = np.empty_like(Y)
        out[:, :3] = Y[:, 3:]
        out[:, 3:] = accel(Y[:, :3])
        return out.ravel()

    sol = solve_ivp(rhs, (t0, t1), X.ravel(), method="DOP853", rtol=_RTOL, atol=_ATOL)
    return sol.y[:, -1].reshape(shape)


def trajectory(x0: np.ndarray, t: np.ndarray) -> np.ndarray:
    """States at the times t (t[0] is the epoch of x0)."""
    def rhs(_, y):
        return np.concatenate([y[3:], accel(y[:3])])

    sol = solve_ivp(rhs, (t[0], t[-1]), np.asarray(x0, float), t_eval=t, method="DOP853",
                    rtol=_RTOL, atol=_ATOL)
    return sol.y.T


def propagate_stm(x: np.ndarray, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
    """Propagate a state and its 6 x 6 state transition matrix from t0 to t1."""
    if t1 == t0:
        return np.array(x, float), np.eye(6)

    def rhs(_, y):
        r, v = y[:3], y[3:6]
        Phi = y[6:].reshape(6, 6)
        A = np.zeros((6, 6))
        A[:3, 3:] = np.eye(3)
        A[3:, :3] = accel_jacobian(r)
        return np.concatenate([v, accel(r), (A @ Phi).ravel()])

    y0 = np.concatenate([np.asarray(x, float), np.eye(6).ravel()])
    sol = solve_ivp(rhs, (t0, t1), y0, method="DOP853", rtol=_RTOL, atol=_ATOL)
    y = sol.y[:, -1]
    return y[:6], y[6:].reshape(6, 6)


def circular_state(radius_km: float, raan_deg: float, inc_deg: float, arg_lat_deg: float) -> np.ndarray:
    """Inertial state of a circular orbit."""
    O, i, u = np.radians([raan_deg, inc_deg, arg_lat_deg])
    v = np.sqrt(MU / radius_km)
    cO, sO, ci, si, cu, su = np.cos(O), np.sin(O), np.cos(i), np.sin(i), np.cos(u), np.sin(u)
    r = radius_km * np.array([cO * cu - sO * su * ci, sO * cu + cO * su * ci, su * si])
    vv = v * np.array([-cO * su - sO * cu * ci, -sO * su + cO * cu * ci, cu * si])
    return np.concatenate([r, vv])
