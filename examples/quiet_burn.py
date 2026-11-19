"""The quiet-burn scenario.

A geostationary satellite makes a manoeuvre that nobody announces. The catalogue
still holds its pre-burn state and covariance, which are now wrong by the burn.
That night a ground telescope picks the satellite up again and every filter has
to start from the stale catalogue entry.

Truth      two-body + J2, no other perturbation (the filters use the same model)
Satellite  GEO, circular, 0.05 deg inclination, 15 deg east of the site
Catalogue  1 km per position axis, 0.1 m/s per velocity axis (1 sigma)
Burn       magnitude dv in a random direction, right after the catalogue epoch
Gap        12 h from the catalogue epoch to the first measurement (0 h in the
           "no gap" variant); the filters predict across it in 120 s steps
Sensor     topocentric RA/Dec from one site at 32 deg N, every 120 s for 8 h,
           errors N(0, 1 arcsec) truncated at 3 sigma

    python examples/quiet_burn.py [dv m/s] [seed] [gap h]     # one run, prints a summary
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eaf import EAF, UKF, Model  # noqa: E402
from eaf.orbit import circular_state, propagate, propagate_stm, trajectory  # noqa: E402
from eaf.sensor import ARCSEC, Telescope  # noqa: E402

SITE = Telescope(lat_deg=32.0, lon_deg=0.0)
X_NOMINAL = circular_state(42164.17, raan_deg=0.0, inc_deg=0.05, arg_lat_deg=15.0)
P_CAT = np.diag([1.0, 1.0, 1.0, 1e-8, 1e-8, 1e-8])  # (1 km)^2 and (0.1 m/s)^2
SIGMA_ANGLE = 1.0 * ARCSEC
CADENCE_S = 120.0
GAP_S = 12 * 3600.0  # catalogue epoch to the first measurement
TRACK_S = 8 * 3600.0  # length of tonight's track
SIGMA_ACC = 1e-10  # km/s^2, the UKF's white process noise
K0 = float(np.sqrt(chi2.ppf(0.9973, 6)))  # 4.48: the 99.73 % ellipsoid of the catalogue
CHI2_3 = float(chi2.ppf(0.9973, 3))
TOLD_DV = 0.005  # km/s: the "told" UKF knows a burn of up to 5 m/s may have happened (3 sigma)

MODEL = Model(
    n=6,
    propagate=propagate,
    propagate_stm=propagate_stm,
    h=lambda x, t: SITE.radec(x, t),
    H=lambda x, t: SITE.jacobian(x, t),
    residual=Telescope.residual,
)


def q_white(t0, t1, s=SIGMA_ACC):
    dt = t1 - t0
    I = np.eye(3)
    return s**2 * np.block([[dt**3 / 3 * I, dt**2 / 2 * I], [dt**2 / 2 * I, dt * I]])


def q_bound(t0, t1, a_max=3 * SIGMA_ACC):
    """D2: a constant acceleration of at most a_max per axis over the interval. The
    displacements fill the image of a box, enclosed by the ellipsoid 3 a_max^2 G G'."""
    dt = t1 - t0
    G = np.vstack([0.5 * dt * dt * np.eye(3), dt * np.eye(3)])
    return 3.0 * a_max**2 * G @ G.T


def truncated_normal(rng, size, k=3.0):
    x = rng.standard_normal(size)
    while (bad := np.abs(x) > k).any():
        x[bad] = rng.standard_normal(int(bad.sum()))
    return x


@dataclass
class Case:
    t: np.ndarray
    z: np.ndarray
    x_true: np.ndarray
    x_cat: np.ndarray  # catalogue estimate at t = 0 (pre-burn)
    dv: np.ndarray  # the burn, km/s


def make_case(dv_ms: float, rng: np.random.Generator, gap_s: float = GAP_S) -> Case:
    """Catalogue epoch t = 0, burn right after it, measurements from gap_s to gap_s + TRACK_S.
    Rows of z before gap_s are NaN (no measurement; the filters only predict)."""
    t = np.arange(0.0, gap_s + TRACK_S + 1.0, CADENCE_S)
    d = rng.standard_normal(3)
    dv = dv_ms * 1e-3 * d / np.linalg.norm(d)
    x0 = X_NOMINAL.copy()
    x0[3:] += dv
    x_true = trajectory(x0, t)
    z = np.array([SITE.radec(x, tk) for x, tk in zip(x_true, t)])
    z += SIGMA_ANGLE * truncated_normal(rng, z.shape)
    z[t < gap_s] = np.nan
    x_cat = X_NOMINAL + np.sqrt(np.diag(P_CAT)) * truncated_normal(rng, 6)
    return Case(t, z, x_true, x_cat, dv)


def make_filters() -> dict:
    R = SIGMA_ANGLE**2 * np.eye(2)
    told = P_CAT.copy()
    told[3:, 3:] = np.eye(3) * (TOLD_DV / 3.0) ** 2
    return {
        "UKF": (UKF(MODEL, R, q_white), P_CAT),
        "UKF told of burn": (UKF(MODEL, R, q_white), told),
        "EAF": (EAF(MODEL, resolution=np.full(2, 3.0 * SIGMA_ANGLE * np.sqrt(2.0)), q_bound=q_bound),
                K0**2 * P_CAT),
    }


def run(filt, P0: np.ndarray, case: Case) -> dict:
    """Run one filter. Containment: the truth inside the EAF support, or inside the
    99.73 % position ellipsoid of a UKF."""
    K = case.t.size
    est, Spos = np.empty((K, 6)), np.empty((K, 3, 3))
    widened = np.zeros(K, bool)
    filt.initialize(case.x_cat, P0, case.t[0])
    for k in range(K):
        filt.predict(case.t[k])
        if np.isfinite(case.z[k, 0]):
            filt.update(case.z[k])
        c, P = filt.region()
        est[k] = c
        Spos[k] = P[:3, :3] if isinstance(filt, EAF) else CHI2_3 * P[:3, :3]
        if isinstance(filt, EAF) and np.isfinite(case.z[k, 0]):
            widened[k] = filt.diag["widened"]
    err = est[:, :3] - case.x_true[:, :3]
    q = np.einsum("ki,ki->k", err, np.linalg.solve(Spos, err[..., None])[..., 0])
    trk = np.isfinite(case.z[:, 0])  # report the tracked part only
    err, q, Spos, widened = err[trk], q[trk], Spos[trk], widened[trk]
    return {
        "t_h": case.t[trk] / 3600.0,
        "err_km": np.linalg.norm(err, axis=1),
        "inside": q <= 1.0,
        "ratio": np.sqrt(np.maximum(q, 0.0)),
        "halfwidth_km": np.sqrt(np.maximum(np.linalg.eigvalsh(Spos)[:, -1], 0.0)),
        "widened": widened,
    }


if __name__ == "__main__":
    import time

    dv_ms = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    gap_h = float(sys.argv[3]) if len(sys.argv) > 3 else GAP_S / 3600.0
    case = make_case(dv_ms, np.random.default_rng(seed), gap_h * 3600.0)
    for name, (f, P0) in make_filters().items():
        t0 = time.time()
        r = run(f, P0, case)
        e = r["err_km"]
        print(f"{name:18s} rms {np.sqrt(np.mean(e**2)):8.2f} km  final {e[-1]:8.3f} km  inside {100*r['inside'].mean():5.1f} %"
              f"  first-hour inside {100*r['inside'][:30].mean():5.1f} %  median halfwidth {np.median(r['halfwidth_km']):8.2f} km"
              f"  widened {r['widened'].sum():3d}  ({time.time()-t0:.1f} s)")
