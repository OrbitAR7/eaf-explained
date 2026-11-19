"""A ground telescope that measures topocentric right ascension and declination."""

from __future__ import annotations

import numpy as np

from .orbit import OMEGA_E, RE

ARCSEC = np.pi / (180.0 * 3600.0)


def wrap(a):
    """Wrap an angle to (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


class Telescope:
    """Optical site on a spherical rotating Earth. The inertial and Earth-fixed
    frames coincide at t = 0."""

    def __init__(self, lat_deg: float, lon_deg: float):
        self.lat, self.lon = np.radians(lat_deg), np.radians(lon_deg)

    def position(self, t: float) -> np.ndarray:
        th = self.lon + OMEGA_E * t
        return RE * np.array([np.cos(self.lat) * np.cos(th), np.cos(self.lat) * np.sin(th), np.sin(self.lat)])

    def radec(self, x: np.ndarray, t: float) -> np.ndarray:
        """(RA, Dec) in radians of the state x (6,) or states (N, 6) at time t."""
        rho = np.asarray(x)[..., :3] - self.position(t)
        ra = np.arctan2(rho[..., 1], rho[..., 0])
        dec = np.arcsin(rho[..., 2] / np.linalg.norm(rho, axis=-1))
        return np.stack([ra, dec], axis=-1)

    def jacobian(self, x: np.ndarray, t: float) -> np.ndarray:
        """d(RA, Dec)/dx, 2 x 6."""
        rho = np.asarray(x)[:3] - self.position(t)
        px, py, pz = rho
        s2 = px * px + py * py
        r2 = s2 + pz * pz
        s = np.sqrt(s2)
        H = np.zeros((2, 6))
        H[0, :3] = [-py / s2, px / s2, 0.0]
        H[1, :3] = [-px * pz / (r2 * s), -py * pz / (r2 * s), s / r2]
        return H

    @staticmethod
    def residual(z: np.ndarray, zhat: np.ndarray) -> np.ndarray:
        d = np.asarray(z, float) - np.asarray(zhat, float)
        d[..., 0] = wrap(d[..., 0])
        return d
