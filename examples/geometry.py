"""One EAF update drawn in two dimensions.

State (position, velocity), one position measurement. Panel (a): the predicted
support and the measurement cut overlap; the Schweppe family gives ellipsoids
that contain the overlap and the filter keeps the one of least volume. Panel (b):
they do not overlap, so a declaration is false; the filter widens the support by
a velocity impulse until they do. Panel (c): the emptiness statistic delta_lambda
for both cases.

    python examples/geometry.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eaf.filter import chol, delta_star, min_volume_lambda, schweppe_member  # noqa: E402
from eaf import plotting as pl  # noqa: E402

H = np.array([[1.0, 0.0]])
R_CUT = 0.04  # half-width of the cut (rho * r): much narrower than the support, as in orbit work
PHI_V = np.array([[1.0], [1.0]])  # an impulse one time unit ago moves position and velocity alike
EPS = 0.002
P_FAC = (1 + EPS) ** 2 - 1
LAM_EPS = (1 + EPS) ** -2


def ellipse(c, P, n=300):
    a = np.linspace(0, 2 * np.pi, n)
    return (c[:, None] + chol(P) @ np.vstack([np.cos(a), np.sin(a)])).T


def whiten(c, P, y):
    L = chol(P)
    return L, (H @ L) / R_CUT, (y - H @ c) / R_CUT


def delta_curve(Ht, nu, lam):
    s2 = float((Ht @ Ht.T)[0, 0])
    return lam * (1 - lam) * float(nu @ nu) / (lam + (1 - lam) * s2)


def draw_cut(ax, y):
    ax.axvspan(y - R_CUT, y + R_CUT, color=pl.ORANGE, alpha=0.12, lw=0)
    for s in (-1, 1):
        ax.axvline(y + s * R_CUT, color=pl.ORANGE, lw=0.8)


def main():
    pl.style()
    c = np.array([0.0, 0.0])
    P = np.array([[1.0, 0.45], [0.45, 0.8]])
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.3))

    # (a) consistent
    ax = axes[0]
    y_a = np.array([0.6])
    draw_cut(ax, y_a[0])
    L, Ht, nu = whiten(c, P, y_a)
    for lam in (0.2, 0.5, 0.8):
        cm, Pm = schweppe_member(c, L, Ht, nu, lam)
        ax.plot(*ellipse(cm, Pm).T, color=pl.GRAY, lw=0.6, ls="--")
    lam = min_volume_lambda(Ht, nu, LAM_EPS)
    cu, Pu = schweppe_member(c, L, Ht, nu, lam)
    ax.plot(*ellipse(c, P).T, color=pl.BLUE, label="predicted support")
    ax.plot(*ellipse(cu, Pu).T, color=pl.GREEN, lw=1.6, label=f"update, $\\lambda$ = {lam:.4f}")
    ax.plot([], [], color=pl.GRAY, lw=0.6, ls="--", label="other family members")
    ax.plot([], [], color=pl.ORANGE, lw=0.8, label="measurement cut")
    ax.plot(*c, "+", color=pl.BLUE)
    ax.plot(*cu, "+", color=pl.GREEN)
    ax.set_xlabel("position")
    ax.set_ylabel("velocity")
    pl.panel(ax, "a", f"cut reachable, $\\delta^*$ = {delta_star(Ht, nu):.2f}")
    ax.legend(loc="lower left")

    # (b) conflict -> widening
    ax = axes[1]
    y_b = np.array([1.6])
    draw_cut(ax, y_b[0])
    L, Ht, nu = whiten(c, P, y_b)
    ds0 = delta_star(Ht, nu)
    dv = 0.05
    while True:
        Pw = P + (1 + 1 / P_FAC) * dv**2 * PHI_V @ PHI_V.T
        Lw, Htw, nuw = whiten(c, Pw, y_b)
        if delta_star(Htw, nuw) <= 1.0:
            break
        dv *= 1.2
    lam = min_volume_lambda(Htw, nuw, LAM_EPS)
    cu, Pu = schweppe_member(c, Lw, Htw, nuw, lam)
    ax.plot(*ellipse(c, P).T, color=pl.BLUE, label="predicted support")
    ax.plot(*ellipse(c, Pw).T, color=pl.RED, ls="-.", label=f"widened by an impulse")
    ax.plot(*ellipse(cu, Pu).T, color=pl.GREEN, lw=1.6, label="update")
    ax.plot(*cu, "+", color=pl.GREEN)
    ax.set_xlabel("position")
    pl.panel(ax, "b", f"cut out of reach, $\\delta^*$ = {ds0:.2f}")
    ax.legend(loc="lower left")
    for a in axes[:2]:
        a.set_xlim(-3.5, 3.5)
        a.set_ylim(-3.5, 3.5)
        a.set_aspect("equal")

    # (c) delta_lambda
    ax = axes[2]
    lam_g = np.linspace(1e-4, 1 - 1e-4, 400)
    for y, col, lab in ((y_a, pl.BLUE, "case (a)"), (y_b, pl.RED, "case (b), before widening")):
        _, Ht, nu = whiten(c, P, y)
        ax.plot(lam_g, delta_curve(Ht, nu, lam_g), color=col, label=lab)
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.text(0.03, 1.0, "  above: empty intersection", va="bottom", fontsize=7.5, transform=ax.get_yaxis_transform())
    ax.set_xlabel("$\\lambda$")
    ax.set_ylabel("$\\delta_\\lambda$")
    ax.set_xlim(0, 1)
    ax.set_yscale("log"); ax.set_ylim(1e-2, 10)
    pl.panel(ax, "c", "emptiness statistic")
    ax.legend(loc="lower right")

    fig.tight_layout()
    pl.save(fig, ROOT / "figures" / "fig1_geometry")


if __name__ == "__main__":
    main()
