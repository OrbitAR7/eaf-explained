"""Figures 2-4 from results/sweep.json and results/curves.npz (run sweep.py first)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
from eaf import plotting as pl  # noqa: E402

COL = {"UKF": pl.BLUE, "UKF told of burn": pl.PURPLE, "EAF": pl.ORANGE}
STY = {"UKF": "-", "UKF told of burn": "--", "EAF": "-"}


def time_figure(C, setting, dv, stem, err_log=False):
    t = C["t_h"]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
    for name in COL:
        E = C[f"{setting}|{dv}|{name}|err"]
        I = C[f"{setting}|{dv}|{name}|inside"]
        axes[0].plot(t, np.median(E, axis=0), STY[name], color=COL[name], label=name)
        axes[1].plot(t, 100 * I.mean(axis=0), STY[name], color=COL[name], label=name)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("median position error (km)")
    axes[1].set_ylabel("runs with the truth inside (%)")
    axes[1].set_ylim(-3, 103)
    for ax in axes:
        ax.set_xlabel("time since first measurement (h)")
        ax.set_xlim(0, t[-1])
    pl.panel(axes[0], "a", "position error")
    pl.panel(axes[1], "b", "containment")
    axes[1].legend(loc="upper right", bbox_to_anchor=(0.98, 0.88))
    fig.tight_layout()
    pl.save(fig, stem)


def single_run_figure(stem, dv=3.0, seed=1):
    """One run of the quiet-burn scenario (the same one `python examples/quiet_burn.py 3 1` prints)."""
    import quiet_burn as qb
    case = qb.make_case(dv, np.random.default_rng(seed))
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
    for name, (f, P0) in qb.make_filters().items():
        r = qb.run(f, P0, case)
        t = r["t_h"] - r["t_h"][0]
        axes[0].plot(t, r["err_km"], STY[name], color=COL[name], label=name)
        axes[1].plot(t, r["ratio"], STY[name], color=COL[name], label=name)
        if name == "EAF":
            for tw in t[r["widened"]]:
                for ax in axes:
                    ax.axvline(tw, color=pl.ORANGE, lw=0.6, ls=":")
    axes[1].axhline(1.0, color="k", lw=0.8, ls="--")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel("time since first measurement (h)")
        ax.set_xlim(0, 8)
    axes[0].set_ylabel("position error (km)")
    axes[1].set_ylabel("error / distance to boundary")
    pl.panel(axes[0], "a", "position error")
    pl.panel(axes[1], "b", "truth outside the region above 1")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    pl.save(fig, stem)


def sweep_figure(S, stem):
    rows = S["gap12"]
    dvs = np.array([float(k) for k in rows])
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 2.9))
    for name in COL:
        kw = dict(color=COL[name], ls=STY[name], marker="o", ms=3.5, label=name)
        axes[0].plot(dvs, [rows[k][name]["inside_pct"] for k in rows], **kw)
        axes[1].plot(dvs, [rows[k][name]["rms_km_median"] for k in rows], **kw)
        axes[2].plot(dvs, [rows[k][name]["halfwidth_km_median"] for k in rows], **kw)
    axes[0].set_ylabel("epochs with the truth inside (%)")
    axes[0].set_ylim(-3, 103)
    axes[1].set_ylabel("median RMS position error (km)")
    axes[2].set_ylabel("median largest semi-axis (km)")
    for ax in axes[1:]:
        ax.set_yscale("log")
    for ax in axes:
        ax.set_xscale("symlog", linthresh=0.3, linscale=0.5)
        ax.set_xlim(0, 11)
        ax.set_xticks([0, 0.3, 1, 2, 3, 5, 10])
        ax.set_xticklabels(["0", "0.3", "1", "2", "3", "5", "10"])
        ax.set_xlabel("burn $\\Delta v$ (m/s)")
    pl.panel(axes[0], "a", "containment")
    pl.panel(axes[1], "b", "accuracy")
    pl.panel(axes[2], "c", "size of the region")
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    pl.save(fig, stem)


def main():
    pl.style()
    S = json.loads((ROOT / "results" / "sweep.json").read_text())["results"]
    C = np.load(ROOT / "results" / "curves.npz")
    single_run_figure(ROOT / "figures" / "fig2_quiet_burn_3ms")
    sweep_figure(S, ROOT / "figures" / "fig3_burn_sweep")
    time_figure(C, "gap0", 3.0, ROOT / "figures" / "fig4_no_gap_3ms")


if __name__ == "__main__":
    main()
