"""Monte Carlo of the quiet-burn scenario over the size of the burn.

Two settings:
  gap12  the catalogue entry is 12 h old when tracking starts (the main case);
  gap0   the burn happens just before tracking starts.

Writes results/sweep.json (summary) and results/curves.npz (per-epoch curves).

    python examples/sweep.py            # about 10 minutes on 10 cores
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quiet_burn as qb  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
N_RUNS = 20
SETTINGS = {
    "gap12": (12 * 3600.0, [0.0, 0.3, 1.0, 2.0, 3.0, 5.0, 10.0]),
    "gap0": (0.0, [0.0, 1.0, 3.0, 10.0]),
}


def one(args):
    setting, gap_s, dv, i = args
    seed = 1000 * int(round(10 * dv)) + i + (0 if setting == "gap12" else 500000)
    case = qb.make_case(dv, np.random.default_rng(seed), gap_s)
    out = {}
    for name, (f, P0) in qb.make_filters().items():
        r = qb.run(f, P0, case)
        out[name] = {k: r[k] for k in ("err_km", "inside", "halfwidth_km", "widened")}
    return setting, dv, i, out


def main():
    OUT.mkdir(exist_ok=True)
    jobs = [(s, gap, dv, i) for s, (gap, dvs) in SETTINGS.items() for dv in dvs for i in range(N_RUNS)]
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as ex:
        results = list(ex.map(one, jobs, chunksize=1))

    summary, curves = {}, {}
    for s, (gap, dvs) in SETTINGS.items():
        summary[s] = {}
        for dv in dvs:
            runs = [r[3] for r in results if r[0] == s and r[1] == dv]
            summary[s][str(dv)] = {}
            for name in runs[0]:
                E = np.array([r[name]["err_km"] for r in runs])
                I = np.array([r[name]["inside"] for r in runs])
                W = np.array([r[name]["halfwidth_km"] for r in runs])
                rms = np.sqrt(np.mean(E**2, axis=1))
                summary[s][str(dv)][name] = {
                    "inside_pct": float(100 * I.mean()),
                    "inside_first_hour_pct": float(100 * I[:, :30].mean()),
                    "rms_km_median": float(np.median(rms)),
                    "final_km_median": float(np.median(E[:, -1])),
                    "runs_final_over_5km": int(np.sum(E[:, -1] > 5.0)),
                    "halfwidth_km_median": float(np.median(W)),
                    "widened_runs": int(np.sum([r[name]["widened"].any() for r in runs])),
                }
                key = f"{s}|{dv}|{name}"
                curves[key + "|err"] = E
                curves[key + "|inside"] = I
                curves[key + "|halfwidth"] = W
    curves["t_h"] = np.arange(E.shape[1]) * qb.CADENCE_S / 3600.0
    (OUT / "sweep.json").write_text(json.dumps({"n_runs": N_RUNS, "results": summary}, indent=1))
    np.savez_compressed(OUT / "curves.npz", **curves)

    for s in summary:
        print(f"\n{s}: inside % / median RMS km / runs ending > 5 km off / median half-width km")
        for dv, row in summary[s].items():
            print(f"  dv {dv:>5} m/s  " + "  |  ".join(
                f"{n}: {v['inside_pct']:5.1f} / {v['rms_km_median']:6.2f} / {v['runs_final_over_5km']:2d} / {v['halfwidth_km_median']:5.1f}"
                for n, v in row.items()))


if __name__ == "__main__":
    main()
