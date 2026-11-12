"""Figures from reports/falsification.json (written by carla_tools/falsify_carla.py).

  python scripts/plot_falsification.py      -> reports/figures/falsification_boundary.png, falsification_search.png

1. Failure boundary: GP-predicted P(collision) on three 2-D slices, with the trials that ran in CARLA marked
   (red x = collision, green o = avoided). Points are projections of 6-D trials, so they are shown for orientation,
   not as exact members of the slice.
2. Search efficiency: cumulative failures found vs trials, falsifier vs random search on the same budget.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TITLES = {"speed_vs_sun_occluded": "Pedestrian hidden by a parked van",
          "speed_vs_sun_clear_view": "Clear line of sight",
          "speed_vs_ped_speed": "Daylight, clear view"}
LABELS = {"ego_kmh": "Ego speed (km/h)", "sun_altitude_deg": "Sun altitude (°, < 0 = night)",
          "ped_kmh": "Pedestrian walking speed (km/h)"}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "reports" / "falsification.json")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "figures")
    args = ap.parse_args()
    d = json.loads(args.input.read_text())
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    slices = d.get("boundary", {})
    fig, axes = plt.subplots(1, len(slices), figsize=(5.2 * len(slices), 4.4), squeeze=False)
    trials = d["falsifier"]["trials"]
    for ax, (key, g) in zip(axes[0], slices.items()):
        x, y, p = np.array(g["x_values"]), np.array(g["y_values"]), np.array(g["p_fail"]).T
        im = ax.contourf(x, y, p, levels=np.linspace(0, 1, 11), cmap="RdYlGn_r", vmin=0, vmax=1)
        ax.contour(x, y, p, levels=[0.5], colors="black", linewidths=1.5)
        for t in trials:
            px, py = t["params"][g["x"]], t["params"][g["y"]]
            if t["margin"] < 0:
                ax.scatter(px, py, marker="x", s=28, color="#b00020", linewidths=1.4, alpha=0.8)
            else:
                ax.scatter(px, py, marker="o", s=28, facecolors="none", edgecolors="#1b7f3a", linewidths=1.4,
                           alpha=0.8)
        ax.set_title(TITLES.get(key, key), fontsize=11)
        ax.set_xlabel(LABELS.get(g["x"], g["x"]))
        ax.set_ylabel(LABELS.get(g["y"], g["y"]))
    fig.colorbar(im, ax=axes[0].tolist(), label="Predicted P(collision)", shrink=0.85)
    fig.suptitle("Where the AEB fails: GP failure map from the falsification search (black line: P = 0.5)", fontsize=12)
    fig.savefig(out / "falsification_boundary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, color in (("falsifier", "#3b5bdb"), ("random", "#868e96")):
        found = d[name]["summary"]["failures_found_by_trial"]
        if found:
            ax.plot(np.arange(1, len(found) + 1), found, color=color, lw=2,
                    label=f"{'Falsification search' if name == 'falsifier' else 'Random search'} ({found[-1]} failures)")
    ax.set_xlabel("CARLA trials")
    ax.set_ylabel("Failures found")
    ax.set_title("Same budget, more failures found")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(out / "falsification_search.png", dpi=150, bbox_inches="tight")
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
