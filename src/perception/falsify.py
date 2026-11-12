"""Safety falsification: search the scenario space for the conditions where AEB fails, and map the failure boundary.

Euro NCAP-style tests check a handful of fixed points (CPNA-25 at 20/30/40 km/h, ...). A system can pass all of them
and still fail just next to them, e.g. at night with a faster pedestrian. Falsification treats the closed-loop
simulation as a black box f(scenario parameters) -> safety margin and *searches* for negative margins, spending the
simulation budget where the outcome is uncertain instead of on a fixed grid.

  margin  =  minimum gap to the hazard in metres, if the car avoided it       (> 0: safe)
          =  -impact speed in km/h, if it hit                                   (< 0: failure; worse = more negative)

Search: a Latin-hypercube start, then a Gaussian-process surrogate of the margin over the unit cube, and each new
trial maximises the *straddle* acquisition 1.96 sigma - |mu| (Bryan et al., 2005), which is high where the model is
unsure which side of zero the margin falls on, i.e. along the failure boundary. Every `falsify_every`-th trial instead
minimises the lower confidence bound mu - 2 sigma, to dig for the worst case. A random-search baseline with the
same budget is run by the caller, so "the search finds failures faster" is measured, not assumed.

Pure numpy and no CARLA, so the search itself is unit-tested in CI on synthetic simulators.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Param:
    name: str
    low: float
    high: float
    integer: bool = False     # e.g. occluded 0/1; rounded after scaling

    def scale(self, u: float) -> float:
        v = self.low + float(u) * (self.high - self.low)
        return float(round(v)) if self.integer else v


# The default AEB search space (CARLA). Units are what carla_tools/falsify_carla.py expects.
SPACE = [
    Param("ego_kmh", 15.0, 50.0),
    Param("ped_kmh", 3.0, 10.0),
    Param("overlap", 0.1, 0.9),
    Param("sun_altitude_deg", -10.0, 60.0),   # < 0: night (headlights only)
    Param("rain", 0.0, 80.0),                 # CARLA precipitation and wetness, %
    Param("occluded", 0, 1, integer=True),    # a parked car hides the pedestrian until they step out
]


def to_params(u: np.ndarray, space: list[Param] = SPACE) -> dict:
    return {p.name: p.scale(x) for p, x in zip(space, u)}


def latin_hypercube(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    cut = (np.arange(n)[:, None] + rng.random((n, d))) / n
    for j in range(d):
        cut[:, j] = cut[rng.permutation(n), j]
    return cut


def margin(collided: bool, min_gap_m: float, impact_kmh: float) -> float:
    return -float(impact_kmh) if collided else float(min_gap_m)


class GP:
    """Zero-mean GP regression on [0, 1]^d with an ARD squared-exponential kernel. Hyper-parameters (one length scale
    for all dims, signal and noise variance) are picked by maximising the log marginal likelihood over a small grid,
    which is robust for the tens-to-hundreds of points a falsification run produces."""

    LENGTHS = (0.1, 0.15, 0.25, 0.4, 0.6)
    NOISES = (1e-4, 1e-2, 0.05, 0.2)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GP":
        self.X = np.asarray(X, float)
        self.mu_y, self.sd_y = float(np.mean(y)), float(np.std(y)) or 1.0
        z = (np.asarray(y, float) - self.mu_y) / self.sd_y
        best = None
        for ls in self.LENGTHS:
            K0 = self._k(self.X, self.X, ls)
            for nz in self.NOISES:
                try:
                    L = np.linalg.cholesky(K0 + (nz + 1e-8) * np.eye(len(z)))
                except np.linalg.LinAlgError:
                    continue
                a = np.linalg.solve(L.T, np.linalg.solve(L, z))
                lml = -0.5 * z @ a - np.log(np.diag(L)).sum()
                if best is None or lml > best[0]:
                    best = (lml, ls, L, a)
        _, self.ls, self.L, self.alpha = best
        return self

    @staticmethod
    def _k(A: np.ndarray, B: np.ndarray, ls: float) -> np.ndarray:
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-0.5 * d2 / ls ** 2)

    def predict(self, Xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Ks = self._k(np.asarray(Xs, float), self.X, self.ls)
        mu = Ks @ self.alpha
        v = np.linalg.solve(self.L, Ks.T)
        var = np.clip(1.0 - (v ** 2).sum(0), 1e-12, None)
        return mu * self.sd_y + self.mu_y, np.sqrt(var) * self.sd_y


@dataclass
class Falsifier:
    simulate: Callable[[dict], float]          # scenario parameters -> safety margin
    space: list[Param] = field(default_factory=lambda: list(SPACE))
    n_init: int = 12
    candidates: int = 4000
    falsify_every: int = 4
    seed: int = 0
    history: list[dict] = field(default_factory=list)

    def _u(self) -> np.ndarray:
        return np.array([h["u"] for h in self.history])

    def _y(self) -> np.ndarray:
        return np.array([h["margin"] for h in self.history])

    def _run(self, u: np.ndarray, how: str) -> dict:
        params = to_params(u, self.space)
        rec = {"trial": len(self.history), "how": how, "u": [round(float(x), 5) for x in u], "params": params,
               "margin": float(self.simulate(params))}
        self.history.append(rec)
        return rec

    def propose(self, rng: np.random.Generator) -> tuple[np.ndarray, str]:
        gp = GP().fit(self._u(), self._y())
        C = rng.random((self.candidates, len(self.space)))
        for j, p in enumerate(self.space):  # integer dims: only the levels that exist
            if p.integer:
                C[:, j] = np.round(C[:, j] * (p.high - p.low)) / max(p.high - p.low, 1)
        mu, sd = gp.predict(C)
        if (len(self.history) - self.n_init) % self.falsify_every == self.falsify_every - 1:
            return C[np.argmin(mu - 2 * sd)], "lcb"
        return C[np.argmax(1.96 * sd - np.abs(mu))], "straddle"

    def run(self, budget: int) -> list[dict]:
        rng = np.random.default_rng(self.seed)
        for u in latin_hypercube(min(self.n_init, budget), len(self.space), rng):
            for j, p in enumerate(self.space):
                if p.integer:
                    u[j] = np.round(u[j] * (p.high - p.low)) / max(p.high - p.low, 1)
            self._run(u, "init")
        while len(self.history) < budget:
            u, how = self.propose(rng)
            self._run(u, how)
        return self.history


def random_search(simulate: Callable[[dict], float], budget: int, space: list[Param] = SPACE,
                  seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed + 10_000)
    out = []
    for t in range(budget):
        u = rng.random(len(space))
        params = to_params(u, space)
        out.append({"trial": t, "how": "random", "u": [round(float(x), 5) for x in u], "params": params,
                    "margin": float(simulate(params))})
    return out


def summary(history: list[dict]) -> dict:
    m = np.array([h["margin"] for h in history])
    fails = np.where(m < 0)[0]
    worst = int(np.argmin(m))
    return {"trials": int(len(m)), "failures": int(len(fails)), "failure_rate": round(float(len(fails) / len(m)), 3),
            "first_failure_trial": int(fails[0]) if len(fails) else None,
            "worst": {"margin": round(float(m[worst]), 2), "params": history[worst]["params"]},
            "failures_found_by_trial": np.cumsum(m < 0).astype(int).tolist()}


def boundary_grid(history: list[dict], space: list[Param], x: str, y: str, fixed: dict | None = None,
                  n: int = 60) -> dict:
    """GP-predicted P(failure) on a 2-D slice (x, y) with the other parameters at `fixed` (default: mid-range,
    occluded=0). P(failure) = P(margin < 0) = Phi(-mu / sigma)."""
    from math import erf, sqrt

    gp = GP().fit(np.array([h["u"] for h in history]), np.array([h["margin"] for h in history]))
    names = [p.name for p in space]
    base = np.full(len(space), 0.5)
    for k, v in (fixed or {"occluded": 0}).items():
        p = space[names.index(k)]
        base[names.index(k)] = (v - p.low) / (p.high - p.low)
    gx, gy = np.linspace(0, 1, n), np.linspace(0, 1, n)
    U = np.tile(base, (n * n, 1))
    U[:, names.index(x)] = np.repeat(gx, n)
    U[:, names.index(y)] = np.tile(gy, n)
    mu, sd = gp.predict(U)
    pfail = np.array([0.5 * (1 + erf(-a / (b * sqrt(2)))) for a, b in zip(mu, sd)]).reshape(n, n)
    px, py = space[names.index(x)], space[names.index(y)]
    return {"x": x, "y": y, "x_values": [px.scale(v) for v in gx], "y_values": [py.scale(v) for v in gy],
            "p_fail": np.round(pfail, 4).tolist(), "fixed": fixed or {"occluded": 0}}


def save(path, falsifier: list[dict], random: list[dict], extra: dict | None = None) -> dict:
    out = {"space": [vars(p) for p in SPACE], "falsifier": {"summary": summary(falsifier), "trials": falsifier},
           "random": {"summary": summary(random), "trials": random}, **(extra or {})}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return out
