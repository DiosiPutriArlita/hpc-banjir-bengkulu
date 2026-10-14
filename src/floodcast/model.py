from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import lru_cache

import numpy as np

T = 120
FLOOD_STAGE_M = 4.5
RATING_A, RATING_B = 4.5 / np.sqrt(350.0), 0.5
BASEFLOW = 20.0

AREA = np.array([120.0, 90.0, 150.0, 80.0, 60.0])
CN0 = np.array([68.0, 72.0, 65.0, 78.0, 85.0])
K_H = np.array([5.0, 4.0, 4.0, 3.0, 2.0])
DELAY = np.array([12, 9, 6, 3, 0])
AMC_PROB = np.array([0.25, 0.45, 0.30])


@dataclass(frozen=True)
class Params:
    seed: int = 0
    forecast_error: float = 0.4
    ens_spread: float = -1.0
    obs_hours: int = 6
    n_members: int = 100

    def as_dict(self):
        return asdict(self)


@lru_cache(maxsize=1)
def kernels():
    t = np.arange(T)
    s, tt = np.meshgrid(t, t, indexing="ij")
    lag = tt - s
    mats = []
    for k, d in zip(K_H, DELAY):
        x = t + 0.5
        base = x / k**2 * np.exp(-x / k)
        base /= base.sum()
        kern = np.zeros(T)
        kern[d:] = base[:T - d]
        mats.append(np.where(lag >= 0, kern[np.clip(lag, 0, T - 1)], 0.0))
    return np.array(mats)


def amc_cn(amc: np.ndarray) -> np.ndarray:
    cn = CN0[None, :]
    dry = 4.2 * cn / (10 - 0.058 * cn)
    wet = 23 * cn / (10 + 0.13 * cn)
    return np.where(amc[:, None] == 0, dry, np.where(amc[:, None] == 2, wet, cn))


def hydrograph(rain: np.ndarray, cn: np.ndarray) -> np.ndarray:
    S = 25400.0 / cn - 254.0
    Ia = 0.2 * S
    cum = np.cumsum(rain, axis=2)
    ex = np.maximum(cum - Ia[:, :, None], 0.0)
    qcum = ex**2 / (ex + S[:, :, None])
    inc = np.diff(qcum, axis=2, prepend=0.0)
    M = kernels()
    q = np.full((rain.shape[0], T), BASEFLOW)
    for i in range(5):
        q += (inc[:, i, :] * AREA[i] / 3.6) @ M[i]
    return q


def stage(q: np.ndarray) -> np.ndarray:
    return RATING_A * np.power(q, RATING_B)


def run_event(p: Params, return_state: bool = False) -> dict:
    rt = np.random.default_rng(np.random.SeedSequence([p.seed, 0]))
    P = rt.lognormal(np.log(70.0), 0.55)
    D = rt.uniform(6, 36)
    u = rt.uniform(0.2, 0.8)
    a, b = 1 + 4 * u, 1 + 4 * (1 - u)
    x = (np.arange(T) + 0.5) / D
    w = np.where(x < 1, x ** (a - 1) * np.clip(1 - x, 0, None) ** (b - 1), 0.0)
    w /= w.sum()
    spatial = np.exp(rt.normal(0, 0.25, 5))
    rain_t = P * spatial[:, None] * w[None, :]
    amc_t = rt.choice(3, p=AMC_PROB)


    rf = np.random.default_rng(np.random.SeedSequence([p.seed, 1, int(p.forecast_error * 1000), p.obs_hours]))
    sm = p.forecast_error if p.ens_spread < 0 else p.ens_spread
    M = p.n_members
    center = np.exp(rf.normal(0, p.forecast_error))
    fac = center * np.exp(rf.normal(0, sm, (M, 1, 1)))
    future = (np.arange(T) >= p.obs_hours)[None, None, :]
    rain_m = np.where(future, rain_t[None] * fac, rain_t[None])
    amc_m = rf.choice(3, size=M, p=AMC_PROB)

    rain = np.concatenate([rain_t[None], rain_m], axis=0)
    cn = amc_cn(np.concatenate([[amc_t], amc_m]))
    h = stage(hydrograph(rain, cn))

    ht, hm = h[0], h[1:]
    over = np.where(ht >= FLOOD_STAGE_M)[0]
    peak_m = hm.max(axis=1)
    res = {
        **p.as_dict(),
        "rain_total_mm": float(P), "rain_duration_h": float(D), "amc": int(amc_t),
        "truth_peak_stage": float(ht.max()), "truth_flood": bool(over.size > 0),
        "truth_exceed_hour": float(over[0]) if over.size else float("nan"),
        "p_exceed": float((peak_m >= FLOOD_STAGE_M).mean()),
        "ens_p10": float(np.percentile(peak_m, 10)), "ens_p50": float(np.percentile(peak_m, 50)),
        "ens_p90": float(np.percentile(peak_m, 90)),
    }
    if return_state:
        res["_state"] = dict(rain=rain_t.mean(axis=0), truth=ht,
                             p10=np.percentile(hm, 10, axis=0), p50=np.percentile(hm, 50, axis=0),
                             p90=np.percentile(hm, 90, axis=0))
    return res
