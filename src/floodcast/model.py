"""
Prakiraan banjir probabilistik untuk satu DAS SINTETIS (5 sub-DAS, ~500 km2)
dengan titik kota di hilir.

Rantai model (hidrologi sederhana, hanya numpy):
  hujan per jam  ->  SCS Curve Number (limpasan)  ->  hidrograf satuan gamma (Nash)
  ->  debit di kota  ->  kurva debit-muka air (rating curve)  ->  muka air (m)

Eksperimen "prakiraan":
  - Ada satu kejadian hujan "kenyataan" (truth) yang tidak diketahui sebelumnya.
  - Pada jam ke-`obs_hours` sejak hujan mulai, pengamat hanya mengetahui hujan yang SUDAH turun.
  - Hujan sesudahnya diprakirakan dengan ketidakpastian (`forecast_error`) lewat
    ensemble `n_members` anggota; kelembapan tanah awal juga tidak diketahui.
  - Hasil: peluang muka air melewati batas siaga banjir (`p_exceed`) vs kenyataan.

Semua angka adalah ASUMSI untuk pembelajaran, bukan kalibrasi sungai nyata.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import lru_cache

import numpy as np

T = 120                      # jam yang disimulasikan
FLOOD_STAGE_M = 4.5          # batas siaga banjir di kota (m)
RATING_A, RATING_B = 4.5 / np.sqrt(350.0), 0.5   # h = a * Q^b  (Q=350 m3/s -> 4,5 m)
BASEFLOW = 20.0              # m3/s

AREA = np.array([120.0, 90.0, 150.0, 80.0, 60.0])        # km2, hulu -> hilir
CN0 = np.array([68.0, 72.0, 65.0, 78.0, 85.0])           # curve number kondisi normal
K_H = np.array([5.0, 4.0, 4.0, 3.0, 2.0])                # skala waktu hidrograf satuan (jam)
DELAY = np.array([12, 9, 6, 3, 0])                       # waktu tempuh sungai ke kota (jam)
AMC_PROB = np.array([0.25, 0.45, 0.30])                  # kering, normal, basah


@dataclass(frozen=True)
class Params:
    seed: int = 0
    forecast_error: float = 0.4    # sigma log-normal galat prakiraan hujan sisa
    ens_spread: float = -1.0       # sigma sebaran anggota (<0 = sama dgn forecast_error -> terkalibrasi)
    obs_hours: int = 6             # jam sejak hujan mulai saat peringatan diputuskan
    n_members: int = 100

    def as_dict(self):
        return asdict(self)


@lru_cache(maxsize=1)
def kernels():
    """Matriks konvolusi (T x T) per sub-DAS: hidrograf satuan gamma + waktu tempuh."""
    t = np.arange(T)
    s, tt = np.meshgrid(t, t, indexing="ij")
    lag = tt - s
    mats = []
    for k, d in zip(K_H, DELAY):
        x = t + 0.5
        base = x / k**2 * np.exp(-x / k)                 # gamma(n=2)
        base /= base.sum()
        kern = np.zeros(T); kern[d:] = base[:T - d]
        mats.append(np.where(lag >= 0, kern[np.clip(lag, 0, T - 1)], 0.0))
    return np.array(mats)                                 # (5, T, T)


def amc_cn(amc: np.ndarray) -> np.ndarray:
    """CN per kelas kelembapan awal. amc: (B,) in {0,1,2} -> (B,5)."""
    cn = CN0[None, :]
    dry = 4.2 * cn / (10 - 0.058 * cn)
    wet = 23 * cn / (10 + 0.13 * cn)
    return np.where(amc[:, None] == 0, dry, np.where(amc[:, None] == 2, wet, cn))


def hydrograph(rain: np.ndarray, cn: np.ndarray) -> np.ndarray:
    """rain (B,5,T) mm/jam, cn (B,5) -> debit di kota (B,T) m3/s."""
    S = 25400.0 / cn - 254.0
    Ia = 0.2 * S
    cum = np.cumsum(rain, axis=2)
    ex = np.maximum(cum - Ia[:, :, None], 0.0)
    qcum = ex**2 / (ex + S[:, :, None])
    inc = np.diff(qcum, axis=2, prepend=0.0)              # limpasan mm/jam
    M = kernels()
    q = np.full((rain.shape[0], T), BASEFLOW)
    for i in range(5):
        q += (inc[:, i, :] * AREA[i] / 3.6) @ M[i]         # km2*mm/jam -> m3/s = /3.6
    return q


def stage(q: np.ndarray) -> np.ndarray:
    return RATING_A * np.power(q, RATING_B)


def run_event(p: Params, return_state: bool = False) -> dict:
    # --- "kenyataan" (truth): sama untuk semua konfigurasi dengan seed yang sama
    rt = np.random.default_rng(np.random.SeedSequence([p.seed, 0]))
    P = rt.lognormal(np.log(70.0), 0.55)                   # total hujan (mm)
    D = rt.uniform(6, 36)                                  # durasi (jam)
    u = rt.uniform(0.2, 0.8)
    a, b = 1 + 4 * u, 1 + 4 * (1 - u)
    x = (np.arange(T) + 0.5) / D
    w = np.where(x < 1, x ** (a - 1) * np.clip(1 - x, 0, None) ** (b - 1), 0.0)
    w /= w.sum()
    spatial = np.exp(rt.normal(0, 0.25, 5))
    rain_t = P * spatial[:, None] * w[None, :]             # (5,T)
    amc_t = rt.choice(3, p=AMC_PROB)

    # --- ensemble prakiraan
    rf = np.random.default_rng(np.random.SeedSequence([p.seed, 1, int(p.forecast_error * 1000), p.obs_hours]))
    sm = p.forecast_error if p.ens_spread < 0 else p.ens_spread
    M = p.n_members
    center = np.exp(rf.normal(0, p.forecast_error))        # bias pusat prakiraan (sama utk semua anggota)
    fac = center * np.exp(rf.normal(0, sm, (M, 1, 1)))
    future = (np.arange(T) >= p.obs_hours)[None, None, :]
    rain_m = np.where(future, rain_t[None] * fac, rain_t[None])   # (M,5,T)
    amc_m = rf.choice(3, size=M, p=AMC_PROB)

    rain = np.concatenate([rain_t[None], rain_m], axis=0)
    cn = amc_cn(np.concatenate([[amc_t], amc_m]))
    h = stage(hydrograph(rain, cn))                         # (1+M, T)

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
