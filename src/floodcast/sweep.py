from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import pandas as pd

from .model import Params, run_event

PRESETS = {
    "demo": dict(forecast_error=[0.2, 0.4, 0.6], obs_hours=[3, 6, 12], n_members=[100], events=300),
    "full": dict(forecast_error=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8], obs_hours=[3, 6, 9, 12, 18, 24],
                 n_members=[200], events=2000),
}


def build_scenarios(preset: str) -> list[Params]:
    cfg = dict(PRESETS[preset])
    n = cfg.pop("events")
    keys = list(cfg)
    return [Params(**dict(zip(keys, combo)), seed=s)
            for combo in itertools.product(*(cfg[k] for k in keys)) for s in range(n)]


def _run(p):
    return run_event(p)


def run_serial(sc):
    return [_run(p) for p in sc]


def run_mp(sc, workers):
    from multiprocessing import Pool
    with Pool(workers) as pool:
        return pool.map(_run, sc, chunksize=max(1, len(sc) // (workers * 8)))


def run_mpi(sc):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    mine = [_run(p) for p in sc[rank::size]]
    gathered = comm.gather(mine, root=0)
    return None if rank != 0 else [r for part in gathered for r in part]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=PRESETS, default="demo")
    ap.add_argument("--backend", choices=["serial", "mp", "mpi"], default="serial")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="results/sweep.csv")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    sc = build_scenarios(a.preset)
    if a.limit:
        sc = sc[:a.limit]
    t0 = time.perf_counter()
    rows = run_serial(sc) if a.backend == "serial" else run_mp(sc, a.workers) if a.backend == "mp" else run_mpi(sc)
    dt = time.perf_counter() - t0
    if rows is not None:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)
        print(f"{len(rows)} kejadian | backend={a.backend} | {dt:.1f} s | -> {a.out}")


if __name__ == "__main__":
    main()
