"""Ukur speed-up multiprocessing: waktu vs jumlah worker. Jalankan dari root repo."""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from floodcast.sweep import build_scenarios, run_mp, run_serial  # noqa: E402
import pandas as pd

sc = build_scenarios("demo")[:400]
maxw = os.cpu_count() or 1
t0 = time.perf_counter(); run_serial(sc); t1 = time.perf_counter() - t0
rows = [dict(workers=1, seconds=t1, speedup=1.0, efficiency=1.0)]
w = 2
while w <= maxw:
    t0 = time.perf_counter(); run_mp(sc, w); tw = time.perf_counter() - t0
    rows.append(dict(workers=w, seconds=tw, speedup=t1 / tw, efficiency=t1 / tw / w)); w *= 2
df = pd.DataFrame(rows); print(df.round(2).to_string(index=False))
Path("results").mkdir(exist_ok=True); df.to_csv("results/scaling.csv", index=False)
