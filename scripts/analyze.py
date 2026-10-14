import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from floodcast import Params, run_event, FLOOD_STAGE_M

out = Path("results"); out.mkdir(exist_ok=True)
csv = sys.argv[1] if len(sys.argv) > 1 else "results/sweep.csv"
df = pd.read_csv(csv)
df["lead_h"] = df.truth_exceed_hour - df.obs_hours
THR = np.linspace(0.05, 0.95, 19)


def verify(g, thr):
    alert = g.p_exceed >= thr
    fl = g.truth_flood
    hit, miss, fa = (alert & fl).sum(), (~alert & fl).sum(), (alert & ~fl).sum()
    return dict(POD=hit / max(hit + miss, 1), FAR=fa / max(hit + fa, 1), CSI=hit / max(hit + miss + fa, 1))


rows = []
for (fe, oh), g in df.groupby(["forecast_error", "obs_hours"]):
    v = verify(g, 0.3)
    hits = g[(g.p_exceed >= 0.3) & g.truth_flood & (g.lead_h > 0)]
    rows.append(dict(forecast_error=fe, obs_hours=oh, brier=((g.p_exceed - g.truth_flood) ** 2).mean(),
                     **v, lead_hit_h=hits.lead_h.mean(), n=len(g)))
summ = pd.DataFrame(rows)
summ.to_csv(out / "summary_by_config.csv", index=False)
print(f"Kejadian: {len(df)} | frekuensi banjir: {df.truth_flood.mean():.1%}")
print("\nRINGKASAN (ambang peringatan p>=30%):")
print(summ.round(2).to_string(index=False))


ref = df[(df.forecast_error == 0.4) & (df.obs_hours == 6) & df.truth_flood & df.p_exceed.between(0.4, 0.9)]
seed = int(ref.sort_values("truth_peak_stage").iloc[len(ref) // 2].seed)
r = run_event(Params(seed=seed, forecast_error=0.4, obs_hours=6), return_state=True)
st = r.pop("_state"); t = np.arange(len(st["truth"]))
fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, gridspec_kw=dict(height_ratios=[1, 2.2]))
a1.bar(t, st["rain"], color="tab:blue"); a1.set_ylabel("Hujan (mm/jam)"); a1.invert_yaxis()
a1.set_title(f"Contoh kejadian: hujan {r['rain_total_mm']:.0f} mm, peluang banjir diprakirakan {r['p_exceed']:.0%} (jam ke-6)")
a2.fill_between(t, st["p10"], st["p90"], alpha=.3, color="tab:orange", label="ensemble P10-P90")
a2.plot(t, st["p50"], c="tab:orange", label="ensemble median")
a2.plot(t, st["truth"], c="k", lw=2, label="kenyataan (truth)")
a2.axhline(FLOOD_STAGE_M, c="red", ls="--", label="siaga banjir")
a2.axvline(6, c="gray", ls=":"); a2.text(6.5, a2.get_ylim()[1] * .92, "peringatan diputuskan", fontsize=8)
a2.set_xlim(0, 80); a2.set_xlabel("Jam sejak hujan mulai"); a2.set_ylabel("Muka air di kota (m)")
a2.legend(fontsize=8); a2.grid(alpha=.3); fig.tight_layout(); fig.savefig(out / "example_event.png", dpi=150)


fig, ax = plt.subplots(figsize=(5.5, 5))
bins = np.linspace(0, 1, 11)
for fe in sorted(df.forecast_error.unique()):
    g = df[(df.forecast_error == fe) & (df.obs_hours == 6)]
    b = pd.cut(g.p_exceed, bins, include_lowest=True)
    d = g.groupby(b, observed=True).agg(fc=("p_exceed", "mean"), ob=("truth_flood", "mean"))
    ax.plot(d.fc, d.ob, marker="o", label=f"galat hujan sigma={fe}")
ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlabel("Peluang banjir diprakirakan"); ax.set_ylabel("Frekuensi banjir teramati")
ax.set_title("Reliabilitas prakiraan (jam ke-6)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(out / "reliability.png", dpi=150)


fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
for fe, g in summ.groupby("forecast_error"):
    a1.plot(g.obs_hours, g.CSI, marker="o", label=f"sigma={fe}")
    a2.plot(g.obs_hours, g.lead_hit_h, marker="o", label=f"sigma={fe}")
a1.set_xlabel("Jam pengamatan hujan sebelum memutuskan"); a1.set_ylabel("Skor CSI (makin tinggi makin baik)")
a2.set_xlabel("Jam pengamatan hujan sebelum memutuskan"); a2.set_ylabel("Waktu tunggu rata-rata sebelum banjir (jam)")
a1.set_title("Menunggu data lebih lama -> prakiraan lebih tepat"); a2.set_title("...tetapi waktu evakuasi makin pendek")
for a in (a1, a2): a.grid(alpha=.3); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(out / "skill_vs_leadtime.png", dpi=150)


g = df[(df.forecast_error == 0.4) & (df.obs_hours == 6)]
fig, ax = plt.subplots(figsize=(7.5, 4.8))
opt = {}
for alpha in (0.05, 0.1, 0.2, 0.4):
    cost = []
    for th in THR:
        al = g.p_exceed >= th
        cost.append((alpha * al + (~al & g.truth_flood) * 1.0).mean())
    cost = np.array(cost); i = cost.argmin(); opt[alpha] = (THR[i], cost[i])
    ax.plot(THR, cost, label=f"biaya/kerugian = {alpha}")
    ax.scatter([THR[i]], [cost[i]], s=60, zorder=5)
ax.set_xlabel("Ambang peluang untuk mengeluarkan peringatan"); ax.set_ylabel("Biaya rata-rata per kejadian (satuan kerugian)")
ax.set_title("Ambang peringatan optimal bergantung pada biaya evakuasi"); ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(out / "alert_threshold_cost.png", dpi=150)
print("\nAMBANG OPTIMAL (sigma=0.4, jam ke-6):")
for a_, (th, c) in opt.items():
    print(f"  biaya/kerugian={a_:<5} -> ambang optimal p>={th:.2f} (biaya rata-rata {c:.3f})")
print("\nOK -> results/")
