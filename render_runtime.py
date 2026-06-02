import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

data = [
    ("Branin",       2,  3.4,  4.4,  6.4),
    ("Hartmann",     4,  5.1,  4.6, 10.2),
    ("Ackley",       5,  2.4,  4.0,  5.0),
    ("Hartmann",     6,  4.3,  9.1,  5.8),
    ("Car",          7,  8.6, 13.1, 10.2),
    ("Fashion",      7,  7.5,  4.9, 11.3),
    ("Higgs",        7,  2.9,  3.2,  4.8),
    ("MiniBooNE",    7,  7.9,  7.6,  4.1),
    ("Segment",      7,  1.6,  2.0,  4.1),
    ("Ackley",       8,  1.9,  3.4,  3.8),
    ("HPO-B-5527",   8,  5.5,  9.1,  5.5),
    ("HPO-B-5891",   8, 31.3, 58.7, 57.2),
    ("HPO-B-7609",   9, 65.0, 34.4, 72.4),
    ("HPO-B-5965",  10, 12.1, 16.7, 20.6),
    ("HPO-B-5971",  16, 24.2, 37.6, 31.5),
]
data.sort(key=lambda r: min(r[2], r[3], r[4]))
labels = [f"{n} ({d}D)" for (n, d, *_ ) in data]
mes = np.array([r[2] for r in data])
jes = np.array([r[3] for r in data])
pes = np.array([r[4] for r in data])

C_MES, C_JES, C_PES = "#4C72B0", "#DD8452", "#55A868"

fig, ax = plt.subplots(figsize=(8.0, 7.5))
y = np.arange(len(labels)); h = 0.27
ax.barh(y - h, mes, height=h, color=C_MES, label="MES")
ax.barh(y,     jes, height=h, color=C_JES, label="JES")
ax.barh(y + h, pes, height=h, color=C_PES, label="PES")
ax.set_xscale("log")
ax.set_xlim(1, 110)
ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
ax.set_xticklabels(["1×", "2×", "5×", "10×", "20×", "50×", "100×"])
ax.axvline(1, ls="--", lw=0.9, color="#888", zorder=0)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9); ax.invert_yaxis()
ax.set_xlabel("Speedup over GP fully-Bayesian MCMC baseline (log scale)")
ax.tick_params(axis="x", labelsize=9)
ax.grid(axis="x", which="both", ls=":", color="#cfcfcf", lw=0.6, zorder=0); ax.set_axisbelow(True)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
          frameon=False, fontsize=10, title="ES variant", title_fontsize=10)
ax.set_title(r"$\alpha$-PFN: runtime speedup across 15 BO tasks",
             fontsize=12, pad=10, loc="left")

def annotate(val, row, offset_row, txt, col):
    ax.annotate(txt, xy=(val, row + offset_row), xytext=(val * 1.35, row + offset_row),
                fontsize=9, color=col, va="center", ha="left",
                arrowprops=dict(arrowstyle="-", color=col, lw=0.7))
annotate(1.6, 0, -h, "worst case 1.6×", "#333")
for idx, (n, d, *_) in enumerate(data):
    if n == "HPO-B-7609":
        annotate(72.4, idx, +h, "72×", "#333"); break

plt.tight_layout()
out = "/work/dlclarge2/rakotoah-entropy_search/misc/heri/alphapfn-page/static/images/runtime_speedup.png"
plt.savefig(out, dpi=160, bbox_inches="tight")
import os; print(f"saved {out} ({os.path.getsize(out)//1024} KB)")
