"""Write-up figure: corpus size by year, external-test ranks, time-split metrics."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, "src")
from lit_layers import lit_table  # noqa: E402

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a64", "#e4e3de"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left"})

S = pd.read_csv("results/valency_oob_scores.csv")
lit = lit_table()


def pctiles(col, test_keys):
    u = S[S.y == 0]
    s = u.groupby("comp_key")[col].max()
    s = s[~s.index.isin(S.comp_key[S.y == 1])]
    t = s.index.isin(test_keys)
    return np.array([(s[~t] < v).mean() for v in s[t]])


fig, ax = plt.subplots(1, 3, figsize=(10, 2.9), gridspec_kw={"width_ratios": [1.25, 1, 1]})

# (a) corpus
years = pd.read_json("data/valency/corpus.jsonl.gz", lines=True, convert_dates=False).date.astype(str).str[:4]
yc = years[years >= "2000"].value_counts().sort_index()
ax[0].bar(yc.index.astype(int), yc.values, color=BLUE, width=0.8)
ax[0].set_title("a  Valency corpus: 28,839 perovskite papers")
ax[0].set_ylabel("papers per year")
ax[0].yaxis.grid(True, color=GRID, lw=0.6)
ax[0].set_axisbelow(True)
ax[0].text(2000.5, yc.max() * 0.92, "11,929 formula mentions\n1,914 compositions", color=INK, va="top")

# (b) external test ranks
test_all = lit[lit.n_synth >= 1].index
for i, (col, c, lab) in enumerate([("all | base (comp+struct+DFT)", BLUE, "DFT+composition"),
                                    ("all | base+nbr", ORANGE, "+ Valency neighbours")]):
    p = pctiles(col, test_all) * 100
    jitter = np.random.default_rng(i).uniform(-0.12, 0.12, len(p))
    ax[1].scatter(np.full(len(p), i) + jitter, p, s=18, color=c, edgecolor="white", lw=0.6, zorder=3)
    ax[1].hlines(np.median(p), i - 0.25, i + 0.25, color=INK, lw=1.5, zorder=4)
    ax[1].text(i + 0.28, np.median(p), f"median {np.median(p):.0f}", va="center", color=INK, fontsize=8)
ax[1].set_xticks([0, 1], ["DFT+comp.", "+ Valency\nneighbours"])
ax[1].set_xlim(-0.5, 1.9)
ax[1].set_ylabel("percentile among unlabeled")
ax[1].set_title("b  Literature-confirmed hidden positives")
ax[1].yaxis.grid(True, color=GRID, lw=0.6)
ax[1].set_axisbelow(True)

# (c) time split
R = json.load(open("results/valency_ablation.json"))
metrics = [("ext_auc", "AUC"), ("ext_recall@0.5", "recall@0.5")]
x = np.arange(len(metrics))
for j, (key, c, lab) in enumerate([("2020-01-01 | base (comp+struct+DFT)", BLUE, "DFT+composition"),
                                    ("2020-01-01 | base+nbr", ORANGE, "+ neighbours (pre-2020 only)")]):
    vals = [R[key][m] for m, _ in metrics]
    bars = ax[2].bar(x + (j - 0.5) * 0.36, vals, 0.34, color=c, label=lab)
    for b, v in zip(bars, vals):
        ax[2].text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", color=INK, fontsize=8)
ax[2].set_xticks(x, [m for _, m in metrics])
ax[2].set_ylim(0, 1.38)
ax[2].set_title("c  Prospective test: 2020+ (n=13)")
ax[2].legend(frameon=False, fontsize=7.5, loc="upper left", ncol=1)
ax[2].set_yticks([0, .2, .4, .6, .8, 1.0])
ax[2].yaxis.grid(True, color=GRID, lw=0.6)
ax[2].set_axisbelow(True)

fig.tight_layout()
fig.savefig("results/figure.png", dpi=220)
print("saved results/figure.png")
