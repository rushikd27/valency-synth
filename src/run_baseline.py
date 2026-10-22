"""Baseline bagged-PU ablation on final_data.csv (no Valency features yet)."""
import json
import sys

import pandas as pd

sys.path.insert(0, "src")
from features import feature_sets, load_data  # noqa: E402
from pu import hidden_positive_cv  # noqa: E402

N_BAGS = int(sys.argv[1]) if len(sys.argv) > 1 else 100

df = load_data()
sets, ok = feature_sets(df)
df, sets = df[ok], {k: v[ok] for k, v in sets.items()}
print(f"rows={len(df)} positives={df.y.sum()} unlabeled={(df.y == 0).sum()} compositions={df.comp_key.nunique()}")

results = {}
for name, X in sets.items():
    m, hidden, unl = hidden_positive_cv(X.values, df.y.values, df.comp_key.values, n_bags=N_BAGS)
    results[name] = m
    print(f"{name:32s} " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()))

pd.DataFrame(results).T.to_csv("results/baseline_ablation.csv")
json.dump(results, open("results/baseline_ablation.json", "w"), indent=1)
