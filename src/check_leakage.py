"""Does DFT encode data source rather than chemistry?

1. Among positives only: can source (our_data vs external) be predicted from DFT?
2. Re-run the hidden-positive CV restricted to our_data rows (single DFT workflow).
"""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src")
from features import feature_sets, load_data, DFT_COLS  # noqa: E402
from pu import hidden_positive_cv  # noqa: E402

df = load_data()
sets, ok = feature_sets(df)
df, sets = df[ok], {k: v[ok] for k, v in sets.items()}

pos = df[df.y == 1].copy()
pos["external"] = (pos.source != "our_data").astype(int)
print("positives by source:", pos.source.value_counts().to_dict())
for cols, label in [(DFT_COLS, "DFT"), (list(sets["composition"].columns), "composition")]:
    X = pos[cols].values if label == "DFT" else sets["composition"].loc[pos.index].values
    p = cross_val_predict(HistGradientBoostingClassifier(max_iter=100), X, pos.external,
                          groups=pos.comp_key, cv=GroupKFold(5), method="predict_proba")[:, 1]
    print(f"AUC predicting external-source from {label} (positives only): {roc_auc_score(pos.external, p):.3f}")
print("\nDFT by group (mean):")
print(df.assign(grp=np.where(df.y == 0, "unlabeled", np.where(df.source == "our_data", "pos_our", "pos_external")))
        .groupby("grp")[DFT_COLS].agg(["mean", "median", "count"]).round(3).to_string())

sub = df.source == "our_data"
print(f"\nour_data only: rows={sub.sum()} positives={df[sub].y.sum()}")
for name in ["composition+structure", "composition+structure+DFT", "DFT only"]:
    m, _, _ = hidden_positive_cv(sets[name][sub].values, df[sub].y.values, df[sub].comp_key.values, n_bags=50)
    print(f"{name:28s} recall@0.5={m['recall@0.5']:.3f} pu_auc={m['pu_auc']:.3f} unl_pos_rate={m['unlabeled_pred_pos_rate@0.5']:.3f}")
