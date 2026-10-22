"""Ablation of Valency feature layers + external and time-split evaluation.

1. Hidden-positive CV (as in run_baseline.py) for each feature set.
2. External test: compositions that are unlabeled in final_data.csv but have
   synthesis-context mentions in the Valency corpus. They never enter training
   as positives, so their OOB rank among the other unlabeled compositions is a
   label-independent check of the model.
3. Time split: layers rebuilt from pre-2020 abstracts only; scored against
   unlabeled compositions whose first synthesis-context mention is >= 2020.
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src")
from features import DFT_COLS, composition_features, load_data, structure_onehot  # noqa: E402
from lit_layers import build_layers  # noqa: E402
from pu import BaggedPU, hidden_positive_cv  # noqa: E402

N_BAGS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
CUTOFF = "2020-01-01"

df = load_data()
comp, ok = composition_features(df)
df, comp = df[ok], comp[ok]
base = pd.concat([comp, structure_onehot(df), df[DFT_COLS]], axis=1)


def sets_for(W, N):
    return {
        "base (comp+struct+DFT)": base,
        "base+w2v": pd.concat([base, W], axis=1),
        "base+nbr": pd.concat([base, N], axis=1),
        "base+w2v+nbr": pd.concat([base, W, N], axis=1),
    }


def external_eval(score, test_keys):
    """Composition-level (max over polymorphs) OOB rank of literature-confirmed
    unlabeled compositions against the remaining unlabeled compositions."""
    u = df.y.values == 0
    s = pd.Series(score[u], index=df.comp_key.values[u]).groupby(level=0).max()
    s = s[~s.index.isin(df.comp_key[df.y == 1])]  # compositions with no positive polymorph
    is_t = s.index.isin(test_keys)
    if is_t.sum() == 0:
        return {}
    pct = [(s[~is_t] < v).mean() for v in s[is_t]]
    return {"ext_auc": float(roc_auc_score(is_t, s.values)), "ext_median_pctile": float(np.median(pct)),
            "ext_recall@0.5": float((s[is_t] > 0.5).mean()), "n_ext": int(is_t.sum())}


results, scores = {}, {}
for split in ["all", CUTOFF]:
    W, N, lit, n_ref = build_layers(df, None if split == "all" else split)
    full_lit = lit if split == "all" else build_layers(df)[2]
    if split == "all":
        test = full_lit[full_lit.n_synth >= 1].index
    else:  # first synthesis-context mention after the cutoff, i.e. unseen by the layers
        test = full_lit[(full_lit.n_synth >= 1) & (full_lit.first_synth >= split)].index
    print(f"\n== layers from {split} abstracts: reference set={n_ref}, external test comps in data="
          f"{df.comp_key[(df.y == 0)].isin(test).groupby(df.comp_key).any().sum()}")
    for name, X in sets_for(W, N).items():
        if split != "all" and name == "base (comp+struct+DFT)" and f"all | {name}" in results:
            r = {k: v for k, v in results[f"all | {name}"].items() if not k.startswith("ext")}
        else:
            r, _, _ = hidden_positive_cv(X.values, df.y.values, df.comp_key.values, n_bags=N_BAGS)
        pu = BaggedPU(n_bags=N_BAGS * 2, seed=7).fit(X.values, df.y.values)
        r.update(external_eval(pu.oob_score_, test))
        results[f"{split} | {name}"] = r
        scores[f"{split} | {name}"] = pu.oob_score_
        print(f"{name:24s} " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                         for k, v in r.items()))

pd.DataFrame(results).T.to_csv("results/valency_ablation.csv")
json.dump(results, open("results/valency_ablation.json", "w"), indent=1)
out = df[["Name", "Crystal Structure", "source", "y", "comp_key"]].copy()
for k, v in scores.items():
    out[k] = v
out.to_csv("results/valency_oob_scores.csv", index=False)
