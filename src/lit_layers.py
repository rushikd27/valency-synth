"""Valency feature layers and literature labels at composition level.

- w2v_*      : PCA of composed word2vec vectors + cosine to concept words
- nbr_*      : similarity to the nearest literature-synthesized composition
               (reference set excludes every composition in final_data.csv, so a
               row's own label or own mentions never enter its features)
- lit table  : per-composition mention counts / first synthesis-context date,
               used for external evaluation and label augmentation, not as inputs
"""
import sys

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.decomposition import PCA

sys.path.insert(0, "src")
from composition import parse_name  # noqa: E402
from embeddings import compose  # noqa: E402

CONCEPTS = ["synthesized", "stable", "unstable", "predicted", "photovoltaic", "single"]


def lit_table(cutoff=None):
    M = pd.read_csv("data/valency/mentions.csv")
    M["date"] = pd.to_datetime(M.date, errors="coerce", format="mixed")
    if cutoff:
        M = M[M.date < cutoff]
    S = M[M.synth_context]
    t = M.groupby("comp_key").agg(n_papers=("paper_id", "nunique"), first_mention=("date", "min"),
                                  formula=("formula_raw", "first"))
    t = t.join(S.groupby("comp_key").agg(n_synth=("paper_id", "nunique"), first_synth=("date", "min")))
    t["n_synth"] = t.n_synth.fillna(0).astype(int)
    return t


def w2v_features(names, model, n_pca=8):
    V = np.vstack([compose(model, n) for n in names])
    ok = np.isfinite(V).all(1)
    pcs = np.full((len(names), n_pca), np.nan)
    pcs[ok] = PCA(n_pca, random_state=0).fit_transform(V[ok])
    F = pd.DataFrame(pcs, columns=[f"w2v_pc{i}" for i in range(n_pca)])
    for w in CONCEPTS:
        if w in model.wv:
            u = model.wv[w] / np.linalg.norm(model.wv[w])
            F[f"w2v_cos_{w}"] = V @ u
    return F, V


def frac_vector(name, species):
    c = parse_name(name)
    tot = sum(c.values())
    return np.array([c.get(s, 0) / tot for s in species])


def neighbor_features(names, ref_names, V=None, ref_V=None):
    """Max / mean-top3 similarity to the reference (literature-synthesized) set,
    in element-fraction space (cosine) and, if given, in word2vec space."""
    species = sorted({s for n in list(names) + list(ref_names) for s in parse_name(n)})
    A = np.vstack([frac_vector(n, species) for n in names])
    R = np.vstack([frac_vector(n, species) for n in ref_names])
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    S = A @ R.T
    top = np.sort(S, axis=1)[:, ::-1]
    F = pd.DataFrame({"nbr_frac_max": top[:, 0], "nbr_frac_top3": top[:, :3].mean(1)})
    if V is not None:
        Sv = np.nan_to_num(V) @ np.nan_to_num(ref_V).T
        topv = np.sort(Sv, axis=1)[:, ::-1]
        F["nbr_w2v_max"], F["nbr_w2v_top3"] = topv[:, 0], topv[:, :3].mean(1)
    return F


def reference_set(lit, dataset_keys, min_synth=1):
    """Literature-synthesized compositions that are not in the dataset."""
    r = lit[(lit.n_synth >= min_synth) & ~lit.index.isin(dataset_keys)]
    names = []
    for f in r.formula:
        try:
            parse_name(f)
            names.append(f)
        except Exception:
            pass
    return names


def build_layers(df, cutoff=None):
    tag = cutoff or "all"
    model = Word2Vec.load(f"data/valency/w2v_{tag}.model")
    lit = lit_table(cutoff)
    W, V = w2v_features(df.Name.tolist(), model)
    ref = reference_set(lit, set(df.comp_key))
    refV = np.vstack([compose(model, n) for n in ref])
    N = neighbor_features(df.Name.tolist(), ref, V, refV)
    W.index = N.index = df.index
    return W, N, lit, len(ref)
