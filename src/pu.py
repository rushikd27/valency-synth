"""Bagged positive-unlabeled learning (Mordelet & Vert, Pattern Recognit. Lett. 2014).

Each bag trains a classifier on all positives vs. a bootstrap of unlabeled
examples the same size as the positive set; unlabeled examples are scored only
by bags that did not train on them (out-of-bag), and those scores are averaged.
"""
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier


def default_base(seed=0):
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
                                          l2_regularization=1.0, random_state=seed)


class BaggedPU:
    def __init__(self, base=None, n_bags=100, seed=0):
        self.base = base if base is not None else default_base(seed)
        self.n_bags = n_bags
        self.seed = seed

    def fit(self, X, y):
        """y: 1 = positive (synthesized), 0 = unlabeled."""
        X, y = np.asarray(X, float), np.asarray(y, int)
        rng = np.random.default_rng(self.seed)
        P, U = np.where(y == 1)[0], np.where(y == 0)[0]
        k = len(P)
        self.models_ = []
        oob_sum, oob_n = np.zeros(len(y)), np.zeros(len(y))
        for b in range(self.n_bags):
            ub = rng.choice(U, size=k, replace=True)
            idx = np.concatenate([P, ub])
            lab = np.concatenate([np.ones(k), np.zeros(k)])
            m = clone(self.base).set_params(random_state=self.seed + b)
            m.fit(X[idx], lab)
            self.models_.append(m)
            out = np.setdiff1d(U, ub)
            oob_sum[out] += m.predict_proba(X[out])[:, 1]
            oob_n[out] += 1
        with np.errstate(invalid="ignore"):
            self.oob_score_ = np.where(oob_n > 0, oob_sum / np.maximum(oob_n, 1), np.nan)
        self.oob_score_[P] = np.nan  # positives are never out-of-bag
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        return np.mean([m.predict_proba(X)[:, 1] for m in self.models_], axis=0)


def hidden_positive_cv(X, y, groups, n_folds=5, n_bags=100, seed=0, base=None):
    """Hide each fold of positive groups (relabel as unlabeled), train PU, and
    score the hidden positives out-of-bag alongside the true unlabeled set.

    Returns per-row OOB scores for the fold they were hidden in, plus metrics:
    recall@0.5 on hidden positives, their median percentile among unlabeled,
    and the PU-AUC (hidden positives vs. true unlabeled; a lower bound on AUC
    because some unlabeled rows are unknown positives).
    """
    X, y, groups = np.asarray(X, float), np.asarray(y, int), np.asarray(groups)
    rng = np.random.default_rng(seed)
    pos_groups = np.unique(groups[y == 1])
    rng.shuffle(pos_groups)
    folds = np.array_split(pos_groups, n_folds)
    hidden_score = np.full(len(y), np.nan)
    unl_scores = []
    for f, g in enumerate(folds):
        hid = np.isin(groups, g) & (y == 1)
        y_train = y.copy()
        y_train[hid] = 0
        pu = BaggedPU(base=base, n_bags=n_bags, seed=seed + 1000 * f).fit(X, y_train)
        hidden_score[hid] = pu.oob_score_[hid]
        true_unl = (y == 0)
        unl_scores.append(pu.oob_score_[true_unl])
    unl = np.nanmean(np.vstack(unl_scores), axis=0)
    hs = hidden_score[y == 1]
    hs = hs[np.isfinite(hs)]
    from sklearn.metrics import roc_auc_score
    pct = np.array([(unl < s).mean() for s in hs])
    metrics = {
        "recall@0.5": float((hs > 0.5).mean()),
        "median_pctile_vs_unlabeled": float(np.median(pct)),
        "pu_auc": float(roc_auc_score(np.r_[np.ones(len(hs)), np.zeros(len(unl))], np.r_[hs, unl])),
        "unlabeled_pred_pos_rate@0.5": float((unl > 0.5).mean()),
        "n_hidden_pos": int(len(hs)),
    }
    return metrics, hidden_score, unl
