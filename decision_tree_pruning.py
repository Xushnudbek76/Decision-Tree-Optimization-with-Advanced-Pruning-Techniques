#!/usr/bin/env python3
"""
Decision Tree Optimization with Advanced Pruning Techniques
============================================================
Student : Abdugaffarov Khushnudbek (202537260)

Dataset : Breast Cancer Wisconsin (Diagnostic) - 569 patients, 30 numeric features
Task    : binary classification, positive class = malignant (1)

Pruning methods compared
  1. Unpruned CART                     (baseline, overfits)
  2. Pre-pruning (grid search)         max_depth / min_samples_split / min_samples_leaf / criterion
  3. Cost-complexity pruning (CCP)     alpha chosen by CV, "min-CV" rule and Breiman's "1-SE" rule
  4. Reduced Error Pruning (REP)       own implementation, uses a held-out pruning set
  5. Pessimistic Error Pruning (PEP)   own implementation, Quinlan's continuity correction + c*SE
  6. Random Forest                     ensemble reference (not a single tree)

Usage
  python decision_tree_pruning.py --demo     # ~15-30 s, single split, for the live demo
  python decision_tree_pruning.py --quick    # ~2-4 min, 5 repeated splits + noise study
  python decision_tree_pruning.py            # full run: 20 repeated splits + noise study

Outputs (folder results/): figures (PNG), metrics (CSV), selected hyper-parameters (JSON)
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import wilcoxon
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, train_test_split)
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
TEST_SIZE = 0.30
N_JOBS = -1
METHODS = ["Unpruned", "Pre-pruning", "CCP (min-CV)", "CCP (1-SE)", "REP", "PEP", "Random Forest"]
TREE_METHODS = METHODS[:-1]

# --------------------------------------------------------------------------- #
# Plot style (neutral surface, categorical colours in fixed order)
# --------------------------------------------------------------------------- #
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLORS = {
    "Unpruned": "#898781",
    "Pre-pruning": "#2a78d6",
    "CCP (min-CV)": "#eb6834",
    "CCP (1-SE)": "#1baf7a",
    "REP": "#eda100",
    "PEP": "#e87ba4",
    "Random Forest": "#008300",
}
BLUE_SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "grid.color": GRID, "grid.linewidth": 0.9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
        "font.family": "DejaVu Sans", "font.size": 13, "axes.titlesize": 15,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 13,
        "legend.frameon": False, "legend.fontsize": 12, "savefig.dpi": 170,
    })


def dot(ax, x, y, color, ms=9, **kw):
    ax.plot(x, y, "o", ms=ms, color=color, mec=SURFACE, mew=2, zorder=5, **kw)


# --------------------------------------------------------------------------- #
# 1. Dataset + preprocessing
# --------------------------------------------------------------------------- #
def load_and_preprocess(verbose=True):
    """Load data, check quality, encode target (1 = malignant). Returns X (DataFrame), y, report."""
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame.copy()
    df["target"] = 1 - df["target"]          # sklearn: 0 = malignant -> flip so malignant = 1 (positive class)
    report = {"shape_raw": df.shape, "missing_values": int(df.isna().sum().sum())}
    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = n_dup
    report["shape_clean"] = df.shape
    X = df.drop(columns="target")
    y = df["target"].to_numpy()
    report["class_counts"] = {"benign": int((y == 0).sum()), "malignant": int((y == 1).sum())}
    q1, q3 = X.quantile(0.25), X.quantile(0.75)
    iqr = q3 - q1
    outlier_mask = (X < q1 - 1.5 * iqr) | (X > q3 + 1.5 * iqr)
    report["rows_with_outlier_feature_pct"] = round(100 * outlier_mask.any(axis=1).mean(), 1)
    report["note_scaling"] = "not needed: tree splits are invariant to monotone feature scaling"
    if verbose:
        print("=" * 72)
        print("1. DATA + PREPROCESSING")
        print("=" * 72)
        for k, v in report.items():
            print(f"  {k:32s}: {v}")
        print("  split: stratified 70/30; median imputer fitted on train only (no-op here, kept for safety)")
    return X, y, report


def split_and_impute(X, y, seed, test_size=TEST_SIZE):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    imp = SimpleImputer(strategy="median").fit(X_tr)   # fit on train only -> no leakage
    return imp.transform(X_tr), imp.transform(X_te), y_tr, y_te


# --------------------------------------------------------------------------- #
# 2. Own implementation of REP and PEP on top of a grown CART tree
# --------------------------------------------------------------------------- #
class PrunableTree:
    """Editable copy of a fitted sklearn tree so that pruning rules can be implemented by hand."""
    LEAF = -1

    def __init__(self, clf):
        t = clf.tree_
        self.left, self.right = t.children_left.copy(), t.children_right.copy()
        self.feature, self.threshold = t.feature.copy(), t.threshold.copy()
        v = t.value[:, 0, :]
        self.counts = v / v.sum(1, keepdims=True) * t.n_node_samples[:, None]   # training class counts / node
        self.n_classes = self.counts.shape[1]

    # ---- inference
    def _walk(self, X, y=None):
        n_nodes = len(self.left)
        counts = np.zeros((n_nodes, self.n_classes)) if y is not None else None
        node = np.zeros(len(X), dtype=int)
        active = np.ones(len(X), dtype=bool)
        rows = np.arange(len(X))
        while active.any():
            if counts is not None:
                np.add.at(counts, (node[active], y[active]), 1)
            internal = self.left[node] != self.LEAF
            go_left = X[rows, self.feature[node]] <= self.threshold[node]
            nxt = np.where(go_left, self.left[node], self.right[node])
            active = active & internal
            node = np.where(active, nxt, node)
        return node, counts

    def predict_proba(self, X):
        leaf, _ = self._walk(X)
        c = self.counts[leaf]
        return c / c.sum(1, keepdims=True)

    def node_class_counts(self, X, y):
        """Number of (X, y) samples of each class that pass through every node."""
        return self._walk(X, y)[1]

    # ---- size
    def _reachable(self):
        out, stack = [], [0]
        while stack:
            i = stack.pop()
            out.append(i)
            if self.left[i] != self.LEAF:
                stack += [self.left[i], self.right[i]]
        return out

    @property
    def n_leaves(self):
        return sum(self.left[i] == self.LEAF for i in self._reachable())

    @property
    def depth(self):
        def d(i):
            return 0 if self.left[i] == self.LEAF else 1 + max(d(self.left[i]), d(self.right[i]))
        return d(0)

    def _make_leaf(self, i):
        self.left[i] = self.right[i] = self.LEAF

    # ---- Reduced Error Pruning (Quinlan 1987): bottom-up, validation errors
    def prune_rep(self, X_val, y_val):
        cnt = self.node_class_counts(X_val, y_val)
        majority = self.counts.argmax(1)                         # class the node predicts as a leaf
        leaf_err = cnt.sum(1) - cnt[np.arange(len(majority)), majority]

        def rec(i):
            if self.left[i] == self.LEAF:
                return leaf_err[i]
            sub = rec(self.left[i]) + rec(self.right[i])
            if leaf_err[i] <= sub:                               # ties -> prune (simpler tree)
                self._make_leaf(i)
                return leaf_err[i]
            return sub
        rec(0)
        return self

    # ---- Pessimistic Error Pruning (Quinlan 1987): top-down, training errors + 0.5/leaf, +c*SE
    def prune_pep(self, c=1.0):
        n = self.counts.sum(1)
        leaf_err = n - self.counts.max(1)
        sub_err = np.zeros(len(n))
        sub_leaves = np.zeros(len(n))

        def stats(i):
            if self.left[i] == self.LEAF:
                sub_err[i], sub_leaves[i] = leaf_err[i], 1
                return
            stats(self.left[i]); stats(self.right[i])
            sub_err[i] = sub_err[self.left[i]] + sub_err[self.right[i]]
            sub_leaves[i] = sub_leaves[self.left[i]] + sub_leaves[self.right[i]]
        stats(0)

        def visit(i):
            if self.left[i] == self.LEAF:
                return
            e_leaf = leaf_err[i] + 0.5                           # error if node becomes a leaf (corrected)
            e_sub = sub_err[i] + 0.5 * sub_leaves[i]             # error of its subtree (corrected)
            se = np.sqrt(e_sub * max(n[i] - e_sub, 0) / n[i])
            if e_leaf <= e_sub + c * se:
                self._make_leaf(i)
            else:
                visit(self.left[i]); visit(self.right[i])
        visit(0)
        return self


class _PrunedBase(ClassifierMixin, BaseEstimator):
    def predict_proba(self, X):
        return self.tree_.predict_proba(np.asarray(X, dtype=float))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]

    @property
    def n_leaves_(self):
        return self.tree_.n_leaves

    @property
    def depth_(self):
        return self.tree_.depth


class REPTreeClassifier(_PrunedBase):
    """Grow on (1 - val_size) of the training data, prune with the remaining val_size."""

    def __init__(self, val_size=0.3, criterion="gini", random_state=0):
        self.val_size, self.criterion, self.random_state = val_size, criterion, random_state

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_ = np.unique(y)
        Xg, Xv, yg, yv = train_test_split(X, y, test_size=self.val_size, stratify=y, random_state=self.random_state)
        grown = DecisionTreeClassifier(criterion=self.criterion, random_state=self.random_state).fit(Xg, yg)
        self.tree_ = PrunableTree(grown).prune_rep(Xv, yv)
        return self


class PEPTreeClassifier(_PrunedBase):
    """Grow on all training data, prune with Quinlan's pessimistic error estimate (no extra data needed)."""

    def __init__(self, c=1.0, criterion="gini", random_state=0):
        self.c, self.criterion, self.random_state = c, criterion, random_state

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_ = np.unique(y)
        grown = DecisionTreeClassifier(criterion=self.criterion, random_state=self.random_state).fit(X, y)
        self.tree_ = PrunableTree(grown).prune_pep(self.c)
        return self


# --------------------------------------------------------------------------- #
# 3. Fit + tune every model (all tuning uses CV on the TRAIN split only)
# --------------------------------------------------------------------------- #
PRE_GRID = {"criterion": ["gini", "entropy"], "max_depth": [2, 3, 4, 5, 6, 8, None],
            "min_samples_split": [2, 10, 20], "min_samples_leaf": [1, 2, 4, 8]}
REP_GRID = {"val_size": [0.2, 0.3, 0.4], "criterion": ["gini", "entropy"]}
PEP_GRID = {"c": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0], "criterion": ["gini", "entropy"]}
RF_GRID = {"max_depth": [None, 5], "max_features": ["sqrt", 0.5]}


def fit_all(X_tr, y_tr, seed, include_rf=True):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=seed)
    models, info, extras = {}, {}, {}

    def timed(name, fn):
        t0 = time.perf_counter()
        models[name], info[name] = fn()
        info[name]["fit_time_s"] = round(time.perf_counter() - t0, 3)

    # 1) baseline
    timed("Unpruned", lambda: (DecisionTreeClassifier(random_state=seed).fit(X_tr, y_tr), {"params": {}}))

    # 2) pre-pruning: grid search over stopping rules
    def pre():
        gs = GridSearchCV(DecisionTreeClassifier(random_state=seed), PRE_GRID, cv=cv,
                          scoring="accuracy", n_jobs=N_JOBS).fit(X_tr, y_tr)
        extras["pre_gs"] = gs
        return gs.best_estimator_, {"params": gs.best_params_, "cv_acc": gs.best_score_}
    timed("Pre-pruning", pre)

    # 3) cost-complexity pruning: alpha from the pruning path, picked by CV (min-CV and 1-SE rule)
    def ccp(rule):
        def run():
            if "ccp" not in extras:
                path = DecisionTreeClassifier(random_state=seed).cost_complexity_pruning_path(X_tr, y_tr)
                alphas = np.unique(np.clip(path.ccp_alphas, 0, None))[:-1]      # drop the root-only stump
                gs = GridSearchCV(DecisionTreeClassifier(random_state=seed), {"ccp_alpha": alphas}, cv=cv,
                                  scoring="accuracy", refit=False, n_jobs=N_JOBS).fit(X_tr, y_tr)
                mean, std = gs.cv_results_["mean_test_score"], gs.cv_results_["std_test_score"]
                i_min = np.where(mean >= mean.max() - 1e-12)[0].max()              # ties -> largest alpha
                se = std[i_min] / np.sqrt(cv.get_n_splits())
                i_1se = np.where(mean >= mean[i_min] - se)[0].max()                # simplest within 1 SE
                extras["ccp"] = {"alphas": alphas, "cv_mean": mean, "cv_std": std, "i_min": i_min, "i_1se": i_1se}
            e = extras["ccp"]
            a = e["alphas"][e["i_min"] if rule == "min" else e["i_1se"]]
            return (DecisionTreeClassifier(random_state=seed, ccp_alpha=a).fit(X_tr, y_tr),
                    {"params": {"ccp_alpha": float(a)}})
        return run
    timed("CCP (min-CV)", ccp("min"))
    timed("CCP (1-SE)", ccp("1se"))

    # 4) REP and 5) PEP (own implementations)
    def tuned(est, grid):
        def run():
            gs = GridSearchCV(est, grid, cv=cv, scoring="accuracy", n_jobs=N_JOBS).fit(X_tr, y_tr)
            return gs.best_estimator_, {"params": gs.best_params_, "cv_acc": gs.best_score_}
        return run
    timed("REP", tuned(REPTreeClassifier(random_state=seed), REP_GRID))
    timed("PEP", tuned(PEPTreeClassifier(random_state=seed), PEP_GRID))

    # 6) ensemble reference
    if include_rf:
        def rf():
            gs = GridSearchCV(RandomForestClassifier(n_estimators=150, random_state=seed, n_jobs=1), RF_GRID,
                              cv=StratifiedKFold(5, shuffle=True, random_state=seed), scoring="accuracy",
                              n_jobs=N_JOBS).fit(X_tr, y_tr)
            return gs.best_estimator_, {"params": gs.best_params_, "cv_acc": gs.best_score_}
        timed("Random Forest", rf)
    return models, info, extras


# --------------------------------------------------------------------------- #
# 4. Evaluation
# --------------------------------------------------------------------------- #
def tree_size(model):
    if hasattr(model, "get_n_leaves"):
        return model.get_n_leaves(), model.get_depth()
    if hasattr(model, "n_leaves_"):
        return model.n_leaves_, model.depth_
    return np.nan, np.nan


def evaluate(model, X_te, y_te):
    proba = model.predict_proba(X_te)[:, 1]
    pred = model.predict(X_te)
    leaves, depth = tree_size(model)
    return {
        "accuracy": accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred, zero_division=0),
        "recall": recall_score(y_te, pred),
        "f1": f1_score(y_te, pred),
        "auc": roc_auc_score(y_te, proba),
        "rmse": float(np.sqrt(np.mean((proba - y_te) ** 2))),     # RMSE of predicted P(malignant) vs label
        "leaves": leaves, "depth": depth,
    }


def run_repeats(X, y, n_rep, include_rf=True, noise=0.0, base_seed=0, tag=""):
    rows = []
    for r in range(n_rep):
        seed = base_seed + r
        X_tr, X_te, y_tr, y_te = split_and_impute(X, y, seed)
        if noise > 0:                                              # flip a fraction of TRAIN labels only
            flip = np.random.default_rng(1000 + seed).random(len(y_tr)) < noise
            y_tr = np.where(flip, 1 - y_tr, y_tr)
        models, info, _ = fit_all(X_tr, y_tr, seed, include_rf=include_rf)
        for name, m in models.items():
            rows.append({"repeat": r, "method": name, "noise": noise, **evaluate(m, X_te, y_te),
                         "fit_time_s": info[name]["fit_time_s"]})
        print(f"  {tag}repeat {r + 1}/{n_rep} done", end="\r", flush=True)
    print(" " * 40, end="\r")
    return pd.DataFrame(rows)


def paired_tests(df):
    base = df[df.method == "Unpruned"].set_index("repeat")
    rows = []
    for m in [m for m in df.method.unique() if m != "Unpruned"]:
        cur = df[df.method == m].set_index("repeat")
        for col in ("accuracy", "f1"):
            d = (cur[col] - base[col]).dropna()
            try:
                p = wilcoxon(d).pvalue
            except ValueError:
                p = np.nan
            rows.append({"method": m, "metric": col, "mean_diff_vs_unpruned": d.mean(), "p_value": p, "n": len(d)})
    return pd.DataFrame(rows)


def summarize(df):
    g = df.groupby("method")[["accuracy", "precision", "recall", "f1", "auc", "rmse", "leaves", "depth", "fit_time_s"]]
    out = g.mean().join(g.std(), rsuffix="_std")
    return out.reindex([m for m in METHODS if m in out.index])


# --------------------------------------------------------------------------- #
# 5. Figures
# --------------------------------------------------------------------------- #
def fig_data_overview(X, y, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1, 1.5]})
    counts = [(y == 0).sum(), (y == 1).sum()]
    a.barh([1, 0], counts, height=0.45, color=[COLORS["Pre-pruning"], COLORS["CCP (min-CV)"]])
    a.set_yticks([1, 0]); a.set_yticklabels(["Benign", "Malignant"])
    for yy, c in zip([1, 0], counts):
        a.text(c + 6, yy, f"{c}  ({100 * c / len(y):.1f}%)", va="center", color=INK)
    a.set_xlim(0, max(counts) * 1.3); a.set_title("Class balance (n = %d)" % len(y)); a.grid(axis="y", visible=False)
    a.set_xlabel("patients")
    cols = [c for c in X.columns if c.startswith("mean ")]
    corr = X[cols].corr().to_numpy()
    im = b.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
    b.set_xticks(range(len(cols))); b.set_xticklabels([c[5:] for c in cols], rotation=45, ha="right", fontsize=10.5)
    b.set_yticks(range(len(cols))); b.set_yticklabels([c[5:] for c in cols], fontsize=10.5)
    b.grid(False); b.set_title("Correlation of the 10 'mean' features")
    for s in b.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=b, fraction=0.046, pad=0.03); cb.outline.set_visible(False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_ccp_path(extras, X_tr, y_tr, X_te, y_te, seed, path):
    e = extras["ccp"]
    tr, te, lv = [], [], []
    for a in e["alphas"]:
        t = DecisionTreeClassifier(random_state=seed, ccp_alpha=a).fit(X_tr, y_tr)
        tr.append(t.score(X_tr, y_tr)); te.append(t.score(X_te, y_te)); lv.append(t.get_n_leaves())
    al = e["alphas"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11.5, 7.2), sharex=True, gridspec_kw={"height_ratios": [2.3, 1]})
    series = [("Train accuracy", tr, MUTED), ("5x2 CV accuracy", e["cv_mean"], "#6250d6"), ("Test accuracy", te, "#2a78d6")]
    for lab, v, col in series:
        a1.plot(al, v, color=col, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, label=lab)
    same = e["i_min"] == e["i_1se"]
    marks = ([("i_min", "min-CV = 1-SE", COLORS["CCP (1-SE)"], 0.40)] if same else
             [("i_min", "min-CV", COLORS["CCP (min-CV)"], 0.62), ("i_1se", "1-SE", COLORS["CCP (1-SE)"], 0.80)])
    for key, lab, col, yy in marks:
        for ax in (a1, a2):
            ax.axvline(al[e[key]], color=col, lw=1.8)
        a1.text(al[e[key]], a1.get_ylim()[0] + yy * (a1.get_ylim()[1] - a1.get_ylim()[0]),
                f"selected: {lab}\n  α={al[e[key]]:.4f}  " if False else f"  {lab}, α={al[e[key]]:.4f}",
                color=INK, fontsize=11.5, va="center", ha="right" if al[e[key]] > 0.01 else "left")
    a1.set_ylabel("accuracy"); a1.legend(loc="lower left", ncol=3, bbox_to_anchor=(0.0, 1.0))
    a2.plot(al, lv, color=INK2, lw=2, marker="o", ms=8, mec=SURFACE, mew=2)
    a2.set_ylabel("leaves"); a2.set_xscale("symlog", linthresh=1e-3)
    a2.set_xlabel("ccp_alpha  (larger α → heavier pruning; symlog scale)")
    fig.suptitle("Cost-complexity pruning path: training accuracy keeps falling, test/CV accuracy peaks at a small tree",
                 x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(path); plt.close(fig)


def fig_comparison(rep, path):
    metrics = [("accuracy", "Accuracy  (higher is better)"), ("f1", "F1, malignant  (higher is better)"),
               ("rmse", "RMSE of P(malignant)  (lower is better)")]
    n = len(METHODS)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.2), sharey=True)
    for ax, (col, title) in zip(axes, metrics):
        for i, m in enumerate(METHODS):
            v = rep[rep.method == m][col].to_numpy()
            yy = n - 1 - i
            ax.scatter(v, yy + rng.uniform(-0.17, 0.17, len(v)), s=20, color=COLORS[m], alpha=0.35, linewidths=0)
            mu, sd = v.mean(), v.std(ddof=1)
            ax.plot([mu - sd, mu + sd], [yy, yy], color=COLORS[m], lw=2.2, solid_capstyle="round")
            dot(ax, mu, yy, COLORS[m])
            ax.text(mu, yy + 0.30, f"{mu:.3f}", ha="center", va="bottom", fontsize=11, color=INK)
        ax.set_title(title, fontsize=13.5); ax.grid(axis="y", visible=False)
        ax.set_yticks(range(n)); ax.set_yticklabels(METHODS[::-1]); ax.set_ylim(-0.6, n - 0.2)
    fig.suptitle(f"{rep.repeat.nunique()} random 70/30 splits: dots = splits, marker = mean, bar = ±1 SD",
                 x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(path); plt.close(fig)


def fig_complexity(rep, path):
    s = summarize(rep).loc[TREE_METHODS]
    fig, ax = plt.subplots(figsize=(11, 6.4))
    offs = {"Unpruned": (12, -14, "left"), "Pre-pruning": (12, -20, "left"), "CCP (min-CV)": (0, -26, "center"),
            "CCP (1-SE)": (14, -14, "left"), "REP": (-10, 18, "right"), "PEP": (12, 14, "left")}
    for m, r in s.iterrows():
        ax.errorbar(r["leaves"], r["accuracy"], xerr=r["leaves_std"], yerr=r["accuracy_std"], fmt="none",
                    ecolor=COLORS[m], elinewidth=1.6, alpha=0.55, zorder=2)
        dot(ax, r["leaves"], r["accuracy"], COLORS[m], ms=13)
        ax.annotate(m, (r["leaves"], r["accuracy"]), xytext=offs[m][:2], textcoords="offset points",
                    color=INK, fontsize=12.5, va="center", ha=offs[m][2])
    ax.set_xlabel("mean number of leaves (tree size, bars ±1 SD)"); ax.set_ylabel("mean test accuracy (bars ±1 SD)")
    ax.set_title("Pruning shrinks the tree 1.7-4.8x; accuracy gaps stay within split-to-split noise")
    ax.set_xlim(left=0); fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_tuning_heatmap(gs, path):
    res = pd.DataFrame(gs.cv_results_["params"]).assign(score=gs.cv_results_["mean_test_score"])
    res["max_depth"] = res["max_depth"].map(lambda v: "None" if v is None or (isinstance(v, float) and np.isnan(v)) else str(int(v)))
    order = ["2", "3", "4", "5", "6", "8", "None"]
    piv = res.groupby(["max_depth", "min_samples_leaf"])["score"].max().unstack().reindex(order)
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    im = ax.imshow(piv.to_numpy() * 100, cmap=BLUE_SEQ, aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    vals = piv.to_numpy() * 100
    lo, hi = vals.min(), vals.max()
    bi = np.unravel_index(np.nanargmax(vals), vals.shape)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            frac = (vals[i, j] - lo) / (hi - lo + 1e-9)
            ax.text(j, i, f"{vals[i, j]:.1f}", ha="center", va="center", fontsize=12.5,
                    color="white" if frac > 0.45 else INK, fontweight="bold" if (i, j) == bi else "normal")
    ax.add_patch(plt.Rectangle((bi[1] - 0.5, bi[0] - 0.5), 1, 1, fill=False, ec=INK, lw=2.5))
    ax.set_xlabel("min_samples_leaf"); ax.set_ylabel("max_depth")
    ax.set_title("Pre-pruning grid: mean 5x2 CV accuracy (%)\n(best over criterion and min_samples_split)", fontsize=14)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.outline.set_visible(False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_noise(noise_df, path):
    methods = [m for m in TREE_METHODS if m in noise_df.method.unique()]
    g = noise_df.groupby(["method", "noise"])[["accuracy", "leaves"]].mean().reset_index()
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5.8))
    for m in methods:
        d = g[g.method == m]
        for ax, col in ((a, "accuracy"), (b, "leaves")):
            ax.plot(d.noise * 100, d[col], color=COLORS[m], lw=2.2, marker="o", ms=8, mec=SURFACE, mew=2, label=m)
    a.set_title("Test accuracy vs training-label noise"); b.set_title("Tree size vs training-label noise")
    a.set_ylabel("mean test accuracy"); b.set_ylabel("mean number of leaves")
    for ax in (a, b):
        ax.set_xlabel("% of TRAIN labels flipped (test labels stay clean)")
    a.legend(loc="lower left", ncol=2, fontsize=11.5)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_trees(models, feat, path_unpruned, path_pruned):
    t = models["Unpruned"]
    fig, ax = plt.subplots(figsize=(22, 9))
    plot_tree(t, ax=ax, feature_names=feat, class_names=["benign", "malignant"], impurity=False,
              label="none", fontsize=7, filled=False)
    ax.set_title(f"Unpruned CART: {t.get_n_leaves()} leaves, depth {t.get_depth()}", fontsize=18)
    fig.tight_layout(); fig.savefig(path_unpruned, dpi=140); plt.close(fig)
    p = models["CCP (1-SE)"]
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    plot_tree(p, ax=ax, feature_names=feat, class_names=["benign", "malignant"], impurity=False,
              label="none", fontsize=12, filled=False)
    ax.set_title(f"CCP (1-SE) pruned tree: {p.get_n_leaves()} leaves, depth {p.get_depth()}", fontsize=16)
    fig.tight_layout(); fig.savefig(path_pruned, dpi=180); plt.close(fig)


def fig_roc_cm(models, X_te, y_te, path):
    fig = plt.figure(figsize=(15.5, 5.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1, 1])
    ax = fig.add_subplot(gs[0])
    for m in METHODS:
        if m not in models:
            continue
        fpr, tpr, _ = roc_curve(y_te, models[m].predict_proba(X_te)[:, 1])
        auc = roc_auc_score(y_te, models[m].predict_proba(X_te)[:, 1])
        ax.plot(fpr, tpr, color=COLORS[m], lw=2, label=f"{m}  (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate (recall)")
    ax.set_title("ROC curves, test set"); ax.legend(loc="lower right", fontsize=10.5)
    for k, m in enumerate(["Unpruned", "CCP (1-SE)"], start=1):
        a = fig.add_subplot(gs[k])
        cm = confusion_matrix(y_te, models[m].predict(X_te))
        a.imshow(cm, cmap=BLUE_SEQ, vmin=0, vmax=cm.sum() * 0.7)
        a.grid(False)
        for s in a.spines.values():
            s.set_visible(False)
        for i in range(2):
            for j in range(2):
                a.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=22,
                       color="white" if cm[i, j] > cm.sum() * 0.3 else INK)
        a.set_xticks([0, 1]); a.set_xticklabels(["benign", "malignant"]); a.set_yticks([0, 1])
        a.set_yticklabels(["benign", "malignant"], rotation=90, va="center")
        a.set_xlabel("predicted"); a.set_ylabel("actual"); a.set_title(f"{m}", fontsize=14)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# --------------------------------------------------------------------------- #
# 6. Live-demo helper: explain one prediction
# --------------------------------------------------------------------------- #
def explain_prediction(clf, x, feat, label):
    t = clf.tree_
    print(f"\n   Decision path of {label} for one test patient:")
    for nid in clf.decision_path(x.reshape(1, -1)).indices:
        if t.children_left[nid] == -1:
            v = t.value[nid, 0] / t.value[nid, 0].sum()
            print(f"     -> LEAF: P(benign)={v[0]:.2f}  P(malignant)={v[1]:.2f}  "
                  f"=> {'MALIGNANT' if v[1] > v[0] else 'BENIGN'}")
        else:
            f, thr = t.feature[nid], t.threshold[nid]
            side = "<=" if x[f] <= thr else ">"
            print(f"     {feat[f]:<24s} = {x[f]:9.3f}  {side}  {thr:9.3f}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="single split only (fast, for the live demo)")
    ap.add_argument("--quick", action="store_true", help="5 repeated splits, 3 repeats per noise level")
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--noise-repeats", type=int, default=None)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    n_rep = args.repeats if args.repeats is not None else (0 if args.demo else 5 if args.quick else 20)
    n_noise = args.noise_repeats if args.noise_repeats is not None else (0 if args.demo else 3 if args.quick else 10)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    style()
    t_start = time.perf_counter()

    # ---- 1. data
    X_df, y, report = load_and_preprocess()
    feat = list(X_df.columns)
    fig_data_overview(X_df, y, out / "fig0_data_overview.png")
    X_tr, X_te, y_tr, y_te = split_and_impute(X_df.to_numpy(), y, SEED)
    print(f"  train {X_tr.shape}, test {X_te.shape}; malignant share train {y_tr.mean():.3f} / test {y_te.mean():.3f}")

    # ---- 2-5. models, tuning, evaluation on the main split
    print("\n" + "=" * 72 + "\n2-5. MODELS, TUNING (CV on train), EVALUATION (held-out test)\n" + "=" * 72)
    models, info, extras = fit_all(X_tr, y_tr, SEED)
    main_df = pd.DataFrame({m: evaluate(models[m], X_te, y_te) for m in METHODS}).T
    main_df["fit_time_s"] = [info[m]["fit_time_s"] for m in METHODS]
    main_df.to_csv(out / "metrics_main_split.csv", index_label="method")
    with open(out / "best_params.json", "w") as f:
        json.dump({m: info[m] for m in METHODS}, f, indent=2, default=str)
    print(main_df.round(4).to_string())
    print("\n  selected hyper-parameters:")
    for m in METHODS[1:]:
        print(f"   {m:<14s} {info[m]['params']}")

    # ---- 6. figures from the main split
    fig_ccp_path(extras, X_tr, y_tr, X_te, y_te, SEED, out / "fig1_ccp_pruning_path.png")
    fig_tuning_heatmap(extras["pre_gs"], out / "fig4_tuning_heatmap.png")
    fig_trees(models, feat, out / "fig6a_tree_unpruned.png", out / "fig6b_tree_pruned_ccp1se.png")
    fig_roc_cm(models, X_te, y_te, out / "fig7_roc_confusion.png")

    # ---- repeated random splits (robust comparison)
    rep = None
    if n_rep:
        print("\n" + "=" * 72 + f"\nREPEATED EVALUATION: {n_rep} random 70/30 splits (tuning redone on every split)\n" + "=" * 72)
        rep = run_repeats(X_df.to_numpy(), y, n_rep, tag="")
        rep.to_csv(out / "metrics_repeated_raw.csv", index=False)
        summ = summarize(rep)
        summ.to_csv(out / "metrics_repeated_summary.csv", index_label="method")
        tests = paired_tests(rep)
        tests.to_csv(out / "significance_tests.csv", index=False)
        show = summ[["accuracy", "accuracy_std", "f1", "f1_std", "auc", "rmse", "leaves", "depth", "fit_time_s"]]
        print(show.round(4).to_string())
        print("\n  paired Wilcoxon signed-rank vs Unpruned:")
        print(tests.round(4).to_string(index=False))
        fig_comparison(rep, out / "fig2_model_comparison.png")
        fig_complexity(rep, out / "fig3_complexity_vs_accuracy.png")

    # ---- label-noise robustness (where pruning matters most)
    if n_noise:
        print("\n" + "=" * 72 + f"\nLABEL-NOISE STUDY: {n_noise} splits per noise level\n" + "=" * 72)
        parts = []
        for p in [0.0, 0.05, 0.10, 0.20, 0.30]:
            parts.append(run_repeats(X_df.to_numpy(), y, n_noise, include_rf=False, noise=p, base_seed=100, tag=f"noise {p:.0%}: "))
        nd = pd.concat(parts)
        nd.to_csv(out / "noise_robustness_raw.csv", index=False)
        ns = nd.groupby(["noise", "method"])[["accuracy", "f1", "leaves"]].mean().unstack("method")
        ns.round(4).to_csv(out / "noise_robustness_summary.csv")
        print(nd.groupby(["noise", "method"])["accuracy"].mean().unstack("method")[TREE_METHODS].round(4).to_string())
        fig_noise(nd, out / "fig5_noise_robustness.png")

    # ---- live demo: how the pruned model reasons
    print("\n" + "=" * 72 + "\nDEMO: ONE PREDICTION + THE PRUNED TREE AS RULES\n" + "=" * 72)
    i = int(np.where(y_te == 1)[0][0])
    print(f"   true label of test patient #{i}: {'MALIGNANT' if y_te[i] else 'BENIGN'}")
    explain_prediction(models["Unpruned"], X_te[i], feat, "UNPRUNED tree")
    explain_prediction(models["CCP (1-SE)"], X_te[i], feat, "CCP (1-SE) pruned tree")
    print("\n   Pruned tree as rules:\n")
    print(export_text(models["CCP (1-SE)"], feature_names=feat, class_names=["benign", "malignant"], decimals=3))
    print(f"Total runtime: {time.perf_counter() - t_start:.1f} s   |   outputs in: {out.resolve()}")


if __name__ == "__main__":
    main()
