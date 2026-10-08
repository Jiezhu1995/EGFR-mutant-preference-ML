#!/usr/bin/env python3
"""Supplementary figure: secondary metrics for the ECFP6 mutant classifier benchmark."""

# Resolve shared modules across the six workflow stages.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(p) for p in _Path(__file__).resolve().parents[1].glob("[0-9][0-9]_*") if p.is_dir()]

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import exp, lgamma, log
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results" / "figure3_classifier" / "all_model_fold_metrics.csv"
OUT = ROOT / "figures" / "Supplementary_Figures" / "Supplementary_Figure5"

BLUE = "#5F8FC4"
LIGHT_BLUE = "#8CB7D1"
MID_BLUE = "#A8C6D7"
GREY = "#B7C5CA"
PURPLE = "#7C5AA6"
MUTED = "#6F7782"
BLACK = "#202020"
LIGHT_GREY = "#D9DCDD"
LABEL_RED = "#A12622"
MODELS = ["SVM", "LR", "XGBoost", "ANN", "KNN", "RF"]
NATURE = ["#3B6FB6", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
MODEL_COLORS = dict(zip(MODELS, NATURE))
METRICS = [
    ("accuracy", "Accuracy", "a"),
    ("balanced_accuracy", "Balanced accuracy", "b"),
    ("precision", "Precision", "c"),
    ("recall", "Recall", "d"),
]

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.linewidth": .75,
    "axes.titleweight": "normal", "axes.labelweight": "normal",
    "xtick.major.width": .65, "ytick.major.width": .65,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.facecolor": "white",
})


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, float)
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[idx]))
        adjusted[idx] = running
    return adjusted


def significance_label(p):
    if p < .001:
        return "***"
    if p < .01:
        return "**"
    if p < .05:
        return "*"
    return "n.s."


def _betacf(a, b, x):
    """Continued fraction for the incomplete beta function."""
    qab, qap, qam = a+b, a+1.0, a-1.0
    c = 1.0
    d = 1.0 - qab*x/qap
    if abs(d) < 3e-14:
        d = 3e-14
    d = 1.0/d
    h = d
    for m in range(1, 201):
        m2 = 2*m
        aa = m*(b-m)*x/((qam+m2)*(a+m2))
        d = 1.0 + aa*d
        if abs(d) < 3e-14: d = 3e-14
        c = 1.0 + aa/c
        if abs(c) < 3e-14: c = 3e-14
        d = 1.0/d; h *= d*c
        aa = -(a+m)*(qab+m)*x/((a+m2)*(qap+m2))
        d = 1.0 + aa*d
        if abs(d) < 3e-14: d = 3e-14
        c = 1.0 + aa/c
        if abs(c) < 3e-14: c = 3e-14
        d = 1.0/d
        delta = d*c; h *= delta
        if abs(delta-1.0) < 3e-12:
            break
    return h


def _betai(a, b, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    bt = exp(lgamma(a+b)-lgamma(a)-lgamma(b)+a*log(x)+b*log(1-x))
    if x < (a+1)/(a+b+2):
        return bt*_betacf(a, b, x)/a
    return 1-bt*_betacf(b, a, 1-x)/b


def paired_t_pvalue(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    if np.allclose(d.std(ddof=1), 0):
        return 1.0 if np.allclose(d.mean(), 0) else 0.0
    t = abs(d.mean()/(d.std(ddof=1)/np.sqrt(len(d))))
    df = len(d)-1
    return float(_betai(df/2.0, .5, df/(df+t*t)))


def summarize(frame):
    per_repeat = (frame.groupby(["model", "repeat"], as_index=False)
                  [[x[0] for x in METRICS]].mean())
    rows = []
    for model, block in per_repeat.groupby("model"):
        row = {"model": model, "n_repeats": block.repeat.nunique()}
        for metric, _, _ in METRICS:
            values = block[metric].to_numpy(float)
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_sd"] = values.std(ddof=1)
            row[f"{metric}_ci95"] = 2.262157 * values.std(ddof=1) / np.sqrt(len(values))
        rows.append(row)
    return pd.DataFrame(rows), per_repeat


def paired_p(per_repeat, first, second, metric):
    a = per_repeat[per_repeat.model == first].sort_values("repeat")[metric].to_numpy(float)
    b = per_repeat[per_repeat.model == second].sort_values("repeat")[metric].to_numpy(float)
    if len(a) != 10 or len(b) != 10:
        raise ValueError("Expected ten matched scaffold repeats")
    return float(wilcoxon(a, b, alternative="two-sided", method="exact").pvalue)


def panel(ax, summary, per_repeat, metric, title, letter, adjusted_p):
    ranked = summary.sort_values(f"{metric}_mean", ascending=False).reset_index(drop=True)
    order = ranked.model.tolist()
    top1, top2 = order[:2]
    means = ranked[f"{metric}_mean"].to_numpy(float)
    ci = ranked[f"{metric}_ci95"].to_numpy(float)
    y = np.arange(len(order))
    ax.barh(y, means, height=.58, color=LIGHT_GREY, edgecolor=LIGHT_GREY, linewidth=.65,
            xerr=ci, capsize=1.8,
            error_kw={"elinewidth": .7, "capthick": .7}, zorder=2)
    for i, model in enumerate(order):
        values = (per_repeat[per_repeat.model == model].sort_values("repeat")
                  [metric].to_numpy(float))
        ax.scatter(values, i + np.linspace(-.15, .15, len(values)), s=16,
                   facecolor=MODEL_COLORS[model], edgecolor="white", linewidth=.35, zorder=4)
    ax.set_yticks(y, order)
    for tick, model in zip(ax.get_yticklabels(), order):
        if model in (top1, top2):
            tick.set_color(LABEL_RED)
    ax.invert_yaxis()
    ax.set_xlabel(title)
    ax.set_title(title, fontweight="normal", pad=5)
    ax.text(-.18, 1.08, letter, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    low = float(np.min(means-ci)); high = float(np.max(means+ci)); span = max(high-low, .025)
    left = max(0, low-.14*span); right = min(1, high+.10*span)
    bracket_x = right + .07*span; hook = .016*span
    ax.plot([bracket_x-hook, bracket_x, bracket_x, bracket_x-hook],
            [0, 0, 1, 1], color="black", lw=.8, clip_on=False)
    ax.text(bracket_x+.018*span, .5, significance_label(adjusted_p),
            ha="left", va="center", fontsize=9, fontweight="normal", clip_on=False)
    ax.set_xlim(left, min(1, bracket_x+.14*span))
    return top1, top2


def main():
    frame = pd.read_csv(SOURCE)
    selected = frame[(frame.endpoint == "mutant") &
                     (frame.fingerprint == "ECFP6") &
                     (frame.strategy == "scaffold_5fold") &
                     (frame.model.isin(MODELS))].copy()
    summary, per_repeat = summarize(selected)

    comparisons = []
    raw_p = []
    for metric, _, _ in METRICS:
        order = summary.sort_values(f"{metric}_mean", ascending=False).model.tolist()
        first, second = order[:2]
        comparisons.append((metric, first, second))
        raw_p.append(paired_p(per_repeat, first, second, metric))
    adjusted = holm_adjust(raw_p)

    fig, axes = plt.subplots(2, 2, figsize=(8.27, 5.45))
    for ax, (metric, title, letter), adj in zip(axes.flat, METRICS, adjusted):
        panel(ax, summary, per_repeat, metric, title, letter, adj)
    fig.suptitle("ECFP6 mutant classifier: complementary performance metrics",
                 fontsize=9, fontweight="normal", y=.992)
    fig.text(.5, .015,
             "Bars show mean and 95% CI across ten repeated scaffold splits; points are seed-level five-fold means.",
             ha="center", va="bottom", fontsize=9, color=MUTED)
    fig.subplots_adjust(left=.15, right=.965, top=.90, bottom=.12, hspace=.34, wspace=.42)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "Supplementary_Figure_ECFP6_mutant_secondary_metrics"
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(fig)

    summary.to_csv(OUT / "Supplementary_Figure_metric_summary.csv", index=False)
    per_repeat.to_csv(OUT / "Supplementary_Figure_ten_repeat_source_metrics.csv", index=False)
    pd.DataFrame({
        "panel": [x[2] for x in METRICS],
        "metric": [x[0] for x in METRICS],
        "top_1": [x[1] for x in comparisons],
        "top_2": [x[2] for x in comparisons],
        "raw_p": raw_p,
        "holm_adjusted_p": adjusted,
        "n_paired_repeats": 10,
        "test": "two-sided exact Wilcoxon signed-rank",
    }).to_csv(OUT / "Supplementary_Figure_top1_vs_top2_significance.csv", index=False)


if __name__ == "__main__":
    main()
