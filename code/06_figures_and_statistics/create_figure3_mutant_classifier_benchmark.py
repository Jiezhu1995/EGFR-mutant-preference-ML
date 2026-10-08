#!/usr/bin/env python3
"""Figure 3: ECFP6 triple-mutant classifier benchmark in the Figure 2 bar style."""

# Resolve shared modules across the six workflow stages.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(p) for p in _Path(__file__).resolve().parents[1].glob("[0-9][0-9]_*") if p.is_dir()]

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results" / "figure3_classifier" / "all_model_fold_metrics.csv"
VALIDATION_SOURCE = (ROOT / "results" / "figure3_classifier" /
                     "paired_validation_strategy_metrics.csv")
OUT = ROOT / "figures" / "Figure3"

BLUE = "#5F8FC4"
LIGHT_BLUE = "#8CB7D1"
MID_BLUE = "#A8C6D7"
GREY = "#B7C5CA"
PURPLE = "#7C5AA6"
TEAL = "#279989"
INK = "#152535"
BLACK = "#202020"
LIGHT_GREY = "#D9DCDD"
LABEL_RED = "#A12622"
MUTED = "#6F7782"
GRID = "#D9DEE5"
MODELS = ["SVM", "LR", "XGBoost", "ANN", "KNN", "RF"]
SEEDS = [104, 410, 853, 746, 732, 360, 156, 556, 450, 727]
FINGERPRINTS = ["ECFP6", "CDK_Standard", "CDK_Extended", "CDK_Graph",
                "MACCS", "PubChem", "Substructure"]
FP_LABELS = ["ECFP6", "CDK Standard", "CDK Extended", "CDK Graph",
             "MACCS", "PubChem", "Substructure"]
NATURE = ["#3B6FB6", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#7A5195"]
FP_COLORS = dict(zip(FINGERPRINTS, NATURE))
MODEL_COLORS = dict(zip(MODELS, NATURE[:len(MODELS)]))

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.linewidth": .75,
    "axes.titleweight": "normal", "axes.labelweight": "normal",
    "xtick.major.width": .65, "ytick.major.width": .65,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.facecolor": "white"
})


def summarize(frame, category):
    # The statistical unit is one repeated scaffold split: first average its five folds.
    per_repeat = (frame.groupby([category, "repeat"], as_index=False)
                  [["roc_auc", "mcc", "accuracy", "balanced_accuracy", "precision", "recall"]]
                  .mean())
    rows = []
    for value, block in per_repeat.groupby(category):
        row = {category: value, "n_repeats": block.repeat.nunique()}
        for metric in ["roc_auc", "mcc", "accuracy", "balanced_accuracy", "precision", "recall"]:
            values = block[metric].to_numpy(float)
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_sd"] = values.std(ddof=1)
            row[f"{metric}_ci95"] = 2.262157 * values.std(ddof=1) / np.sqrt(len(values))
        rows.append(row)
    return pd.DataFrame(rows), per_repeat


def workflow_panel(ax):
    ax.axis("off")
    boxes = [
        (.015, .18, .19, .64,
         "Mutant-activity dataset\n$n$ = 2,099\n997 measured active\n339 measured inactive\n763 presumed-negative",
         "#F1F2F2"),
        (.285, .55, .205, .34,
         "Evaluation of seven fingerprints\nSVM held fixed",
         "#F1F2F2"),
        (.285, .11, .205, .34,
         "Evaluation of six classifiers\nECFP6 held fixed",
         "#F1F2F2"),
        (.585, .25, .19, .50,
         "Matched validation\nTen repeated five-fold\nscaffold partitions",
         "#F1F2F2"),
        (.855, .25, .13, .50,
         "Selected model\nECFP6-SVM\ndecision score $\\geq$ 0",
         "#E4E5E5"),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.012,rounding_size=.02",
                                    transform=ax.transAxes, facecolor=color,
                                    edgecolor="#8E99A4", lw=.8))
        ax.text(x+w/2, y+h/2, text, transform=ax.transAxes,
                ha="center", va="center", fontsize=9, linespacing=1.12)
    arrows = [
        ((.205, .50), (.285, .72)), ((.205, .50), (.285, .28)),
        ((.490, .72), (.585, .58)), ((.490, .28), (.585, .42)),
        ((.775, .50), (.855, .50)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, transform=ax.transAxes,
                                     arrowstyle="-|>", mutation_scale=8,
                                     color="#8E99A4", lw=.8,
                                     connectionstyle="arc3,rad=0"))
    ax.text(-.090, .98, "a", transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top")


def significance_label(p):
    if p < .001: return "***"
    if p < .01: return "**"
    if p < .05: return "*"
    return "n.s."


def metric_panel(ax, summary, repeat_values, metric, title, letter,
                 category, order, labels, highlight, comparison, adjusted_p):
    q = summary.set_index(category).reindex(order).copy()
    point_colors = FP_COLORS if category == "fingerprint" else MODEL_COLORS
    y = np.arange(len(order))
    means = q[f"{metric}_mean"].to_numpy(float)
    ci = q[f"{metric}_ci95"].to_numpy(float)
    ax.barh(y, means, height=.58, color=LIGHT_GREY, edgecolor=LIGHT_GREY, linewidth=.65,
            xerr=ci, capsize=1.8, error_kw={"elinewidth": .7, "capthick": .7}, zorder=2)
    for i, value in enumerate(order):
        values = (repeat_values[repeat_values[category] == value]
                  .sort_values("repeat")[metric].to_numpy(float))
        ax.scatter(values, i + np.linspace(-.15, .15, len(values)), s=16,
                   facecolor=point_colors[value], edgecolor="white", linewidth=.35, zorder=4)
    ax.set_yticks(y, labels); ax.invert_yaxis()
    for tick, value in zip(ax.get_yticklabels(), order):
        if value in comparison:
            tick.set_color(LABEL_RED)
    ax.set_xlabel(title)
    panel_title = "Structural representations" if category == "fingerprint" else "Classifier models"
    ax.set_title(panel_title, fontweight="normal", pad=5)
    ax.text(-.22, 1.08, letter, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top")
    ax.grid(False); ax.spines[["top", "right"]].set_visible(False)
    low = float(np.min(means-ci)); high = float(np.max(means+ci)); span = max(high-low, .02)
    left = max(0, low-.13*span)
    right = min(1, high+.12*span)
    # Paired top-versus-runner-up comparison across the same ten repeats.
    first, second = comparison
    y1, y2 = order.index(first), order.index(second)
    bracket_x = right + .07 * span
    hook = .016 * span
    ax.plot([bracket_x-hook, bracket_x, bracket_x, bracket_x-hook],
            [y1, y1, y2, y2], color="black", lw=.8, clip_on=False)
    ax.text(bracket_x + .018*span, (y1+y2)/2, significance_label(adjusted_p),
            ha="left", va="center", fontsize=9, fontweight="normal", clip_on=False)
    ax.set_xlim(left, min(1, bracket_x + .13*span))
    if letter == "d":
        ax.locator_params(axis="x", nbins=4)


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, float)
    order = np.argsort(pvalues); adjusted = np.empty(len(pvalues)); running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(pvalues)-rank)*pvalues[idx]))
        adjusted[idx] = running
    return adjusted


def paired_test(repeat_values, category, first, second, metric):
    a = repeat_values[repeat_values[category] == first].sort_values("repeat")[metric].to_numpy(float)
    b = repeat_values[repeat_values[category] == second].sort_values("repeat")[metric].to_numpy(float)
    if len(a) != len(b) or len(a) != 10:
        raise ValueError("Top-model significance test requires ten matched repeats")
    return float(wilcoxon(a, b, alternative="two-sided", method="exact").pvalue)


def validation_values(frame, metric):
    q = (frame.groupby(["strategy", "seed"], as_index=False)[metric].mean()
         .sort_values(["strategy", "seed"]))
    random = (q[q.strategy == "random_stratified_5fold"].set_index("seed")
              .loc[SEEDS, metric].to_numpy(float))
    scaffold = (q[q.strategy == "scaffold_5fold"].set_index("seed")
                .loc[SEEDS, metric].to_numpy(float))
    return random, scaffold


def validation_panel(ax, frame, metric, title, letter, raw_p, adjusted_p):
    random, scaffold = validation_values(frame, metric)
    groups = [random, scaffold]
    labels = ["Random stratified\n10 seeds x 5 folds",
              "Scaffold-based\n10 seeds x 5 folds"]
    means = np.array([x.mean() for x in groups])
    # t critical values: df=4 and df=9.
    ci = np.array([2.776445*x.std(ddof=1)/np.sqrt(len(x)) if len(x) == 5 else
                   2.262157*x.std(ddof=1)/np.sqrt(len(x)) for x in groups])
    y = np.arange(2)
    ax.barh(y, means, height=.58, color=LIGHT_GREY, edgecolor=LIGHT_GREY,
            linewidth=.45, xerr=ci, capsize=1.8,
            error_kw={"elinewidth": .7, "capthick": .7}, zorder=2)
    for i, values in enumerate(groups):
        ax.scatter(values, i + np.linspace(-.15, .15, len(values)), s=16,
                   facecolor=NATURE[i], edgecolor="white", linewidth=.35, zorder=4)
    ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlabel(title)
    ax.set_title("ECFP6-SVM validation", fontweight="normal", pad=5)
    ax.text(-.22, 1.48, letter, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top")
    ax.spines[["top", "right"]].set_visible(False)
    # Use the same honest scale in panels f and g so the stricter scaffold
    # result is not visually exaggerated by a truncated high-value axis.
    bracket_x = .995
    ax.plot([bracket_x-.008, bracket_x, bracket_x, bracket_x-.008],
            [0, 0, 1, 1], color="black", lw=.8, clip_on=False)
    ax.text(bracket_x+.004, .5, significance_label(adjusted_p),
            ha="left", va="center", fontsize=9, fontweight="normal", clip_on=False)
    ax.set_xlim(.70, 1.00)
    return pd.DataFrame({"strategy": ["random_stratified_5fold", "scaffold_5fold"],
                         "metric": metric, "mean": means, "ci95": ci,
                         "n_independent_units": [len(random), len(scaffold)],
                         "wilcoxon_raw_p": raw_p,
                         "holm_adjusted_p": adjusted_p})


def main():
    frame = pd.read_csv(SOURCE)
    validation_frame = pd.read_csv(VALIDATION_SOURCE)
    mutant = frame[frame.endpoint == "mutant"].copy()
    scaffold = mutant[mutant.strategy == "scaffold_5fold"].copy()
    fingerprint_frame = scaffold[scaffold.model == "SVM"].copy()
    algorithm_frame = scaffold[scaffold.fingerprint == "ECFP6"].copy()
    fp_summary, fp_repeats = summarize(fingerprint_frame, "fingerprint")
    model_summary, model_repeats = summarize(algorithm_frame, "model")

    fig = plt.figure(figsize=(8.27, 8.85))
    gs = fig.add_gridspec(4, 2, height_ratios=[.82, 1.2, 1.2, 1.2],
                          hspace=.42, wspace=.43)
    workflow_panel(fig.add_subplot(gs[0, :]))
    axes = [fig.add_subplot(gs[r, c]) for r in range(1, 4) for c in range(2)]
    comparisons = [
        ("fingerprint", "ECFP6", "CDK_Standard", "roc_auc", fp_repeats),
        ("fingerprint", "ECFP6", "CDK_Standard", "mcc", fp_repeats),
        ("model", "SVM", "LR", "roc_auc", model_repeats),
        ("model", "SVM", "LR", "mcc", model_repeats),
    ]
    raw_p = [paired_test(values, category, first, second, metric)
             for category, first, second, metric, values in comparisons]
    adjusted_p = holm_adjust(raw_p)
    metric_panel(axes[0], fp_summary, fp_repeats, "roc_auc", "ROC-AUC", "b",
                 "fingerprint", FINGERPRINTS, FP_LABELS, "ECFP6",
                 ("ECFP6", "CDK_Standard"), adjusted_p[0])
    metric_panel(axes[1], fp_summary, fp_repeats, "mcc", "MCC", "c",
                 "fingerprint", FINGERPRINTS, FP_LABELS, "ECFP6",
                 ("ECFP6", "CDK_Standard"), adjusted_p[1])
    metric_panel(axes[2], model_summary, model_repeats, "roc_auc", "ROC-AUC", "d",
                 "model", MODELS, MODELS, "SVM", ("SVM", "LR"), adjusted_p[2])
    metric_panel(axes[3], model_summary, model_repeats, "mcc", "MCC", "e",
                 "model", MODELS, MODELS, "SVM", ("SVM", "LR"), adjusted_p[3])
    validation_raw_p = []
    for metric in ("roc_auc", "mcc"):
        random, scaffold_values = validation_values(validation_frame, metric)
        validation_raw_p.append(float(wilcoxon(
            random, scaffold_values, alternative="two-sided", method="exact").pvalue))
    validation_adjusted_p = holm_adjust(validation_raw_p)
    validation = [validation_panel(axes[4], validation_frame, "roc_auc", "ROC-AUC", "f",
                                   validation_raw_p[0], validation_adjusted_p[0]),
                  validation_panel(axes[5], validation_frame, "mcc", "MCC", "g",
                                   validation_raw_p[1], validation_adjusted_p[1])]

    fig.subplots_adjust(left=.15, right=.975, top=.985, bottom=.065)
    # Panels f and g contain only two rows. Compress their axes vertically so
    # bar thickness and centre-to-centre spacing match the six-row panels,
    # rather than drawing unusually thin bars in a full-height axes box.
    for ax in axes[4:6]:
        pos = ax.get_position()
        new_height = pos.height * .36
        new_y0 = pos.y0 + (pos.height - new_height) / 2
        ax.set_position([pos.x0, new_y0, pos.width, new_height])
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "Figure3_ECFP6_mutant_classifier_benchmark"
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(fig)
    fp_summary.to_csv(OUT / "Figure3_fingerprint_metric_summary.csv", index=False)
    fp_repeats.to_csv(OUT / "Figure3_fingerprint_ten_repeat_metrics.csv", index=False)
    model_summary.to_csv(OUT / "Figure3_model_metric_summary.csv", index=False)
    model_repeats.to_csv(OUT / "Figure3_model_ten_repeat_metrics.csv", index=False)
    pd.concat(validation, ignore_index=True).to_csv(
        OUT / "Figure3_validation_strategy_summary.csv", index=False)
    pd.DataFrame({
        "panel": ["b", "c", "d", "e"],
        "family": ["fingerprint", "fingerprint", "algorithm", "algorithm"],
        "metric": ["roc_auc", "mcc", "roc_auc", "mcc"],
        "top_1": ["ECFP6", "ECFP6", "SVM", "SVM"],
        "top_2": ["CDK_Standard", "CDK_Standard", "LR", "LR"],
        "raw_p": raw_p, "holm_adjusted_p": adjusted_p,
        "n_paired_repeats": 10,
        "test": "two-sided exact Wilcoxon signed-rank",
    }).to_csv(OUT / "Figure3_top1_vs_top2_significance.csv", index=False)


if __name__ == "__main__":
    main()
