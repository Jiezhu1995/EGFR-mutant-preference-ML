#!/usr/bin/env python3
"""Supplementary source-control validation figure in the Figure 3 visual style."""

# Resolve shared modules across the six workflow stages.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(p) for p in _Path(__file__).resolve().parents[1].glob("[0-9][0-9]_*") if p.is_dir()]

from pathlib import Path
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results" / "figure3_classifier" / "source_control_fold_metrics.csv"
OUT = ROOT / "figures" / "Supplementary_Figures" / "Supplementary_Figure6"

BLUE = "#5F8FC4"
LIGHT_BLUE = "#8CB7D1"
MID_BLUE = "#A8C6D7"
GREY = "#B7C5CA"
PURPLE = "#7C5AA6"
INK = "#152535"
MUTED = "#6F7782"

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.linewidth": .75,
    "axes.titleweight": "normal", "axes.labelweight": "normal",
    "xtick.major.width": .65, "ytick.major.width": .65,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.facecolor": "white"
})

ORDER = ["ChEMBL", "Patent", "BindingDB", "Source-balanced", "Y-randomized"]
COLORS = [BLUE, LIGHT_BLUE, MID_BLUE, PURPLE, GREY]
BLACK = "#202020"
LIGHT_GREY = "#D9DCDD"
LABEL_RED = "#A12622"
COLORS = [LIGHT_GREY, LIGHT_GREY, LIGHT_GREY, BLACK, "#F2F2F2"]
NATURE = ["#3B6FB6", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]


def workflow_panel(ax):
    ax.axis("off")
    boxes = [
        (.03, .25, .18, .50, "Archived ECFP6\nmutant dataset", "#F1F2F2"),
        (.28, .17, .20, .66, "Source strata\nChEMBL / Patent /\nBindingDB", "#F1F2F2"),
        (.55, .17, .18, .66, "Source-balanced\nactive / inactive", "#F1F2F2"),
        (.80, .17, .17, .66, "Scaffold-CV\n10 seeds x 5 folds", "#E4E5E5"),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.012,rounding_size=.02",
                                    transform=ax.transAxes, facecolor=color,
                                    edgecolor="#8E99A4", lw=.8))
        ax.text(x+w/2, y+h/2, text, transform=ax.transAxes,
                ha="center", va="center", fontsize=9)
    for start, end in ((.21, .28), (.48, .55), (.73, .80)):
        ax.add_patch(FancyArrowPatch((start, .5), (end, .5), transform=ax.transAxes,
                                     arrowstyle="-|>", mutation_scale=8,
                                     color="#8E99A4", lw=.8))
    ax.text(-.076, .98, "a", transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top")


def independent_values(frame, metric):
    rows = []
    for dataset in ["ChEMBL", "Patent", "BindingDB"]:
        q = frame[(frame.experiment == "within_source_scaffold_cv") &
                  (frame.dataset == dataset)]
        vals = q.groupby("seed")[metric].mean().to_numpy(float)
        rows.extend({"group": dataset, "value": v} for v in vals)
    q = frame[frame.experiment == "source_balanced_scaffold_cv"]
    vals = q.groupby("seed")[metric].mean().to_numpy(float)
    rows.extend({"group": "Source-balanced", "value": v} for v in vals)
    q = frame[frame.experiment == "source_balanced_y_randomization"]
    rows.extend({"group": "Y-randomized", "value": v} for v in q[metric].to_numpy(float))
    return pd.DataFrame(rows)


def metric_panel(ax, values, metric, title, letter, adjusted_p):
    means = np.array([values.loc[values.group.eq(g), "value"].mean() for g in ORDER])
    cis = []
    for g in ORDER:
        x = values.loc[values.group.eq(g), "value"].to_numpy(float)
        tcrit = 2.262157 if len(x) == 10 else 2.093024
        cis.append(tcrit * x.std(ddof=1) / np.sqrt(len(x)))
    cis = np.asarray(cis)
    y = np.arange(len(ORDER))
    ax.barh(y, means, height=.58, color=LIGHT_GREY, edgecolor=LIGHT_GREY, linewidth=.65,
            xerr=cis, capsize=1.8, error_kw={"elinewidth": .7, "capthick": .7}, zorder=2)
    for i, g in enumerate(ORDER):
        x = values.loc[values.group.eq(g), "value"].to_numpy(float)
        offsets = np.linspace(-.14, .14, len(x))
        ax.scatter(x, i + offsets, s=16, facecolor=NATURE[i], edgecolor="white",
                   linewidth=.35, zorder=4)
    ax.set_yticks(y, ["ChEMBL", "Patent", "BindingDB*", "Source-balanced", "Y-randomized"])
    for tick, group in zip(ax.get_yticklabels(), ORDER):
        if group in ("Source-balanced", "Y-randomized"):
            tick.set_color(LABEL_RED)
    ax.invert_yaxis(); ax.set_title(title, fontweight="normal", pad=5); ax.set_xlabel(title)
    ax.text(-.18, 1.08, letter, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(False)
    if metric == "roc_auc":
        ax.set_xlim(.40, 1.02); bracket_x = 1.005; hook = .012
    else:
        ax.set_xlim(-.10, 1.02); bracket_x = .995; hook = .022
    ax.plot([bracket_x-hook, bracket_x, bracket_x, bracket_x-hook],
            [3, 3, 4, 4], color="black", lw=.8, clip_on=False)
    stars = "***" if adjusted_p < .001 else "**" if adjusted_p < .01 else "*" if adjusted_p < .05 else "n.s."
    ax.text(bracket_x + hook*.32, 3.5, stars, ha="left", va="center",
            fontsize=9, fontweight="normal", clip_on=False)
    return pd.DataFrame({"group": ORDER, "metric": metric, "mean": means,
                         "ci95": cis,
                         "n_independent_units": [10, 10, 10, 10, 20]})


def main():
    frame = pd.read_csv(SOURCE)
    fig = plt.figure(figsize=(8.27, 4.45))
    gs = fig.add_gridspec(2, 2, height_ratios=[.68, 2.2], hspace=.30, wspace=.45)
    workflow_panel(fig.add_subplot(gs[0, :]))
    ax_b, ax_c = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    roc = independent_values(frame, "roc_auc")
    mcc = independent_values(frame, "mcc")
    tests = []
    for metric, values in [("roc_auc", roc), ("mcc", mcc)]:
        observed = values.loc[values.group.eq("Source-balanced"), "value"].to_numpy()
        null = values.loc[values.group.eq("Y-randomized"), "value"].to_numpy()
        result = mannwhitneyu(observed, null, alternative="two-sided", method="exact")
        tests.append(dict(comparison="Source-balanced vs Y-randomized", metric=metric,
                          test="two-sided exact Mann-Whitney U", statistic=result.statistic,
                          n_observed_seeds=len(observed), n_null_permutations=len(null), exact_p=result.pvalue))
    statistics = pd.DataFrame(tests)
    statistics["holm_adjusted_p"] = multipletests(statistics.exact_p, method="holm")[1]
    summaries = [metric_panel(ax_b, roc, "roc_auc", "ROC-AUC", "b", statistics.holm_adjusted_p.iloc[0]),
                 metric_panel(ax_c, mcc, "mcc", "MCC", "c", statistics.holm_adjusted_p.iloc[1])]
    fig.text(.5, .055,
             "Bars show mean and 95% CI; points represent seed-level estimates or label permutations.\n"
             "*BindingDB n = 92; exploratory. ***P < 0.001, two-sided exact Mann-Whitney U test with Holm correction.",
             ha="center", va="bottom", fontsize=9, color=MUTED, linespacing=1.25)
    fig.subplots_adjust(left=.16, right=.965, top=.97, bottom=.22)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "Supplementary_Figure_source_controlled_scaffold_CV"
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(fig)
    pd.concat(summaries, ignore_index=True).to_csv(
        OUT / "Supplementary_Figure_source_control_summary.csv", index=False)
    pd.concat([roc.assign(metric="roc_auc"), mcc.assign(metric="mcc")], ignore_index=True).to_csv(
        OUT / "Supplementary_Figure_source_control_independent_values.csv", index=False)
    statistics.to_csv(OUT / "Supplementary_Figure_source_control_significance.csv", index=False)


if __name__ == "__main__":
    main()
