#!/usr/bin/env python3
"""Point-range Figure 2 using ten repeated scaffold-split seeds."""

# Resolve shared modules across the six workflow stages.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(p) for p in _Path(__file__).resolve().parents[1].glob("[0-9][0-9]_*") if p.is_dir()]

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.stats import pearsonr
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results" / "figure2_regression" / "summary" / "benchmark_master.csv"
REPEATED = ROOT / "results" / "figure2_regression" / "repeated_scaffold"
FIG2 = ROOT / "figures" / "Figure2"
SUPP = ROOT / "figures" / "Supplementary_Figures"
PREDICTIONS = ROOT / "results" / "figure2_regression" / "raw" / "protein_representation" / "svr" / "test_predictions.tsv.gz"
SEEDS = [104, 410, 853, 746, 732, 360, 156, 556, 450, 727]
SEED_METRICS = None

BLACK = "#202020"
LIGHT_GREY = "#D9DCDD"
LABEL_RED = "#A12622"
NATURE = ["#3B6FB6", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
          "#56B4E9", "#7A5195", "#5B6770", "#8C6D31"]
COLORS = {"Compound fingerprint + classical ML": BLACK,
          "Protein/mutation representation + baseline": BLACK,
          "Deep learning": BLACK, "WT no-mutation models": BLACK}
FP_ORDER = ["ecfp4", "ecfp6", "maccs", "rdkit", "atompair", "cdk_standard", "cdk_ecfp6"]
FP_LABEL = {"ecfp4": "ECFP4", "ecfp6": "ECFP6", "maccs": "MACCS",
            "rdkit": "RDKit topological", "atompair": "Atom pair",
            "cdk_standard": "CDK Standard", "cdk_ecfp6": "CDK ECFP6",
            "learned_2d_graph": "2D graph"}
PROT_ORDER = ["no_mutation", "hash", "explicit", "esm_difference", "explicit_esm"]
PROT_LABEL = {"no_mutation": "No mutation", "hash": "Hash token",
              "explicit": "Explicit token", "esm_difference": "ESM difference",
              "explicit_esm": "Explicit + ESM"}
MODEL_LABEL = {"ridge": "Ridge", "elasticnet": "Elastic Net", "knn": "kNN",
               "randomforest": "Random Forest", "extratrees": "Extra Trees",
               "histgradientboosting": "HistGradientBoosting", "svr": "RBF-SVR",
               "multitask_mlp": "Multitask MLP", "GIN_multitask": "Graph GIN"}
CLASSICAL = ["ridge", "elasticnet", "knn", "randomforest", "extratrees",
             "histgradientboosting", "svr"]


def build_repeated_master():
    """Return configuration summaries and the underlying seed-level metrics."""
    metric_columns = ["rmse_mean", "mae_mean", "pearson_mean", "spearman_mean"]
    rows = []
    for seed in SEEDS:
        if seed == 746:
            frame = pd.read_csv(DATA)
            frame = frame[["model", "compound_representation", "protein_representation",
                           "target", *metric_columns]].copy()
        else:
            seed_dir = REPEATED / f"seed{seed}"
            parts = []
            for path in sorted(seed_dir.glob("*/aggregate_metrics.csv")):
                block = pd.read_csv(path)
                if "mutation_encoding" in block:
                    block = block.rename(columns={"mutation_encoding": "protein_representation"})
                if "fingerprint" in block:
                    block = block.rename(columns={"fingerprint": "compound_representation"})
                required = ["model", "compound_representation", "protein_representation", "target"]
                if set(required).issubset(block.columns):
                    parts.append(block[[*required, *metric_columns]])
            frame = pd.concat(parts, ignore_index=True)
        frame["protein_representation"] = frame.protein_representation.replace({"none": "no_mutation"})
        frame["seed"] = seed
        rows.append(frame)
    seeds = pd.concat(rows, ignore_index=True).drop_duplicates(
        ["seed", "model", "compound_representation", "protein_representation", "target"])
    keys = ["model", "compound_representation", "protein_representation", "target"]
    seeds = seeds.groupby(keys, dropna=False).filter(lambda x: x.seed.nunique() == len(SEEDS))
    summaries = []
    for config, block in seeds.groupby(keys, dropna=False):
        record = dict(zip(keys, config)); record.update({"n_seeds": len(SEEDS), "seed": "10 repeated seeds",
                                                         "split": "repeated scaffold-based five-fold"})
        for metric in metric_columns:
            values = block.set_index("seed").loc[SEEDS, metric].to_numpy(float)
            mean = values.mean(); sem = values.std(ddof=1) / np.sqrt(len(values))
            half = 2.262157 * sem  # t(0.975, 9)
            record[metric] = mean
            record[metric.replace("_mean", "_sd")] = values.std(ddof=1)
            record[metric.replace("_mean", "_ci95_halfwidth")] = half
        summaries.append(record)
    summary = pd.DataFrame(summaries)
    summary["model_class"] = np.where(summary.model.isin(["multitask_mlp", "GIN_multitask"]),
                                      "deep_learning", "classical")
    summary["primary_eligible"] = "yes"
    return summary, seeds


def setup_style():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "axes.titleweight": "normal", "axes.labelweight": "normal",
        "axes.linewidth": 0.7, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3, "ytick.major.size": 0, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white",
    })


def save(fig, directory, stem):
    directory.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(directory / f"{stem}.{ext}", dpi=600)
    plt.close(fig)


def select_main_rows(data, target):
    rows = []
    # For every compound representation, retain its best observed classical
    # regressor with the explicit mutation token held as the protein context.
    x = data[(data.target == target) & data.model.isin(CLASSICAL) &
             (data.protein_representation == "explicit") &
             data.compound_representation.isin(FP_ORDER)].copy()
    for fp in FP_ORDER:
        hit = x[x.compound_representation == fp].nsmallest(1, "rmse_mean")
        if len(hit):
            hit = hit.copy(); hit["section"] = "Compound fingerprint + classical ML"
            hit["display"] = FP_LABEL[fp] + " · " + hit.model.map(MODEL_LABEL)
            rows.append(hit)

    # For every protein/mutation representation, retain the best classical
    # baseline while fixing the molecular representation to ECFP6.
    x = data[(data.target == target) & data.model.isin(CLASSICAL) &
             (data.compound_representation == "ecfp6") &
             data.protein_representation.isin(PROT_ORDER)].copy()
    for rep in PROT_ORDER:
        hit = x[x.protein_representation == rep].nsmallest(1, "rmse_mean")
        if len(hit):
            hit = hit.copy(); hit["section"] = "Protein/mutation representation + baseline"
            hit["display"] = PROT_LABEL[rep] + " · " + hit.model.map(MODEL_LABEL)
            rows.append(hit)

    # Show the best completed configuration for each deep architecture.
    x = data[(data.target == target) & (data.model_class == "deep_learning")].copy()
    for model in ("multitask_mlp", "GIN_multitask"):
        hit = x[x.model == model].nsmallest(1, "rmse_mean")
        if len(hit):
            hit = hit.copy(); hit["section"] = "Deep learning"
            hit["display"] = (hit.model.map(MODEL_LABEL) + " · " +
                              hit.compound_representation.map(FP_LABEL) + " + " +
                              hit.protein_representation.map(PROT_LABEL))
            rows.append(hit)
    return pd.concat(rows, ignore_index=True)


def wt_rows(data):
    # WT is restricted to models that receive no protein/mutation representation.
    x = data[(data.target == "p_activity_wt") &
             (data.protein_representation == "no_mutation") &
             data.model.isin(CLASSICAL)].copy()
    fps = []
    for fp in ("ecfp6", "cdk_standard"):
        hit = x[(x.compound_representation == fp) & (x.model == "svr")].copy()
        if len(hit):
            hit = hit.copy(); hit["section"] = "Compound fingerprint + classical ML"
            hit["display"] = FP_LABEL[fp] + " · " + hit.model.map(MODEL_LABEL)
            fps.append(hit)
    models = x[x.compound_representation == "ecfp6"].copy()
    models["section"] = "WT no-mutation models"
    models["display"] = models.model.map(MODEL_LABEL)
    models["model_order"] = models.model.map({m: i for i, m in enumerate(CLASSICAL)})
    models = models.sort_values("model_order")
    return pd.concat(fps + [models], ignore_index=True)


def y_layout(frame):
    positions, headers, cursor = [], [], 0.0
    for section in frame.section.drop_duplicates():
        block = frame[frame.section == section]
        headers.append((section, cursor - 0.70))
        positions.extend(cursor + np.arange(len(block)))
        cursor += len(block) + 1.25
    return np.array(positions), headers


def draw_metric_pair(ax_left, ax_right, frame, left_metric, right_metric,
                     left_sd, right_sd, title, letter, show_y=True):
    y, headers = y_layout(frame)
    colors = [COLORS[x] for x in frame.section]
    for i, (value, error, color) in enumerate(zip(frame[left_metric], frame[left_sd], colors)):
        if pd.notna(error):
            ax_left.errorbar(value, y[i], xerr=error, fmt="none", ecolor=color,
                             elinewidth=0.8, capsize=1.8, zorder=1)
    ax_left.scatter(frame[left_metric], y, c=colors, s=20, edgecolor="white",
                    linewidth=0.35, zorder=2)
    for i, (value, error, color) in enumerate(zip(frame[right_metric], frame[right_sd], colors)):
        if pd.notna(error):
            ax_right.errorbar(value, y[i], xerr=error, fmt="none", ecolor=color,
                              elinewidth=0.8, capsize=1.8, zorder=1)
    ax_right.scatter(frame[right_metric], y, c=colors, s=20, marker="D",
                     edgecolor="white", linewidth=0.35, zorder=2)
    for ax in (ax_left, ax_right):
        ax.set_ylim(y.max() + 0.75, -1.25)
        ax.grid(axis="x", color="#D9DEE5", lw=0.55)
        ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)
        for _, header_y in headers[1:]:
            ax.axhline(header_y - 0.15, color="#C8CDD4", lw=0.55)
    ax_left.set_yticks(y, frame.display if show_y else [])
    ax_right.set_yticks(y, [])
    ax_left.set_xlabel("RMSE (lower is better)" if left_metric == "rmse_mean" else "MAE (lower is better)")
    ax_right.set_xlabel("Pearson r (higher is better)" if right_metric == "pearson_mean"
                        else "Spearman ρ (higher is better)")
    ax_left.set_title(title, loc="left", fontweight="normal", pad=10)
    ax_left.text(-0.48, 1.06, letter, transform=ax_left.transAxes, fontsize=12,
                 fontweight="bold", va="top")
    xmin, xmax = ax_left.get_xlim()
    for section, header_y in headers:
        ax_left.text(xmin, header_y, section, color=COLORS[section], fontsize=9,
                     fontweight="normal", va="center", ha="left")
    ax_left.set_xlim(xmin, xmax)


def fixed_fingerprint_rows(data):
    return data[(data.model == "svr") & (data.protein_representation == "explicit") &
                data.compound_representation.isin(FP_ORDER) &
                data.target.isin(["p_activity_mut", "delta_pactivity"])].copy()


def fixed_protein_rows(data):
    return data[(data.model == "svr") & (data.compound_representation == "ecfp6") &
                data.protein_representation.isin(PROT_ORDER) &
                data.target.isin(["p_activity_mut", "delta_pactivity"])].copy()


def model_comparison_rows(data):
    """Return the best tested input configuration for every algorithm.

    Selection is based only on the lowest mean RMSE across the ten repeated
    scaffold partitions. The same selected row is subsequently used for all
    displayed metrics, including Pearson r, to avoid metric-specific cherry
    picking.
    """
    model_labels = {
        "ridge": "Ridge",
        "elasticnet": "Elastic net",
        "knn": "KNN",
        "randomforest": "Random forest",
        "extratrees": "ExtraTrees",
        "histgradientboosting": "HistGradientBoosting",
        "svr": "RBF-SVR",
        "multitask_mlp": "Multitask MLP",
        "GIN_multitask": "Graph GIN",
    }
    rows = []
    eligible = data[data.model.isin(model_labels)].copy()
    for target in ("p_activity_mut", "delta_pactivity"):
        for model, label in model_labels.items():
            candidates = eligible[(eligible.target == target) & (eligible.model == model)]
            if candidates.empty:
                continue
            selected = candidates.loc[[candidates.rmse_mean.idxmin()]].copy()
            selected["comparison"] = label
            selected["selection_rule"] = "lowest mean RMSE across ten scaffold seeds"
            rows.append(selected)
    return pd.concat(rows, ignore_index=True)


def bar_panel(ax, frame, category, order, labels, metric, error, title, letter):
    targets = [("p_activity_mut", "Mutant activity", "#3B6FB6"),
               ("delta_pactivity", "Selectivity (ΔpActivity)", "#269A8B")]
    x = np.arange(len(order)); width = 0.36
    for offset, (target, target_label, color) in zip((-width / 2, width / 2), targets):
        sub = frame[frame.target == target].set_index(category).reindex(order)
        ax.bar(x + offset, sub[metric], width, yerr=sub[error], color=color,
               edgecolor="white", linewidth=0.35, capsize=1.7,
               error_kw={"elinewidth": 0.7, "capthick": 0.7}, label=target_label)
    ax.set_xticks(x, labels, rotation=36 if len(order) > 4 else 25, ha="right")
    ax.set_ylabel("RMSE" if metric == "rmse_mean" else "Pearson r")
    ax.set_title(title, loc="left", fontweight="normal", pad=5)
    ax.text(-0.15, 1.08, letter, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top")
    ax.grid(axis="y", color="#D9DEE5", lw=0.55); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    if metric == "rmse_mean":
        ax.text(0.99, 0.97, "Lower is better", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color="#6F7782")
    else:
        ax.text(0.99, 0.97, "Higher is better", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color="#6F7782")


def fold_values(row, metric):
    column = metric if metric.endswith("_mean") else f"{metric}_mean"
    mask = ((SEED_METRICS.target == row.target) & (SEED_METRICS.model == row.model) &
            (SEED_METRICS.compound_representation == row.compound_representation) &
            (SEED_METRICS.protein_representation == row.protein_representation))
    values = SEED_METRICS.loc[mask].set_index("seed").loc[SEEDS, column]
    return values.to_numpy(float)


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, float)
    order = np.argsort(pvalues); adjusted = np.empty(len(pvalues))
    running = 0.0; n = len(pvalues)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (n - rank) * pvalues[idx]))
        adjusted[idx] = running
    return adjusted


def p_label(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "n.s."


def horizontal_panel(ax, frame, category, order, labels, metric, error,
                     title, letter, comparisons, bar_height=.58):
    task_info = [("p_activity_mut", "Mutant activity", BLACK),
                 ("delta_pactivity", "Selectivity (ΔpActivity)", BLACK)]
    color_map = dict(zip(order, NATURE[:len(order)]))
    y_positions, row_lookup, rows, cursor = [], {}, [], 0.0
    for target, task_label, task_color in task_info:
        sub = frame[frame.target == target].set_index(category).reindex(order).reset_index()
        for i, row in sub.iterrows():
            y_positions.append(cursor); row_lookup[(target, order[i])] = cursor
            rows.append((row, task_color, color_map[order[i]])); cursor += 1
        cursor += 1.1
    y_positions = np.asarray(y_positions)
    for y, (row, task_color, color) in zip(y_positions, rows):
        value, err = row[metric], row[error]
        ax.barh(y, value, height=bar_height, color=LIGHT_GREY, edgecolor=LIGHT_GREY, linewidth=0.65,
                xerr=err, capsize=2.0,
                error_kw={"elinewidth": 0.75, "capthick": 0.75}, zorder=1)
        values = fold_values(row, metric.replace("_mean", ""))
        jitter = np.linspace(-0.15, 0.15, len(values))
        ax.scatter(values, y + jitter, s=14, facecolor=color, edgecolor="white",
                   linewidth=0.35, alpha=0.90, zorder=4)
    ylabels = labels + labels
    ax.set_yticks(y_positions, ylabels)
    ax.invert_yaxis(); ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#E7E9EA", lw=0.55); ax.set_axisbelow(True)
    ax.set_xlabel({"rmse_mean": "RMSE", "pearson_mean": "Pearson correlation coefficient",
                                  "mae_mean": "MAE", "spearman_mean": "Spearman correlation coefficient"}[metric])
    ax.set_title(title, fontweight="normal", pad=4)
    ax.text(-0.24, 1.08, letter, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top")
    first_count = len(order)
    ax.text(0.0, -0.55, "Mutant activity", transform=ax.get_yaxis_transform(),
            ha="left", color=task_info[0][2], fontweight="normal", fontsize=9)
    ax.text(0.0, first_count + 0.55, "Selectivity (ΔpActivity)",
            transform=ax.get_yaxis_transform(), ha="left", color=task_info[1][2],
            fontweight="normal", fontsize=9)

    tests = []
    for target, _, _ in task_info:
        sub = frame[frame.target == target].set_index(category).reindex(order)
        ascending = metric in ("rmse_mean", "mae_mean")
        ranked_order = sub.sort_values(metric, ascending=ascending).index.tolist()
        for left, right in [tuple(ranked_order[:2])]:
            a = frame[(frame.target == target) & (frame[category] == left)].iloc[0]
            b = frame[(frame.target == target) & (frame[category] == right)].iloc[0]
            va, vb = fold_values(a, metric.replace("_mean", "")), fold_values(b, metric.replace("_mean", ""))
            tests.append((target, left, right,
                          wilcoxon(va, vb, alternative="two-sided", method="exact").pvalue))
    adjusted = holm_adjust([x[3] for x in tests])
    # Red labels identify the two leading configurations that are compared
    # statistically within each task and metric panel.
    tick_labels = ax.get_yticklabels()
    task_index = {target: i for i, (target, _, _) in enumerate(task_info)}
    for target, first, second, _ in tests:
        for item in (first, second):
            idx = task_index[target] * len(order) + order.index(item)
            tick_labels[idx].set_color(LABEL_RED)
    finite_right = max(float(row[metric] + (row[error] if pd.notna(row[error]) else 0)) for row, _, _ in rows)
    finite_left = min(float(row[metric] - (row[error] if pd.notna(row[error]) else 0)) for row, _, _ in rows)
    span = max(finite_right - finite_left, 0.05); base = finite_right + span * 0.08
    per_task_level = {task_info[0][0]: 0, task_info[1][0]: 0}
    for (target, left, right, _), p in zip(tests, adjusted):
        level = per_task_level[target]; per_task_level[target] += 1
        x = base + level * span * 0.09
        y1, y2 = row_lookup[(target, left)], row_lookup[(target, right)]
        ax.plot([x - span * 0.018, x, x, x - span * 0.018], [y1, y1, y2, y2],
                color="black", lw=0.65, clip_on=False)
        ax.text(x + span * 0.012, (y1 + y2) / 2, p_label(p), va="center", ha="left", fontsize=9)
    ax.set_xlim(finite_left - span * 0.12,
                base + (max(per_task_level.values()) + 1.3) * span * 0.16)
    return pd.DataFrame(tests, columns=["target", "comparison_a", "comparison_b", "raw_p"]).assign(
        holm_p=adjusted, metric=metric, panel=letter, n_seeds=len(SEEDS),
        test="two-sided exact Wilcoxon signed-rank")


def single_horizontal_panel(ax, frame, metric, error, title, letter, bar_height=.58, adjusted_p=None):
    y, headers = y_layout(frame); ascending = metric != "pearson_mean" and metric != "spearman_mean"
    fp_color = dict(zip(FP_ORDER, NATURE[:len(FP_ORDER)]))
    model_color = dict(zip(CLASSICAL, NATURE[:len(CLASSICAL)]))
    for i, (_, row) in enumerate(frame.iterrows()):
        color = (model_color.get(row.model, NATURE[-1])
                 if row.section == "WT no-mutation models"
                 else fp_color.get(row.compound_representation, NATURE[-1]))
        ax.barh(y[i], row[metric], height=bar_height, color=LIGHT_GREY, edgecolor=LIGHT_GREY, linewidth=0.65,
                xerr=row[error], capsize=1.6, error_kw={"elinewidth": 0.65, "capthick": 0.65})
        values = fold_values(row, metric.replace("_mean", ""))
        ax.scatter(values, y[i] + np.linspace(-0.15, 0.15, len(values)), s=14,
                   facecolor=color, edgecolor="white", linewidth=0.35, zorder=4)
    ax.set_yticks(y, frame.display); ax.invert_yaxis(); ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel({"rmse_mean": "RMSE", "pearson_mean": "Pearson correlation coefficient",
                   "mae_mean": "MAE", "spearman_mean": "Spearman correlation coefficient"}[metric])
    ax.set_title(title, fontweight="normal", pad=4)
    ax.text(-0.65, 1.08, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
    for section, header_y in headers:
        ax.text(0.0, header_y, section, transform=ax.get_yaxis_transform(),
                fontsize=9, fontweight="normal", va="center", ha="left")
    # Fingerprint comparison: ECFP6 versus CDK Standard.
    if len(frame) >= 2:
        for tick in ax.get_yticklabels()[:2]:
            tick.set_color(LABEL_RED)
        va, vb = fold_values(frame.iloc[0], metric.replace("_mean", "")), fold_values(frame.iloc[1], metric.replace("_mean", ""))
        p = wilcoxon(va, vb, alternative="two-sided", method="exact").pvalue
        right = max(float(r[metric] + r[error]) for _, r in frame.iterrows())
        left = min(float(r[metric] - r[error]) for _, r in frame.iterrows()); span = max(right - left, 0.05)
        x = right + span * 0.08; y1, y2 = y[0], y[1]
        ax.plot([x - span * 0.018, x, x, x - span * 0.018], [y1, y1, y2, y2], color="black", lw=0.65)
        ax.text(x + span * 0.012, (y1 + y2) / 2, p_label(p if adjusted_p is None else adjusted_p), va="center", fontsize=9)
        ax.set_xlim(left - span * 0.12, x + span * 0.18)
    return {"panel": letter, "metric": metric, "comparison_a": frame.iloc[0].display,
            "comparison_b": frame.iloc[1].display, "raw_p": p, "holm_p": p if adjusted_p is None else adjusted_p,
            "n_seeds": len(SEEDS), "test": "two-sided exact Wilcoxon signed-rank"}


def workflow_panel(ax):
    ax.set_axis_off()
    boxes = [
        (0.03, 0.20, 0.27, 0.59, "Compound representations\nECFP4 / ECFP6 / MACCS\nRDKit / atom pair / CDK", "#F1F2F2"),
        (0.36, 0.20, 0.27, 0.59, "Protein/mutation representations\nNo mutation / hash / explicit\nESM difference / explicit + ESM", "#F1F2F2"),
        (0.69, 0.28, 0.17, 0.43, "Repeated scaffold\n10 seeds × 5 folds", "#F1F2F2"),
        (0.91, 0.28, 0.075, 0.43, "Mutant\nΔpActivity", "#E4E5E5"),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                    transform=ax.transAxes, facecolor=color, edgecolor="#8E99A4", lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, transform=ax.transAxes,
                ha="center", va="center", fontsize=9)
    for start, end in ((0.30, 0.36), (0.63, 0.69), (0.86, 0.91)):
        ax.add_patch(FancyArrowPatch((start, 0.5), (end, 0.5), transform=ax.transAxes,
                                     arrowstyle="-|>", mutation_scale=7, color="#8E99A4", lw=0.7))
    ax.text(-0.097, 0.97, "a", transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top")
    ax.text(0.5, 0.04, "Identical compounds, endpoints and within-seed folds across all comparisons",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color="#6F7782")


def repeated_predictions(target, protein_representation):
    blocks = []
    for seed in SEEDS:
        path = (PREDICTIONS if seed == 746 else
                REPEATED / f"seed{seed}" / "protein_svr" / "test_predictions.tsv.gz")
        frame = pd.read_csv(path, sep="\t")
        frame["seed"] = seed; blocks.append(frame)
    predictions = pd.concat(blocks, ignore_index=True)
    selected = predictions[(predictions.fingerprint == "ecfp6") &
                           (predictions.protein_representation == protein_representation) &
                           (predictions.model == "svr") & (predictions.target == target)].copy()
    return selected.groupby("row_index", as_index=False).agg(
        observed=("observed", "mean"), predicted=("predicted", "mean"),
        prediction_sd_across_seeds=("predicted", "std"), n_seeds=("seed", "nunique"))


def scatter_panel(ax, predictions, target, protein_representation, title, letter,
                  axis_label, color):
    pred = repeated_predictions(target, protein_representation)
    rmse = np.sqrt(np.mean((pred.observed - pred.predicted) ** 2))
    r = pearsonr(pred.observed, pred.predicted).statistic
    ax.scatter(pred.observed, pred.predicted, s=8, alpha=0.18, color=color, edgecolors="none")
    low = min(pred.observed.min(), pred.predicted.min()); high = max(pred.observed.max(), pred.predicted.max())
    ax.plot([low, high], [low, high], ls="--", lw=0.8, color="#6F7782")
    ax.set_xlim(low, high); ax.set_ylim(low, high); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Observed {axis_label}"); ax.set_ylabel("Out-of-fold prediction")
    ax.set_title(title, loc="left", fontweight="normal", pad=5)
    ax.text(-0.17, 1.08, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
    ax.text(0.04, 0.95, f"n = {len(pred):,}\nRMSE = {rmse:.2f}\nPearson r = {r:.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    return pred


def main_figure(data):
    fps = fixed_fingerprint_rows(data)
    proteins = fixed_protein_rows(data)
    models = model_comparison_rows(data)
    fig = plt.figure(figsize=(8.27, 10.55))
    gs = fig.add_gridspec(4, 2, height_ratios=[0.58, 1.78, 1.58, 2.72],
                          hspace=0.46, wspace=0.46)
    workflow_panel(fig.add_subplot(gs[0, :]))
    axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]),
            fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1])]
    stats = []
    stats.append(horizontal_panel(axes[0], fps, "compound_representation", FP_ORDER,
                 [FP_LABEL[x] for x in FP_ORDER], "rmse_mean", "rmse_ci95_halfwidth",
                 "Compound fingerprints", "b", [("ecfp6", "cdk_standard")], bar_height=.68))
    stats.append(horizontal_panel(axes[1], fps, "compound_representation", FP_ORDER,
                 [FP_LABEL[x] for x in FP_ORDER], "pearson_mean", "pearson_ci95_halfwidth",
                 "Compound fingerprints", "c", [("ecfp6", "cdk_standard")], bar_height=.68))
    stats.append(horizontal_panel(axes[2], proteins, "protein_representation", PROT_ORDER,
                 [PROT_LABEL[x] for x in PROT_ORDER], "rmse_mean", "rmse_ci95_halfwidth",
                 "Protein/mutation representations", "d", [("explicit", "esm_difference")], bar_height=.68))
    stats.append(horizontal_panel(axes[3], proteins, "protein_representation", PROT_ORDER,
                 [PROT_LABEL[x] for x in PROT_ORDER], "pearson_mean", "pearson_ci95_halfwidth",
                 "Protein/mutation representations", "e", [("explicit", "esm_difference")], bar_height=.68))
    comparison_order = ["Ridge", "Elastic net", "KNN", "Random forest", "ExtraTrees",
                        "HistGradientBoosting", "RBF-SVR", "Multitask MLP", "Graph GIN"]
    model_comparisons = [("RBF-SVR", "ExtraTrees")]
    stats.append(horizontal_panel(axes[4], models, "comparison", comparison_order,
                 comparison_order, "rmse_mean", "rmse_ci95_halfwidth", "Classical and deep-learning models",
                 "f", model_comparisons, bar_height=.68))
    stats.append(horizontal_panel(axes[5], models, "comparison", comparison_order,
                 comparison_order, "pearson_mean", "pearson_ci95_halfwidth", "Classical and deep-learning models",
                 "g", model_comparisons, bar_height=.68))
    fig.subplots_adjust(left=0.12, right=0.975, bottom=0.055, top=0.985)
    source = pd.concat([fps.assign(comparison_family="fingerprint"),
                        proteins.assign(comparison_family="protein"),
                        models.assign(comparison_family="model")], ignore_index=True)
    FIG2.mkdir(parents=True, exist_ok=True)
    source.to_csv(FIG2 / "Figure2_source_data.csv", index=False)
    pd.concat(stats, ignore_index=True).to_csv(FIG2 / "Figure2_significance_tests.csv", index=False)
    save(fig, FIG2, "Figure2_paired_regression_representation_benchmark")


def supplementary_best_model_scatters():
    predictions = None
    configs = [
        ("p_activity_mut", "esm_difference", "Best mutant-activity model", "a", "mutant pActivity", "#4478B8"),
        ("delta_pactivity", "esm_difference", "Best selectivity model", "b", "ΔpActivity", "#279989"),
        ("p_activity_wt", "no_mutation", "Best WT-activity model", "c", "WT pActivity", "#7C5AA6"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(8.27, 3.05))
    source = []
    for ax, config in zip(axes, configs):
        pred = scatter_panel(ax, predictions, *config)
        pred = pred.copy(); pred["scatter_panel"] = config[3]
        source.append(pred)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.20, top=0.88, wspace=0.42)
    combined = pd.concat(source, ignore_index=True)
    combined.to_csv(SUPP / "Supplementary_Figure_best_models_scatter_source_data.tsv.gz",
                    sep="\t", index=False, compression="gzip")
    save(fig, SUPP, "Supplementary_Figure_best_models_OOF_scatter")


def supplementary_other_metrics(data):
    fps, proteins, models = fixed_fingerprint_rows(data), fixed_protein_rows(data), model_comparison_rows(data)
    # Match the physical panel widths and per-row bar heights used in Fig. 2.
    # The model-comparison row is taller because it contains more categories;
    # this keeps its bars and vertical spacing visually consistent with rows 1–2.
    fig = plt.figure(figsize=(8.27, 8.70))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.78, 1.58, 2.72],
                          hspace=0.46, wspace=0.46)
    axes = np.array([[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
                     [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
                     [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]])
    stats = []
    stats.append(horizontal_panel(axes[0, 0], fps, "compound_representation", FP_ORDER,
                 [FP_LABEL[x] for x in FP_ORDER], "mae_mean", "mae_ci95_halfwidth",
                 "Compound fingerprints", "a", [("ecfp6", "cdk_standard")], bar_height=.68))
    stats.append(horizontal_panel(axes[0, 1], fps, "compound_representation", FP_ORDER,
                 [FP_LABEL[x] for x in FP_ORDER], "spearman_mean", "spearman_ci95_halfwidth",
                 "Compound fingerprints", "b", [("ecfp6", "cdk_standard")], bar_height=.68))
    stats.append(horizontal_panel(axes[1, 0], proteins, "protein_representation", PROT_ORDER,
                 [PROT_LABEL[x] for x in PROT_ORDER], "mae_mean", "mae_ci95_halfwidth",
                 "Protein/mutation representations", "c", [("explicit", "esm_difference")], bar_height=.68))
    stats.append(horizontal_panel(axes[1, 1], proteins, "protein_representation", PROT_ORDER,
                 [PROT_LABEL[x] for x in PROT_ORDER], "spearman_mean", "spearman_ci95_halfwidth",
                 "Protein/mutation representations", "d", [("explicit", "esm_difference")], bar_height=.68))
    comp = ["Ridge", "Elastic net", "KNN", "Random forest", "ExtraTrees",
            "HistGradientBoosting", "RBF-SVR", "Multitask MLP", "Graph GIN"]
    pairs = []
    stats.append(horizontal_panel(axes[2, 0], models, "comparison", comp, comp, "mae_mean", "mae_ci95_halfwidth",
                 "Classical and deep-learning models", "e", pairs, bar_height=.68))
    stats.append(horizontal_panel(axes[2, 1], models, "comparison", comp, comp, "spearman_mean", "spearman_ci95_halfwidth",
                 "Classical and deep-learning models", "f", pairs, bar_height=.68))
    # Extra left margin protects the longest model label; the top and inter-row
    # spacing keep panel letters, titles and task headers from overlapping.
    fig.subplots_adjust(left=0.18, right=0.975, bottom=0.065, top=0.975)
    source = pd.concat([fps.assign(comparison_family="fingerprint"),
                        proteins.assign(comparison_family="protein"),
                        models.assign(comparison_family="model")], ignore_index=True)
    SUPP.mkdir(parents=True, exist_ok=True)
    source.to_csv(SUPP / "Supplementary_Figure_secondary_metrics_source_data.csv", index=False)
    pd.concat(stats, ignore_index=True).to_csv(
        SUPP / "Supplementary_Figure_secondary_metrics_significance.csv", index=False)
    save(fig, SUPP, "Supplementary_Figure_mutant_selectivity_secondary_metrics")


def supplementary_wt(data):
    from statsmodels.stats.multitest import multipletests
    wt = wt_rows(data)
    metrics = ["rmse", "pearson", "mae", "spearman"]
    raw_p = [wilcoxon(fold_values(wt.iloc[0], m), fold_values(wt.iloc[1], m),
                      alternative="two-sided", method="exact").pvalue for m in metrics]
    adjusted = multipletests(raw_p, method="holm")[1]
    fig, axes = plt.subplots(2, 2, figsize=(8.27, 6.25))
    stats = [single_horizontal_panel(axes[0, 0], wt, "rmse_mean", "rmse_ci95_halfwidth", "WT activity", "a", adjusted_p=adjusted[0]),
             single_horizontal_panel(axes[0, 1], wt, "pearson_mean", "pearson_ci95_halfwidth", "WT activity", "b", adjusted_p=adjusted[1]),
             single_horizontal_panel(axes[1, 0], wt, "mae_mean", "mae_ci95_halfwidth", "WT activity", "c", adjusted_p=adjusted[2]),
             single_horizontal_panel(axes[1, 1], wt, "spearman_mean", "spearman_ci95_halfwidth", "WT activity", "d", adjusted_p=adjusted[3])]
    for row in stats:
        row.update(correction="Holm", correction_family="SF2 four endpoint-metric comparisons", n_comparisons=4)
    fig.subplots_adjust(left=0.30, right=0.98, bottom=0.08, top=0.94, wspace=0.95, hspace=0.40)
    wt.to_csv(SUPP / "Supplementary_Figure_WT_source_data.csv", index=False)
    pd.DataFrame(stats).to_csv(SUPP / "Supplementary_Figure_WT_significance.csv", index=False)
    save(fig, SUPP, "Supplementary_Figure_WT_compound_and_model_benchmark")


def main():
    global SEED_METRICS
    setup_style()
    data, SEED_METRICS = build_repeated_master()
    FIG2.mkdir(parents=True, exist_ok=True)
    SEED_METRICS.to_csv(FIG2 / "Figure2_ten_seed_metrics.csv", index=False)
    main_figure(data)
    supplementary_other_metrics(data)
    supplementary_wt(data)
    supplementary_best_model_scatters()


if __name__ == "__main__":
    main()
