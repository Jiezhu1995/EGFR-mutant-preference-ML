#!/usr/bin/env python3
"""Combine seed 746 with repeated scaffold results and test reported pairs."""

# Resolve shared modules across the six workflow stages.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(p) for p in _Path(__file__).resolve().parents[1].glob("[0-9][0-9]_*") if p.is_dir()]

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t, ttest_rel, wilcoxon


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "figure2_regression"
SEEDS = [104, 410, 853, 746, 732, 360, 156, 556, 450, 727]


def holm(values):
    values = np.asarray(values, float)
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def load_family(family):
    rows = []
    for seed in SEEDS:
        if seed == 746:
            path = (RESULTS / "raw" / "fingerprint_and_encoding" / "fold_metrics.csv"
                    if family == "fingerprint" else
                    RESULTS / "raw" / "protein_representation" / "svr" / "fold_metrics.csv")
        else:
            path = RESULTS / "repeated_scaffold" / f"seed{seed}" / (
                "fingerprints_svr" if family == "fingerprint" else "protein_svr"
            ) / "fold_metrics.csv"
        frame = pd.read_csv(path)
        if "mutation_encoding" in frame:
            frame = frame.rename(columns={"mutation_encoding": "protein_representation"})
        if "pearson" not in frame:
            predictions = pd.read_csv(path.with_name("test_predictions.tsv.gz"), sep="\t")
            if "mutation_encoding" in predictions:
                predictions = predictions.rename(columns={"mutation_encoding": "protein_representation"})
            pearson = predictions.groupby(
                ["fingerprint", "protein_representation", "model", "fold", "target"]
            ).apply(lambda x: x.observed.corr(x.predicted), include_groups=False).rename("pearson")
            frame = frame.merge(pearson.reset_index(),
                                on=["fingerprint", "protein_representation", "model", "fold", "target"],
                                how="left", validate="one_to_one")
        group = ["fingerprint", "protein_representation", "model", "target"]
        frame = frame.groupby(group, as_index=False).agg(
            rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"),
            mae_mean=("mae", "mean"), mae_sd=("mae", "std"),
            pearson_mean=("pearson", "mean"), pearson_sd=("pearson", "std"),
            spearman_mean=("spearman", "mean"), spearman_sd=("spearman", "std"),
        )
        frame["seed"] = seed
        frame["family"] = family
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def summarize(frame):
    group = ["family", "fingerprint", "protein_representation", "model", "target"]
    records = []
    for keys, block in frame.groupby(group, dropna=False):
        if block.seed.nunique() != len(SEEDS):
            continue
        record = dict(zip(group, keys)); record["n_seeds"] = block.seed.nunique()
        for metric in ("rmse_mean", "mae_mean", "pearson_mean", "spearman_mean"):
            values = block[metric].dropna().to_numpy(float)
            mean = values.mean(); sd = values.std(ddof=1); sem = sd / np.sqrt(len(values))
            bound = t.ppf(0.975, len(values) - 1) * sem
            record.update({f"{metric}_across_seed_mean": mean,
                           f"{metric}_across_seed_sd": sd,
                           f"{metric}_ci95_low": mean - bound,
                           f"{metric}_ci95_high": mean + bound})
        records.append(record)
    return pd.DataFrame(records)


def paired_tests(frame):
    specs = [
        ("fingerprint", "fingerprint", "ecfp6", "cdk_standard", "explicit"),
        ("protein", "protein_representation", "explicit", "esm_difference", "ecfp6"),
    ]
    records = []
    for family, column, left, right, fixed in specs:
        base = frame[(frame.family == family) & (frame.model == "svr")]
        base = base[(base.protein_representation == fixed) if family == "fingerprint"
                    else (base.fingerprint == fixed)]
        for target in ("p_activity_mut", "delta_pactivity"):
            for metric in ("rmse_mean", "pearson_mean"):
                table = base[base.target == target].pivot(index="seed", columns=column, values=metric)
                table = table.loc[SEEDS, [left, right]].dropna()
                difference = table[left] - table[right]
                mean = difference.mean(); sd = difference.std(ddof=1)
                bound = t.ppf(0.975, len(difference) - 1) * sd / np.sqrt(len(difference))
                records.append({
                    "family": family, "target": target, "metric": metric,
                    "comparison_a": left, "comparison_b": right,
                    "n_seeds": len(difference), "mean_a_minus_b": mean,
                    "difference_ci95_low": mean - bound,
                    "difference_ci95_high": mean + bound,
                    "paired_t_raw_p": ttest_rel(table[left], table[right]).pvalue,
                    "wilcoxon_raw_p": wilcoxon(table[left], table[right], method="exact").pvalue,
                })
    result = pd.DataFrame(records)
    result["paired_t_holm_p"] = np.nan
    result["wilcoxon_holm_p"] = np.nan
    for (_, _), indices in result.groupby(["family", "metric"]).groups.items():
        result.loc[indices, "paired_t_holm_p"] = holm(result.loc[indices, "paired_t_raw_p"])
        result.loc[indices, "wilcoxon_holm_p"] = holm(result.loc[indices, "wilcoxon_raw_p"])
    return result


def deep_comparison():
    rows = []
    configurations = {
        "Best traditional (SVR)": ("svr", "ecfp6", "esm_difference"),
        "Multitask MLP": ("multitask_mlp", "ecfp6", "explicit_esm"),
        "Graph GIN": ("GIN_multitask", "learned_2d_graph", "explicit_esm"),
    }
    for seed in SEEDS:
        for label, (model, fingerprint, protein) in configurations.items():
            if seed == 746:
                if model == "svr":
                    path = RESULTS / "raw/protein_representation/svr/fold_metrics.csv"
                elif model == "multitask_mlp":
                    path = RESULTS / "raw/deep_learning/stage2_fingerprint_mlp/ecfp6__explicit_esm/fold_metrics.csv"
                else:
                    path = RESULTS / "raw/deep_learning/stage2_gin/explicit_esm/fold_metrics.csv"
            else:
                folder = {"svr": "protein_svr", "multitask_mlp": "mlp_ecfp6_explicit_esm",
                          "GIN_multitask": "gin_explicit_esm"}[model]
                path = RESULTS / "repeated_scaffold" / f"seed{seed}" / folder / "fold_metrics.csv"
            folds = pd.read_csv(path)
            block = folds[(folds.model == model) & (folds.fingerprint == fingerprint) &
                          (folds.protein_representation == protein)]
            for target, target_block in block.groupby("target"):
                rows.append({"seed": seed, "configuration": label, "target": target,
                             **{f"{metric}_mean": target_block[metric].mean()
                                for metric in ("rmse", "mae", "pearson", "spearman")}})
    frame = pd.DataFrame(rows)
    tests = []
    for target in ("p_activity_mut", "delta_pactivity"):
        for metric in ("rmse_mean", "pearson_mean"):
            table = frame[frame.target == target].pivot(index="seed", columns="configuration", values=metric)
            table = table.loc[SEEDS]
            for challenger in ("Multitask MLP", "Graph GIN"):
                difference = table["Best traditional (SVR)"] - table[challenger]
                mean = difference.mean(); sd = difference.std(ddof=1)
                bound = t.ppf(0.975, len(difference) - 1) * sd / np.sqrt(len(difference))
                tests.append({"target": target, "metric": metric,
                              "comparison_a": "Best traditional (SVR)", "comparison_b": challenger,
                              "n_seeds": len(difference), "mean_a_minus_b": mean,
                              "difference_ci95_low": mean - bound,
                              "difference_ci95_high": mean + bound,
                              "paired_t_raw_p": ttest_rel(table["Best traditional (SVR)"], table[challenger]).pvalue,
                              "wilcoxon_raw_p": wilcoxon(table["Best traditional (SVR)"], table[challenger], method="exact").pvalue})
    tests = pd.DataFrame(tests)
    for metric, indices in tests.groupby("metric").groups.items():
        tests.loc[indices, "paired_t_holm_p"] = holm(tests.loc[indices, "paired_t_raw_p"])
        tests.loc[indices, "wilcoxon_holm_p"] = holm(tests.loc[indices, "wilcoxon_raw_p"])
    return frame, tests


def main():
    combined = pd.concat([load_family("fingerprint"), load_family("protein")], ignore_index=True)
    output = RESULTS / "repeated_scaffold" / "summary"
    output.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output / "five_seed_configuration_metrics.csv", index=False)
    summarize(combined).to_csv(output / "five_seed_summary_95ci.csv", index=False)
    tests = paired_tests(combined)
    tests.to_csv(output / "five_seed_paired_significance.csv", index=False)
    deep_metrics, deep_tests = deep_comparison()
    deep_metrics.to_csv(output / "five_seed_deep_model_metrics.csv", index=False)
    deep_tests.to_csv(output / "five_seed_deep_model_significance.csv", index=False)
    print(tests.to_string(index=False))
    print("\nDeep-model comparisons\n", deep_tests.to_string(index=False))


if __name__ == "__main__":
    main()
