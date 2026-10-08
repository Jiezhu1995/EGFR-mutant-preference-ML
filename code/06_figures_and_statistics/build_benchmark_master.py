#!/usr/bin/env python3
"""Build publication-ready comparison tables from frozen seed-746 experiments."""

# Resolve shared modules across the six workflow stages.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(p) for p in _Path(__file__).resolve().parents[1].glob("[0-9][0-9]_*") if p.is_dir()]

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "figure2_regression" / "raw"
OUT = ROOT / "results" / "figure2_regression" / "summary"
FIELDS = [
    "experiment_family", "model_class", "model", "compound_representation",
    "protein_representation", "target", "rmse_mean", "rmse_sd", "mae_mean",
    "mae_sd", "pearson_mean", "pearson_sd", "spearman_mean", "spearman_sd",
    "primary_eligible", "exclusion_reason", "n_folds", "seed", "split", "source_file",
]


def value(row, name):
    raw = row.get(name, "")
    return "" if raw is None else str(raw)


def model_class(name):
    if name in {"multitask_mlp", "GIN_multitask"}:
        return "deep_learning"
    if name in {"ridge", "elasticnet"}:
        return "linear"
    if name in {"randomforest", "extratrees", "histgradientboosting"}:
        return "tree_ensemble"
    if name == "svr":
        return "kernel"
    return "other"


def normalize(path, family):
    rows = []
    fold_sd = {}
    fold_path = path.with_name("fold_metrics.csv")
    if fold_path.exists():
        grouped = {}
        with fold_path.open(newline="") as handle:
            for fold_row in csv.DictReader(handle):
                target = fold_row.get("target", "")
                protein = (fold_row.get("protein_representation") or
                           fold_row.get("mutation_encoding") or "")
                if protein == "none":
                    protein = "no_mutation"
                identity = (target, fold_row.get("model", ""),
                            fold_row.get("fingerprint", ""), protein)
                for metric in ("rmse", "mae", "pearson", "spearman"):
                    raw = fold_row.get(metric, "")
                    if raw not in ("", None):
                        grouped.setdefault(identity + (metric,), []).append(float(raw))
        for key, values in grouped.items():
            if len(values) > 1:
                fold_sd[key] = str(statistics.stdev(values))
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            protein = row.get("protein_representation") or row.get("mutation_encoding") or ""
            if protein == "none":
                protein = "no_mutation"
            name = row.get("model", "")
            target = row.get("target", "")
            eligible = not (target == "p_activity_wt" and protein != "no_mutation")
            sd_key = (target, name, row.get("fingerprint", ""), protein)
            rows.append({
                "experiment_family": family,
                "model_class": model_class(name),
                "model": name,
                "compound_representation": row.get("fingerprint", ""),
                "protein_representation": protein,
                "target": target,
                "rmse_mean": value(row, "rmse_mean"),
                "rmse_sd": value(row, "rmse_sd") or fold_sd.get(sd_key + ("rmse",), ""),
                "mae_mean": value(row, "mae_mean"),
                "mae_sd": value(row, "mae_sd") or fold_sd.get(sd_key + ("mae",), ""),
                "pearson_mean": value(row, "pearson_mean"),
                "pearson_sd": value(row, "pearson_sd") or fold_sd.get(sd_key + ("pearson",), ""),
                "spearman_mean": value(row, "spearman_mean"),
                "spearman_sd": value(row, "spearman_sd") or fold_sd.get(sd_key + ("spearman",), ""),
                "primary_eligible": "yes" if eligible else "no",
                "exclusion_reason": "" if eligible else "WT task received mutation-aware representation",
                "n_folds": "5", "seed": "746", "split": "scaffold-based five-fold",
                "source_file": str(path.relative_to(ROOT)),
            })
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = []
    for path in sorted((RAW / "classical").glob("*/aggregate_metrics.csv")):
        candidates.extend(normalize(path, "fingerprint_and_traditional_model"))
    for path in sorted((RAW / "protein_representation").glob("*/aggregate_metrics.csv")):
        candidates.extend(normalize(path, "protein_representation"))
    for path in sorted((RAW / "deep_learning").glob("*/*/aggregate_metrics.csv")):
        candidates.extend(normalize(path, "deep_learning"))

    # The protein-representation benchmark supersedes exact duplicate traditional
    # configurations because it contains the complete metric/SD schema.
    priority = {"deep_learning": 3, "protein_representation": 2,
                "fingerprint_and_traditional_model": 1}
    unique = {}
    for row in candidates:
        key = (row["model"], row["compound_representation"],
               row["protein_representation"], row["target"])
        old = unique.get(key)
        if old is None or priority[row["experiment_family"]] > priority[old["experiment_family"]]:
            unique[key] = row
    rows = sorted(unique.values(), key=lambda x: (x["target"], float(x["rmse_mean"]),
                                                  x["model"], x["compound_representation"]))

    with (OUT / "benchmark_master.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)

    top = []
    for target in ("p_activity_wt", "p_activity_mut", "delta_pactivity"):
        ranked = [dict(row) for row in rows
                  if row["target"] == target and row["primary_eligible"] == "yes"]
        for rank, row in enumerate(ranked, 1):
            row["rank_by_rmse"] = rank
        fields = ["rank_by_rmse"] + FIELDS
        with (OUT / f"ranking_{target}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(ranked)
        top.extend(ranked[:5])
    with (OUT / "top5_each_task.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank_by_rmse"] + FIELDS)
        writer.writeheader(); writer.writerows(top)

    coverage = {}
    for row in rows:
        key = (row["model_class"], row["model"], row["compound_representation"],
               row["protein_representation"])
        coverage.setdefault(key, set()).add(row["target"])
    with (OUT / "method_coverage.csv").open("w", newline="") as handle:
        fields = ["model_class", "model", "compound_representation",
                  "protein_representation", "targets", "n_targets"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for key, targets in sorted(coverage.items()):
            writer.writerow(dict(zip(fields[:4], key), targets=";".join(sorted(targets)),
                                 n_targets=len(targets)))

    manifest = {
        "seed": 746,
        "split": "frozen compound-scaffold five-fold",
        "targets": ["p_activity_wt", "p_activity_mut", "delta_pactivity"],
        "n_unique_configurations_by_target": {
            target: sum(row["target"] == target for row in rows)
            for target in ("p_activity_wt", "p_activity_mut", "delta_pactivity")
        },
        "deduplication_key": ["model", "compound_representation", "protein_representation", "target"],
        "source_priority": ["deep_learning", "protein_representation", "fingerprint_and_traditional_model"],
        "large_model_weights_in_git": False,
    }
    (OUT / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
