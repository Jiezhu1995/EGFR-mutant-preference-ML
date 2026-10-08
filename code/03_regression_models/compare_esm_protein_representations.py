#!/usr/bin/env python3
"""Compare mutation/protein representations with frozen seed-746 scaffold folds."""

# Resolve shared modules across the six workflow stages.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(p) for p in _Path(__file__).resolve().parents[1].glob("[0-9][0-9]_*") if p.is_dir()]

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, normalize

from compare_regression_representations import (
    ENDPOINTS, compound_features, context_features, make_model, mols,
)


TARGETS = ["p_activity_wt", "p_activity_mut", "delta_pactivity"]
MUTANT_REPRESENTATIONS = ["no_mutation", "hash", "explicit", "esm_difference", "explicit_esm"]


def scores(y, prediction):
    return {
        "rmse": float(mean_squared_error(y, prediction) ** 0.5),
        "mae": float(mean_absolute_error(y, prediction)),
        "pearson": float(pearsonr(y, prediction).statistic),
        "spearman": float(spearmanr(y, prediction).statistic),
    }


def endpoint_features(frame):
    values = np.zeros((len(frame), len(ENDPOINTS)), dtype=np.float32)
    codes = frame.endpoint.map(ENDPOINTS)
    values[np.arange(len(frame)), codes.astype(int)] = 1
    return values


def load_esm(path, frame):
    cache = np.load(path, allow_pickle=True)
    tokens = [str(x) for x in cache["variant_tokens"]]
    lookup = {token: i for i, token in enumerate(tokens)}
    missing = sorted(set(frame.v2_variant_token.astype(str)) - set(tokens))
    if missing:
        raise ValueError(f"Missing ESM variants: {missing}")
    rows = np.array([lookup[str(x)] for x in frame.v2_variant_token])
    difference = np.concatenate([
        cache["difference_global"].astype(np.float32),
        cache["difference_local"].astype(np.float32),
    ], axis=1)[rows]
    quality_names = [str(x) for x in cache["sequence_quality"]]
    quality = np.zeros((len(frame), 3), dtype=np.float32)
    quality_map = {"exact": 0, "surrogate": 1, "unresolved": 2}
    for row, idx in enumerate(rows):
        quality[row, quality_map[quality_names[idx]]] = 1
    return normalize(difference, norm="l2"), quality


def projected_esm(esm, quality, train, test, n_components, seed):
    maximum = min(n_components, len(train) - 1, esm.shape[1])
    pca = PCA(n_components=maximum, random_state=seed)
    train_values = pca.fit_transform(esm[train])
    test_values = pca.transform(esm[test])
    scaler = StandardScaler()
    train_values = scaler.fit_transform(train_values).astype(np.float32)
    test_values = scaler.transform(test_values).astype(np.float32)
    return (
        np.concatenate([train_values, quality[train]], axis=1),
        np.concatenate([test_values, quality[test]], axis=1),
        float(pca.explained_variance_ratio_.sum()),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--cdk", required=True)
    parser.add_argument("--esm", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--models", nargs="+", default=["svr", "extratrees", "histgradientboosting"])
    parser.add_argument("--seed", type=int, default=746)
    parser.add_argument("--esm-components", type=int, default=16)
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.data, sep="\t", low_memory=False)
    molecules = mols(frame.standardized_smiles)
    fingerprints = {"ecfp6": compound_features(molecules, "ecfp6").astype(np.float32)}
    cdk = pd.read_csv(args.cdk)
    columns = [x for x in cdk.columns if x.startswith("CDK_STANDARD_")]
    if len(cdk) != len(frame) or len(columns) != 1024:
        raise ValueError("CDK fingerprint table is not aligned to the frozen data")
    fingerprints["cdk_standard"] = cdk[columns].to_numpy(np.float32)
    esm, quality = load_esm(args.esm, frame)
    endpoint = endpoint_features(frame)
    encoded = {
        "no_mutation": endpoint,
        "hash": context_features(frame, "hash"),
        "explicit": context_features(frame, "explicit"),
    }
    fold_rows, predictions = [], []
    for fp_name, fingerprint in fingerprints.items():
        for model_name in args.models:
            for fold in range(5):
                split = frame[f"unseen_compound_fold{fold}"]
                train = np.flatnonzero(split.eq("train")); test = np.flatnonzero(split.eq("test"))
                esm_train, esm_test, explained = projected_esm(
                    esm, quality, train, test, args.esm_components, args.seed + fold)
                for target in TARGETS:
                    # Mutation-aware representations are not supplied to the WT task.
                    reps = ["no_mutation"] if target == "p_activity_wt" else MUTANT_REPRESENTATIONS
                    for representation in reps:
                        if representation in encoded:
                            x_train = np.concatenate([fingerprint[train], encoded[representation][train]], axis=1)
                            x_test = np.concatenate([fingerprint[test], encoded[representation][test]], axis=1)
                            variance = np.nan
                        else:
                            base_train, base_test = endpoint[train], endpoint[test]
                            if representation == "explicit_esm":
                                explicit = encoded["explicit"]
                                base_train, base_test = explicit[train], explicit[test]
                            x_train = np.concatenate([fingerprint[train], base_train, esm_train], axis=1)
                            x_test = np.concatenate([fingerprint[test], base_test, esm_test], axis=1)
                            variance = explained
                        model = make_model(model_name, args.seed)
                        model.fit(x_train, frame[target].to_numpy()[train])
                        prediction = model.predict(x_test)
                        result = scores(frame[target].to_numpy()[test], prediction)
                        fold_rows.append({
                            "fingerprint": fp_name, "protein_representation": representation,
                            "model": model_name, "fold": fold, "target": target,
                            "n_train": len(train), "n_test": len(test),
                            "esm_explained_variance": variance, **result,
                        })
                        predictions.extend({
                            "row_index": int(index), "fingerprint": fp_name,
                            "protein_representation": representation, "model": model_name,
                            "fold": fold, "target": target, "observed": float(frame[target].iloc[index]),
                            "predicted": float(value),
                        } for index, value in zip(test, prediction))
                print(json.dumps(fold_rows[-1]), flush=True)
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(out / "fold_metrics.csv", index=False)
    summary = folds.groupby(["fingerprint", "protein_representation", "model", "target"], as_index=False).agg(
        rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_sd=("mae", "std"),
        pearson_mean=("pearson", "mean"), pearson_sd=("pearson", "std"),
        spearman_mean=("spearman", "mean"), spearman_sd=("spearman", "std"),
    )
    summary.to_csv(out / "aggregate_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(out / "test_predictions.tsv.gz", sep="\t", index=False, compression="gzip")
    (out / "manifest.json").write_text(json.dumps({
        "seed": args.seed, "esm_components": args.esm_components,
        "fingerprints": list(fingerprints), "models": args.models,
        "protein_representations": MUTANT_REPRESENTATIONS,
        "wt_rule": "compound fingerprint + endpoint only; mutant representation excluded",
        "esm_projection": "fold-specific PCA fitted on training rows, followed by training-only standardization",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
