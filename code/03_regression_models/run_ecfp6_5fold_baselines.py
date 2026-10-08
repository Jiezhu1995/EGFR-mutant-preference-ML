#!/usr/bin/env python3
"""Paper-level ECFP6 classical baselines on frozen scaffold folds.

Historical R/CDK fingerprints are deliberately not supported here.  This
script computes one reproducible RDKit Morgan radius-3 fingerprint matrix and
uses the exact split columns already frozen in the input table.
"""

# Resolve shared modules across the six workflow stages.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(p) for p in _Path(__file__).resolve().parents[1].glob("[0-9][0-9]_*") if p.is_dir()]
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import pearsonr, spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ENDPOINTS = {"IC50": 0, "Ki": 1, "Kd": 2}


TARGETS = ["p_activity_wt", "p_activity_mut", "delta_pactivity"]


def mutation_token(text, dim=256):
    """Hash complete v2 tokens, including substitutions, deletions and insertions."""
    vector = np.zeros(dim, dtype=np.float32)
    tokens = [value.strip() for value in str(text).split(";") if value.strip()]
    for token in tokens:
        for feature in (token, token.split(":", 1)[0], "FULL:" + str(text)):
            digest = hashlib.sha256(feature.encode()).digest()
            index = int.from_bytes(digest[:4], "little") % dim
            vector[index] += 1.0 if digest[4] % 2 == 0 else -1.0
    vector[-1] = len(tokens)
    return vector


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)


def ecfp6(smiles, n_bits=2048):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=n_bits)
    matrix = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    invalid = []
    cxsmiles_fallback = []
    for i, value in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(value))
        if mol is None and " |" in str(value):
            mol = Chem.MolFromSmiles(str(value).split(" |", 1)[0])
            if mol is not None:
                cxsmiles_fallback.append(i)
        if mol is None:
            invalid.append(i)
            continue
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), matrix[i])
    if invalid:
        raise ValueError(f"Invalid standardized SMILES at row indices: {invalid[:20]}")
    if cxsmiles_fallback:
        print(json.dumps({"cxsmiles_fallback_rows": cxsmiles_fallback}), flush=True)
    return matrix


def design_matrix(frame):
    fp = ecfp6(frame.standardized_smiles)
    mutation = np.stack([mutation_token(x) for x in frame.mutation_annotations]).astype(np.float32)
    endpoint = np.zeros((len(frame), len(ENDPOINTS)), dtype=np.float32)
    codes = frame.endpoint.map(ENDPOINTS)
    if codes.isna().any():
        raise ValueError(f"Unknown endpoints: {sorted(frame.loc[codes.isna(), 'endpoint'].unique())}")
    endpoint[np.arange(len(frame)), codes.astype(int)] = 1
    return np.concatenate([fp.astype(np.float32), mutation, endpoint], axis=1)


def metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    return {
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "mae": float(np.mean(np.abs(y - pred))),
        "pearson": float(pearsonr(y, pred).statistic) if len(y) > 2 else None,
        "spearman": float(spearmanr(y, pred).statistic) if len(y) > 2 else None,
    }


def make_model(name, seed):
    if name == "mean":
        return DummyRegressor(strategy="mean")
    if name == "ridge":
        return make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0))
    if name == "elasticnet":
        return make_pipeline(
            StandardScaler(with_mean=False),
            ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=20000, random_state=seed),
        )
    if name == "randomforest":
        return RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2, max_features="sqrt",
            n_jobs=-1, random_state=seed,
        )
    if name == "svr":
        return SVR(C=10.0, gamma="scale", epsilon=0.1, kernel="rbf", cache_size=4096)
    if name == "extratrees":
        return ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=2, max_features="sqrt",
            n_jobs=-1, random_state=seed,
        )
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--models", nargs="+",
        choices=["mean", "ridge", "elasticnet", "randomforest", "extratrees", "svr"],
        default=["mean", "ridge", "elasticnet", "randomforest", "extratrees", "svr"],
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--split-mode", default="unseen_compound")
    args = parser.parse_args()

    seed_all(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.data, sep="\t", low_memory=False)
    if "mutation_annotations" not in frame and "v2_variant_token" in frame:
        frame["mutation_annotations"] = frame.v2_variant_token
    if "mutation_task" not in frame:
        frame["mutation_task"] = frame.uniprot_accession.astype(str) + ":" + frame.mutation_annotations.astype(str)
    x = design_matrix(frame)
    predictions, summaries = [], []

    for fold in range(args.n_folds):
        split_col = f"{args.split_mode}_fold{fold}"
        if split_col not in frame:
            raise KeyError(f"Frozen split column missing: {split_col}")
        split = frame[split_col]
        train_idx = np.flatnonzero(split.eq("train"))
        val_idx = np.flatnonzero(split.eq("val"))
        test_idx = np.flatnonzero(split.eq("test"))
        if not len(train_idx) or not len(val_idx) or not len(test_idx):
            raise ValueError(f"Empty train/val/test subset in {split_col}")

        for model_name in args.models:
            pred_frame = frame.iloc[test_idx][[
                "inchikey_connectivity", "standardized_smiles", "uniprot_accession",
                "mutation_task", "mutation_annotations", "endpoint", "scaffold_group",
                *TARGETS,
            ]].copy()
            pred_frame.insert(0, "seed", args.seed)
            pred_frame.insert(0, "fold", fold)
            pred_frame.insert(0, "model", f"{model_name}_ecfp6")
            target_metrics = {}
            model_dir = out / f"{model_name}_ecfp6_fold{fold}_seed{args.seed}"
            model_dir.mkdir(exist_ok=True)
            for target in TARGETS:
                model = make_model(model_name, args.seed)
                model.fit(x[train_idx], frame.iloc[train_idx][target].to_numpy())
                pred = model.predict(x[test_idx])
                pred_frame[f"pred_{target}"] = pred
                target_metrics[target] = metrics(frame.iloc[test_idx][target], pred)
                dump(model, model_dir / f"{target}.joblib", compress=3)
            predictions.append(pred_frame)
            row = {
                "model": f"{model_name}_ecfp6", "fold": fold, "seed": args.seed,
                "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
            }
            for target, vals in target_metrics.items():
                row.update({f"{target}_{key}": value for key, value in vals.items()})
            summaries.append(row)
            (model_dir / "summary.json").write_text(json.dumps(row, indent=2))
            print(json.dumps(row), flush=True)

    pd.concat(predictions, ignore_index=True).to_csv(
        out / "all_test_predictions.tsv.gz", sep="\t", index=False, compression="gzip"
    )
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "fold_metrics.csv", index=False)
    aggregate = summary.groupby("model").agg(["mean", "std"])
    aggregate.to_csv(out / "aggregate_metrics.csv")
    manifest = {
        "fingerprint": "RDKit Morgan/ECFP6",
        "radius": 3,
        "n_bits": 2048,
        "rdkit_version": rdBase.rdkitVersion,
        "split_mode": args.split_mode,
        "seed": args.seed,
        "models": args.models,
        "data": str(Path(args.data).resolve()),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
