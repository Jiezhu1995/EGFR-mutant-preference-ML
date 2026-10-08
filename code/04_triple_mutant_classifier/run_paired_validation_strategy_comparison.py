#!/usr/bin/env python3
"""Paired random-stratified versus scaffold CV for the archived ECFP6-SVM."""

# Resolve shared modules across the six workflow stages.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(p) for p in _Path(__file__).resolve().parents[1].glob("[0-9][0-9]_*") if p.is_dir()]

from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT.parents[1] if len(ROOT.parents) > 1 else ROOT
DEFAULT_LEGACY = PROJECT / "work" / "egfr_audit" / "EGFRproject" / "ECFP6" / "mutation"
DEFAULT_OUT = ROOT / "results" / "figure3_classifier" / "paired_validation_strategy_metrics.csv"
SEEDS = [104, 410, 853, 746, 732, 360, 156, 556, 450, 727]


def scaffold(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    value = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return value or "ACYCLIC:" + Chem.MolToSmiles(mol, canonical=True)


def metrics(y, score, pred):
    return {
        "roc_auc": roc_auc_score(y, score),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "mcc": matthews_corrcoef(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
    }


def load_data(legacy):
    paths = [legacy / "final_mutation_dataset_trainingset_ECFP6.csv",
             legacy / "final_mutation_dataset_testset_ECFP6.csv"]
    frames = [pd.read_csv(path) for path in paths]
    feature_columns = sorted(
        [c for c in frames[0].columns if re.fullmatch(r"V\d+", str(c))],
        key=lambda x: int(x[1:]))
    if len(feature_columns) != 1024:
        raise ValueError(f"Expected 1,024 archived ECFP6 features; found {len(feature_columns)}")
    df = pd.concat(frames, ignore_index=True)
    if len(df) != 2099 or int(df.label.sum()) != 997:
        raise ValueError("Archived mutant dataset does not match the frozen 2,099-record dataset")
    X = df[feature_columns].to_numpy(dtype=np.uint8)
    y = df.label.to_numpy(dtype=int)
    groups = df.Standard_Smiles.map(scaffold).to_numpy()
    return X, y, groups


def evaluate(X, y, splits, seed, strategy):
    rows = []
    for fold, (train, test) in enumerate(splits, 1):
        model = SVC(C=10, kernel="rbf", gamma="scale", class_weight="balanced",
                    cache_size=1000, random_state=seed)
        model.fit(X[train], y[train])
        score = model.decision_function(X[test])
        pred = (score >= 0).astype(int)
        rows.append({"seed": seed, "strategy": strategy, "fold": fold,
                     "n_test": len(test), **metrics(y[test], score, pred)})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-dir", type=Path, default=DEFAULT_LEGACY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    X, y, groups = load_data(args.legacy_dir)
    rows = []
    for seed in SEEDS:
        random_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        scaffold_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        rows.extend(evaluate(X, y, random_cv.split(X, y), seed,
                             "random_stratified_5fold"))
        rows.extend(evaluate(X, y, scaffold_cv.split(X, y, groups), seed,
                             "scaffold_5fold"))
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    summary = (result.groupby(["strategy", "seed"], as_index=False)
               [["roc_auc", "mcc", "accuracy", "balanced_accuracy",
                 "precision", "recall"]].mean())
    print(summary.groupby("strategy")[["roc_auc", "mcc"]].agg(["mean", "std"]))
    print(f"Saved {len(result)} fold-level rows to {args.output}")


if __name__ == "__main__":
    main()
