#!/usr/bin/env python3
"""Re-evaluate the manuscript model families on archived R fingerprints.

Model configurations are read directly from fixed_parameters.json and evaluated
on the supplied held-out split and repeated scaffold folds. Fingerprints are
loaded from the supplied V* columns and are never regenerated.
"""

from __future__ import annotations

# Resolve shared modules across the six workflow stages.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(p) for p in _Path(__file__).resolve().parents[1].glob("[0-9][0-9]_*") if p.is_dir()]

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from archived_fingerprint_benchmark import audit_endpoint


MODELS = ["SVM", "RF", "XGBoost", "KNN", "LR", "ANN"]
PARAMETER_FILE = Path(__file__).with_name("fixed_parameters.json")

FP_FOLDERS = {
    "CDK_Standard": "Standard",
    "ECFP6": "ECFP6",
    "CDK_Extended": "Extended",
    "CDK_Graph": "Graph",
    "MACCS": "MACCS",
    "PubChem": "Pubchem",
    "Substructure": "Substructure",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--legacy-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument(
        "--all-fingerprints", action="store_true",
        help="Evaluate every endpoint/fingerprint/model combination in the fixed configuration file.")
    return p.parse_args()


def fixed_params(fingerprint: str, endpoint: str, model: str) -> dict:
    configurations = json.loads(PARAMETER_FILE.read_text())["configurations"]
    matches = [r["parameters"] for r in configurations if
               (r["fingerprint"], r["endpoint"], r["model"]) == (fingerprint, endpoint, model)]
    if len(matches) != 1:
        raise ValueError(f"Expected one fixed configuration: {endpoint}/{fingerprint}/{model}")
    return matches[0]


def make_model(name: str, params: dict, seed: int):
    if name == "SVM":
        return SVC(**params, class_weight="balanced", cache_size=1000,
                   random_state=seed)
    if name == "RF":
        return RandomForestClassifier(**params, class_weight="balanced",
                                      random_state=seed, n_jobs=1)
    if name == "XGBoost":
        return XGBClassifier(**params, n_estimators=100, objective="binary:logistic",
                             eval_metric="logloss", random_state=seed, n_jobs=1)
    if name == "KNN":
        # Use the same brute-force backend as the saved benchmark run.
        q = dict(params); q["algorithm"] = "brute"
        return KNeighborsClassifier(**q, n_jobs=1)
    if name == "LR":
        return LogisticRegression(**params, class_weight="balanced",
                                  max_iter=5000, random_state=seed)
    if name == "ANN":
        return MLPClassifier(**params, random_state=seed)
    raise KeyError(name)


def score_model(model, X):
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)


def metrics(y, score, pred):
    return {
        "roc_auc": roc_auc_score(y, score),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "mcc": matthews_corrcoef(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
    }


def fit_evaluate(name, params, X, y, train, test, seed):
    model = make_model(name, params, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X[train], y[train])
    score = score_model(model, X[test])
    pred = model.predict(X[test])
    return model, score, pred, metrics(y[test], score, pred)


def run_task(root, output, endpoint, fingerprint, model_name, meta, X,
             scaffolds, repeats, seed):
    cache = output / "fixed_configuration_checkpoints" / f"{endpoint}_{fingerprint}_{model_name}.joblib"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return joblib.load(cache)
    y = meta.label.to_numpy(dtype=int)
    historical_train = np.flatnonzero(meta.historical_split.eq("training"))
    historical_test = np.flatnonzero(meta.historical_split.eq("test"))
    params = fixed_params(fingerprint, endpoint, model_name)
    parameter_source = "fixed_parameters.json"
    rows, predictions = [], []

    fitted, score, pred, met = fit_evaluate(
        model_name, params, X, y, historical_train, historical_test, seed)
    rows.append({"endpoint": endpoint, "fingerprint": fingerprint,
                 "model": model_name, "strategy": "historical_fixed",
                 "repeat": np.nan, "fold": 0, "n_test": len(historical_test),
                 "parameter_source": parameter_source,
                 "fixed_params": json.dumps(params), **met})
    predictions.append(pd.DataFrame({
        "endpoint": endpoint, "fingerprint": fingerprint, "model": model_name,
        "strategy": "historical_fixed", "repeat": np.nan, "fold": 0,
        "row_id": historical_test, "label": y[historical_test],
        "score": score, "prediction": pred,
    }))
    model_dir = output / "models"; model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted, model_dir / f"{endpoint}_{fingerprint}_{model_name}.joblib")

    random_cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    fold_sets = [("random_stratified_5fold", np.nan, f, tr, te)
                 for f, (tr, te) in enumerate(random_cv.split(X, y))]
    for rep in range(repeats):
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed + rep)
        fold_sets.extend(("scaffold_5fold", rep, f, tr, te)
                         for f, (tr, te) in enumerate(
                             cv.split(X, y, groups=scaffolds.to_numpy())))
    for strategy, rep, fold, tr, te in fold_sets:
        _, score, pred, met = fit_evaluate(
            model_name, params, X, y, tr, te, seed + (0 if np.isnan(rep) else int(rep)))
        rows.append({"endpoint": endpoint, "fingerprint": fingerprint,
                     "model": model_name, "strategy": strategy,
                     "repeat": rep, "fold": fold, "n_test": len(te),
                     "parameter_source": parameter_source,
                     "fixed_params": json.dumps(params), **met})
        predictions.append(pd.DataFrame({
            "endpoint": endpoint, "fingerprint": fingerprint, "model": model_name,
            "strategy": strategy, "repeat": rep, "fold": fold,
            "row_id": te, "label": y[te], "score": score, "prediction": pred,
        }))
    result = rows, pd.concat(predictions, ignore_index=True)
    joblib.dump(result, cache)
    return result


def summarize(table, seed):
    rng = np.random.default_rng(seed); rows = []
    keys = ["endpoint", "fingerprint", "model", "strategy"]
    for key, group in table.groupby(keys, dropna=False):
        row = dict(zip(keys, key)); row["n_evaluations"] = len(group)
        for metric in ["roc_auc", "mcc", "balanced_accuracy", "precision", "recall"]:
            v = group[metric].to_numpy(float)
            row[f"{metric}_mean"] = v.mean()
            row[f"{metric}_sd"] = v.std(ddof=1) if len(v) > 1 else np.nan
            boot = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(10000)])
            row[f"{metric}_ci95_low"] = np.quantile(boot, .025)
            row[f"{metric}_ci95_high"] = np.quantile(boot, .975)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    # Default comparison subset; --all-fingerprints runs the complete benchmark.
    selections = [("mutant", "ECFP6")]
    if args.all_fingerprints:
        selections = [(endpoint, fingerprint)
                      for endpoint in ("mutant",)
                      for fingerprint in FP_FOLDERS]
    models = MODELS
    tasks, skipped = [], []
    for endpoint, fingerprint in selections:
        loaded, scaffolds = audit_endpoint(args.legacy_root, endpoint)
        meta, X, _, _ = loaded[fingerprint]
        for model in models:
            try:
                fixed_params(fingerprint, endpoint, model)
            except ValueError as exc:
                skipped.append({"endpoint": endpoint, "fingerprint": fingerprint,
                                "model": model, "reason": str(exc)})
                continue
            tasks.append((endpoint, fingerprint, model, meta, X, scaffolds))
    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(run_task)(args.legacy_root, args.output_dir, ep, fp, model,
                          meta, X, scaffolds, args.repeats, args.seed)
        for ep, fp, model, meta, X, scaffolds in tasks)
    fold_table = pd.DataFrame([r for result in results for r in result[0]])
    pred_table = pd.concat([result[1] for result in results], ignore_index=True)
    tables = args.output_dir / "tables"; tables.mkdir(exist_ok=True)
    preds = args.output_dir / "predictions"; preds.mkdir(exist_ok=True)
    fold_table.to_csv(tables / "all_model_fold_metrics.csv", index=False)
    summarize(fold_table, args.seed).to_csv(tables / "model_performance_summary.csv", index=False)
    pred_table.to_csv(preds / "all_model_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(skipped, columns=["endpoint", "fingerprint", "model", "reason"]).to_csv(
        tables / "skipped_missing_configurations.csv", index=False)
    print("COMPLETE", args.output_dir)


if __name__ == "__main__":
    main()
