#!/usr/bin/env python3
"""Compare compound fingerprints and mutation encodings on frozen EGFR folds."""

# Resolve shared modules across the six workflow stages.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(p) for p in _Path(__file__).resolve().parents[1].glob("[0-9][0-9]_*") if p.is_dir()]
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

TARGETS = ["p_activity_wt", "p_activity_mut", "delta_pactivity"]
ENDPOINTS = {"IC50": 0, "Ki": 1, "Kd": 2}
SUB = re.compile(r"SUB:([A-Z])(\d+)([A-Z])")
AA = "ACDEFGHIKLMNPQRSTVWY"


def mols(smiles):
    result = []
    for value in smiles:
        value = str(value).split(" |", 1)[0]
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            raise ValueError(value)
        result.append(mol)
    return result


def bit_matrix(fps, size):
    matrix = np.zeros((len(fps), size), dtype=np.uint8)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, matrix[i])
    return matrix


def compound_features(molecules, name):
    if name == "ecfp4":
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        return bit_matrix([gen.GetFingerprint(m) for m in molecules], 2048)
    if name == "ecfp6":
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
        return bit_matrix([gen.GetFingerprint(m) for m in molecules], 2048)
    if name == "rdkit":
        gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048)
        return bit_matrix([gen.GetFingerprint(m) for m in molecules], 2048)
    if name == "atompair":
        gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)
        return bit_matrix([gen.GetFingerprint(m) for m in molecules], 2048)
    if name == "maccs":
        return bit_matrix([MACCSkeys.GenMACCSKeys(m) for m in molecules], 167)
    raise KeyError(name)


def signed_hash(text, dim=256):
    vector = np.zeros(dim, dtype=np.float32)
    tokens = [x.strip() for x in str(text).split(";") if x.strip()]
    for token in tokens:
        for feature in (token, token.split(":", 1)[0], "FULL:" + str(text)):
            digest = hashlib.sha256(feature.encode()).digest()
            vector[int.from_bytes(digest[:4], "little") % dim] += 1 if digest[4] % 2 == 0 else -1
    vector[-1] = len(tokens)
    return vector


def explicit_token(text):
    # Mutation types + normalized positions + WT/mutant amino-acid counts.
    vector = np.zeros(3 + 8 + 2 * len(AA), dtype=np.float32)
    tokens = [x.strip() for x in str(text).split(";") if x.strip()]
    positions = []
    for token in tokens:
        kind = token.split(":", 1)[0]
        if kind in ("SUB", "DEL", "INS"):
            vector[("SUB", "DEL", "INS").index(kind)] += 1
        for match in re.finditer(r"\d+", token):
            positions.append(float(match.group()) / 1210.0)
        match = SUB.fullmatch(token)
        if match:
            wt, _, mut = match.groups()
            if wt in AA:
                vector[11 + AA.index(wt)] += 1
            if mut in AA:
                vector[11 + len(AA) + AA.index(mut)] += 1
    summary = [len(tokens), len(positions), np.mean(positions) if positions else 0,
               np.std(positions) if positions else 0, min(positions) if positions else 0,
               max(positions) if positions else 0, sum("EXON" in x for x in tokens),
               sum("INS" in x for x in tokens)]
    vector[3:11] = summary
    return vector


def context_features(frame, mode):
    endpoint = np.zeros((len(frame), len(ENDPOINTS)), dtype=np.float32)
    code = frame.endpoint.map(ENDPOINTS)
    if code.isna().any():
        raise ValueError("Unknown endpoint")
    endpoint[np.arange(len(frame)), code.astype(int)] = 1
    if mode == "none":
        return endpoint
    values = frame.v2_variant_token
    if mode == "hash":
        mutation = np.stack([signed_hash(x) for x in values])
    elif mode == "explicit":
        mutation = np.stack([explicit_token(x) for x in values])
    else:
        raise KeyError(mode)
    return np.concatenate([mutation, endpoint], axis=1)


def metric(y, pred):
    return {
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "mae": float(mean_absolute_error(y, pred)),
        "pearson": float(pearsonr(y, pred).statistic),
        "spearman": float(spearmanr(y, pred).statistic),
    }


def make_model(name, seed):
    if name == "mean":
        return DummyRegressor(strategy="mean")
    if name == "ridge":
        return make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0))
    if name == "elasticnet":
        return make_pipeline(StandardScaler(with_mean=False),
                             ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000,
                                        random_state=seed))
    if name == "knn":
        return KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=-1)
    if name == "svr":
        return SVR(C=10.0, gamma="scale", epsilon=0.1, kernel="rbf")
    if name == "randomforest":
        return RandomForestRegressor(n_estimators=500, min_samples_leaf=2,
                                     max_features="sqrt", n_jobs=-1,
                                     random_state=seed)
    if name == "extratrees":
        return ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2,
                                   max_features="sqrt", n_jobs=-1, random_state=seed)
    if name == "histgradientboosting":
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                             max_leaf_nodes=31, l2_regularization=1.0,
                                             early_stopping=True, random_state=seed)
    raise KeyError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cdk", help="Row-aligned CDK fingerprint CSV(.gz)")
    parser.add_argument("--seed", type=int, default=746)
    parser.add_argument("--models", nargs="+", default=["svr", "extratrees"])
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.data, sep="\t", low_memory=False)
    molecules = mols(frame.standardized_smiles)
    fingerprints = ["ecfp4", "ecfp6", "maccs", "rdkit", "atompair"]
    cdk_matrices = {}
    if args.cdk:
        cdk = pd.read_csv(args.cdk)
        if len(cdk) != len(frame) or not np.array_equal(cdk.row_index.values, np.arange(len(frame))):
            raise ValueError("CDK matrix is not row-aligned to the frozen paired table")
        for name, prefix in (("cdk_standard", "CDK_STANDARD_"), ("cdk_ecfp6", "CDK_ECFP6_")):
            columns = [x for x in cdk.columns if x.startswith(prefix)]
            matrix = cdk[columns].to_numpy(dtype=np.float32)
            if len(columns) != 1024 or np.isnan(matrix).any():
                raise ValueError(f"Invalid {name} matrix: shape={matrix.shape}")
            cdk_matrices[name] = matrix
        fingerprints += list(cdk_matrices)
    models = args.models
    rows, predictions = [], []
    # Factor A: fingerprints, with the explicit mutation representation fixed.
    experiments = [(fp, "explicit") for fp in fingerprints]
    # Factor B: mutation encodings, with ECFP6 fixed.
    experiments += [("ecfp6", mode) for mode in ("none", "hash")]
    for fp_name, mutation_mode in experiments:
        fp = cdk_matrices.get(fp_name)
        if fp is None:
            fp = compound_features(molecules, fp_name).astype(np.float32)
        x = np.concatenate([fp, context_features(frame, mutation_mode)], axis=1)
        for model_name in models:
            for fold in range(5):
                split = frame[f"unseen_compound_fold{fold}"]
                train = np.flatnonzero(split.eq("train")); test = np.flatnonzero(split.eq("test"))
                for target in TARGETS:
                    model = make_model(model_name, args.seed)
                    model.fit(x[train], frame[target].values[train])
                    pred = model.predict(x[test])
                    scores = metric(frame[target].values[test], pred)
                    rows.append({"fingerprint": fp_name, "mutation_encoding": mutation_mode,
                                 "model": model_name, "fold": fold, "target": target,
                                 "n_train": len(train), "n_test": len(test), **scores})
                    predictions.extend({"row_index": int(i), "fingerprint": fp_name,
                                        "mutation_encoding": mutation_mode, "model": model_name,
                                        "fold": fold, "target": target,
                                        "observed": float(frame[target].iloc[i]),
                                        "predicted": float(p)} for i, p in zip(test, pred))
                print(json.dumps(rows[-1]), flush=True)
    folds = pd.DataFrame(rows)
    folds.to_csv(out / "fold_metrics.csv", index=False)
    summary = folds.groupby(["fingerprint", "mutation_encoding", "model", "target"], as_index=False).agg(
        rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"),
        mae_mean=("mae", "mean"), pearson_mean=("pearson", "mean"),
        spearman_mean=("spearman", "mean"))
    summary.to_csv(out / "aggregate_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(out / "test_predictions.tsv.gz", sep="\t", index=False, compression="gzip")
    (out / "manifest.json").write_text(json.dumps({"seed": args.seed, "rdkit": rdBase.rdkitVersion,
        "fingerprints": fingerprints, "mutation_encodings": ["none", "hash", "explicit"],
        "models": models, "data": str(Path(args.data).resolve())}, indent=2))


if __name__ == "__main__":
    main()
