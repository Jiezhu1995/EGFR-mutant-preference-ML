#!/usr/bin/env python3
"""Five-fold multitask MLP benchmark for paired EGFR WT/mutant regression."""

# Resolve shared modules across the six workflow stages.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(p) for p in _Path(__file__).resolve().parents[1].glob("[0-9][0-9]_*") if p.is_dir()]

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ENDPOINTS = {"IC50": 0, "Ki": 1, "Kd": 2}
AA = "ACDEFGHIKLMNPQRSTVWY"
SUB = re.compile(r"SUB:([A-Z])(\d+)([A-Z])")
TARGETS = ["p_activity_wt", "p_activity_mut", "delta_pactivity"]


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def explicit_token(value):
    vector = np.zeros(51, dtype=np.float32)
    tokens = str(value).split(";")
    positions = []
    for token in tokens:
        kind = token.split(":", 1)[0]
        if kind in ("SUB", "DEL", "INS"):
            vector[("SUB", "DEL", "INS").index(kind)] += 1
        positions.extend(float(x) / 1210.0 for x in re.findall(r"\d+", token))
        match = SUB.fullmatch(token)
        if match:
            wt, _, mutant = match.groups()
            if wt in AA: vector[11 + AA.index(wt)] += 1
            if mutant in AA: vector[31 + AA.index(mutant)] += 1
    summary = [len(tokens), len(positions), np.mean(positions) if positions else 0,
               np.std(positions) if positions else 0, min(positions) if positions else 0,
               max(positions) if positions else 0, sum("EXON" in x for x in tokens),
               sum("INS" in x for x in tokens)]
    vector[3:11] = summary
    return vector


def load_fingerprints(frame, cdk_path, kind):
    if kind == "cdk_standard":
        cdk = pd.read_csv(cdk_path)
        columns = [x for x in cdk if x.startswith("CDK_STANDARD_")]
        if len(cdk) != len(frame) or len(columns) != 1024:
            raise ValueError("Unaligned CDK fingerprint matrix")
        return cdk[columns].to_numpy(np.float32)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
    rows = []
    for smiles in frame.standardized_smiles:
        mol = Chem.MolFromSmiles(smiles)
        fp = generator.GetFingerprint(mol)
        array = np.zeros(2048, dtype=np.float32); DataStructs.ConvertToNumpyArray(fp, array); rows.append(array)
    return np.stack(rows)


def load_esm(frame, path):
    cache = np.load(path, allow_pickle=True)
    lookup = {str(x): i for i, x in enumerate(cache["variant_tokens"])}
    indices = np.array([lookup[str(x)] for x in frame.v2_variant_token])
    difference = np.concatenate([cache["difference_global"], cache["difference_local"]], axis=1).astype(np.float32)[indices]
    norm = np.linalg.norm(difference, axis=1, keepdims=True)
    difference = difference / np.maximum(norm, 1e-8)
    quality_names = np.asarray(cache["sequence_quality"], dtype=str)[indices]
    quality = np.stack([quality_names == "exact", quality_names == "surrogate", quality_names == "unresolved"], axis=1).astype(np.float32)
    return np.concatenate([difference, quality], axis=1)


class Model(nn.Module):
    def __init__(self, fp_dim, protein_dim):
        super().__init__()
        self.compound = nn.Sequential(nn.Linear(fp_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.20),
                                      nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.15))
        self.protein = nn.Sequential(nn.Linear(protein_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.10))
        self.wt = nn.Sequential(nn.Linear(259, 128), nn.GELU(), nn.Dropout(0.10), nn.Linear(128, 1))
        self.mut = nn.Sequential(nn.Linear(387, 192), nn.GELU(), nn.Dropout(0.10), nn.Linear(192, 1))
        self.delta = nn.Sequential(nn.Linear(387, 192), nn.GELU(), nn.Dropout(0.10), nn.Linear(192, 1))

    def forward(self, fp, protein, endpoint):
        compound = self.compound(fp); protein = self.protein(protein)
        wt = self.wt(torch.cat([compound, endpoint], 1)).squeeze(1)
        joint = torch.cat([compound, protein, endpoint], 1)
        return torch.stack([wt, self.mut(joint).squeeze(1), self.delta(joint).squeeze(1)], 1)


def metrics(y, prediction):
    return {"rmse": float(mean_squared_error(y, prediction) ** 0.5),
            "mae": float(mean_absolute_error(y, prediction)),
            "pearson": float(pearsonr(y, prediction).statistic),
            "spearman": float(spearmanr(y, prediction).statistic)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True); p.add_argument("--cdk", required=True); p.add_argument("--esm", required=True)
    p.add_argument("--output", required=True); p.add_argument("--fingerprint", choices=["ecfp6", "cdk_standard"], required=True)
    p.add_argument("--protein", choices=["explicit", "explicit_esm"], required=True)
    p.add_argument("--seed", type=int, default=746); p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--patience", type=int, default=30); p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--consistency-weight", type=float, default=0.0)
    args = p.parse_args(); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = pd.read_csv(args.data, sep="\t", low_memory=False)
    fp = load_fingerprints(frame, args.cdk, args.fingerprint)
    explicit = np.stack([explicit_token(x) for x in frame.v2_variant_token])
    protein = explicit if args.protein == "explicit" else np.concatenate([explicit, load_esm(frame, args.esm)], 1)
    endpoint = np.zeros((len(frame), 3), np.float32); codes = frame.endpoint.map(ENDPOINTS).to_numpy(int)
    endpoint[np.arange(len(frame)), codes] = 1
    targets = frame[TARGETS].to_numpy(np.float32)
    fold_rows, prediction_rows, histories = [], [], []
    for fold in range(5):
        seed_all(args.seed + fold)
        split = frame[f"unseen_compound_fold{fold}"]
        indices = {name: np.flatnonzero(split.eq(name)) for name in ("train", "val", "test")}
        mean = targets[indices["train"]].mean(0); std = np.maximum(targets[indices["train"]].std(0), 1e-6)
        scaled = (targets - mean) / std
        tensors = [torch.from_numpy(x.astype(np.float32)) for x in (fp, protein, endpoint, scaled)]
        loaders = {name: DataLoader(TensorDataset(*[x[idx] for x in tensors]), batch_size=args.batch_size,
                                    shuffle=(name == "train"), num_workers=0) for name, idx in indices.items()}
        model = Model(fp.shape[1], protein.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
        best, best_state, waiting = float("inf"), None, 0
        for epoch in range(1, args.epochs + 1):
            model.train(); losses = []
            for batch in loaders["train"]:
                batch = [x.to(device) for x in batch]; prediction = model(*batch[:3])
                loss = ((prediction - batch[3]) ** 2).mean()
                if args.consistency_weight:
                    derived = prediction[:, 1] * std[1] + mean[1] - (prediction[:, 0] * std[0] + mean[0])
                    direct = prediction[:, 2] * std[2] + mean[2]
                    loss = loss + args.consistency_weight * ((direct - derived) ** 2).mean()
                optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(loss.item())
            model.eval(); val_pred, val_y = [], []
            with torch.inference_mode():
                for batch in loaders["val"]:
                    batch = [x.to(device) for x in batch]; val_pred.append(model(*batch[:3]).cpu().numpy()); val_y.append(batch[3].cpu().numpy())
            val_pred = np.concatenate(val_pred) * std + mean; val_y = np.concatenate(val_y) * std + mean
            val_score = float(np.mean(np.sqrt(np.mean((val_pred - val_y) ** 2, axis=0))))
            histories.append({"fold": fold, "epoch": epoch, "train_loss": np.mean(losses), "val_mean_rmse": val_score})
            if val_score < best - 1e-5:
                best, waiting = val_score, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                waiting += 1
                if waiting >= args.patience: break
        model.load_state_dict(best_state); model.eval(); test_pred = []
        with torch.inference_mode():
            for batch in loaders["test"]:
                batch = [x.to(device) for x in batch]; test_pred.append(model(*batch[:3]).cpu().numpy())
        test_pred = np.concatenate(test_pred) * std + mean; test_y = targets[indices["test"]]
        for column, target in enumerate(TARGETS):
            result = metrics(test_y[:, column], test_pred[:, column])
            fold_rows.append({"fingerprint": args.fingerprint, "protein_representation": args.protein,
                              "model": "multitask_mlp", "fold": fold, "target": target,
                              "best_epoch": int(np.argmin([x["val_mean_rmse"] for x in histories if x["fold"] == fold]) + 1), **result})
            prediction_rows.extend({"row_index": int(i), "fold": fold, "target": target,
                                    "observed": float(y), "predicted": float(pred)}
                                   for i, y, pred in zip(indices["test"], test_y[:, column], test_pred[:, column]))
        torch.save(best_state, out / f"fold{fold}_best.pt")
        print(json.dumps(fold_rows[-1]), flush=True)
    folds = pd.DataFrame(fold_rows); folds.to_csv(out / "fold_metrics.csv", index=False)
    folds.groupby(["fingerprint", "protein_representation", "model", "target"], as_index=False).agg(
        rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"), mae_mean=("mae", "mean"),
        pearson_mean=("pearson", "mean"), spearman_mean=("spearman", "mean")).to_csv(out / "aggregate_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(out / "test_predictions.tsv.gz", sep="\t", index=False, compression="gzip")
    pd.DataFrame(histories).to_csv(out / "history.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({"seed": args.seed, "device": str(device), "torch": torch.__version__,
        "rdkit": rdBase.rdkitVersion, "fingerprint": args.fingerprint, "protein": args.protein,
        "consistency_weight": args.consistency_weight, "wt_input": "compound+endpoint only",
        "mutant_and_delta_input": "compound+endpoint+protein representation"}, indent=2) + "\n")


if __name__ == "__main__": main()
