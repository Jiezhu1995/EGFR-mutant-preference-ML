#!/usr/bin/env python3
"""Five-fold GIN multitask benchmark using frozen EGFR scaffold splits."""

# Resolve shared modules across the six workflow stages.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(p) for p in _Path(__file__).resolve().parents[1].glob("[0-9][0-9]_*") if p.is_dir()]

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, rdBase
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool

from train_fingerprint_multitask_mlp import ENDPOINTS, TARGETS, explicit_token, load_esm, metrics, seed_all


ATOMS = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
HYBRID = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3,
          Chem.HybridizationType.SP3D, Chem.HybridizationType.SP3D2]
BONDS = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]


def one_hot(value, choices):
    return [float(value == x) for x in choices] + [float(value not in choices)]


def graph(smiles, protein, endpoint, target):
    mol = Chem.MolFromSmiles(smiles)
    nodes = []
    for atom in mol.GetAtoms():
        nodes.append(one_hot(atom.GetAtomicNum(), ATOMS) + one_hot(atom.GetDegree(), list(range(6))) +
                     one_hot(atom.GetFormalCharge(), [-2, -1, 0, 1, 2]) + one_hot(atom.GetHybridization(), HYBRID) +
                     [float(atom.GetIsAromatic()), float(atom.IsInRing()), atom.GetTotalNumHs() / 4.0])
    edges, attributes = [], []
    for bond in mol.GetBonds():
        attr = one_hot(bond.GetBondType(), BONDS) + [float(bond.GetIsConjugated()), float(bond.IsInRing())]
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend([[a, b], [b, a]]); attributes.extend([attr, attr])
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(attributes, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long); edge_attr = torch.empty((0, len(BONDS) + 3), dtype=torch.float32)
    return Data(x=torch.tensor(nodes, dtype=torch.float32), edge_index=edge_index, edge_attr=edge_attr,
                protein=torch.tensor(protein[None], dtype=torch.float32),
                endpoint=torch.tensor(endpoint[None], dtype=torch.float32),
                y=torch.tensor(target[None], dtype=torch.float32))


class GIN(nn.Module):
    def __init__(self, node_dim, edge_dim, protein_dim, hidden=192):
        super().__init__()
        self.node = nn.Linear(node_dim, hidden)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(4):
            mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
            self.layers.append(GINEConv(mlp, edge_dim=edge_dim, train_eps=True)); self.norms.append(nn.LayerNorm(hidden))
        self.compound = nn.Sequential(nn.Linear(hidden, 256), nn.GELU(), nn.Dropout(0.15))
        self.protein = nn.Sequential(nn.Linear(protein_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.10))
        self.wt = nn.Sequential(nn.Linear(259, 128), nn.GELU(), nn.Dropout(0.10), nn.Linear(128, 1))
        self.mut = nn.Sequential(nn.Linear(387, 192), nn.GELU(), nn.Dropout(0.10), nn.Linear(192, 1))
        self.delta = nn.Sequential(nn.Linear(387, 192), nn.GELU(), nn.Dropout(0.10), nn.Linear(192, 1))

    def forward(self, batch):
        value = self.node(batch.x)
        for layer, norm in zip(self.layers, self.norms):
            value = norm(value + layer(value, batch.edge_index, batch.edge_attr))
        compound = self.compound(global_mean_pool(value, batch.batch))
        protein = self.protein(batch.protein)
        wt = self.wt(torch.cat([compound, batch.endpoint], 1)).squeeze(1)
        joint = torch.cat([compound, protein, batch.endpoint], 1)
        return torch.stack([wt, self.mut(joint).squeeze(1), self.delta(joint).squeeze(1)], 1)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--data", required=True); p.add_argument("--esm", required=True)
    p.add_argument("--output", required=True); p.add_argument("--protein", choices=["explicit", "explicit_esm"], required=True)
    p.add_argument("--seed", type=int, default=746); p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--patience", type=int, default=30); p.add_argument("--batch-size", type=int, default=96)
    args = p.parse_args(); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = pd.read_csv(args.data, sep="\t", low_memory=False)
    explicit = np.stack([explicit_token(x) for x in frame.v2_variant_token])
    protein = explicit if args.protein == "explicit" else np.concatenate([explicit, load_esm(frame, args.esm)], 1)
    endpoint = np.zeros((len(frame), 3), np.float32); endpoint[np.arange(len(frame)), frame.endpoint.map(ENDPOINTS).to_numpy(int)] = 1
    targets = frame[TARGETS].to_numpy(np.float32)
    raw_graphs = [graph(s, pr, ep, y) for s, pr, ep, y in zip(frame.standardized_smiles, protein, endpoint, targets)]
    fold_rows, prediction_rows, histories = [], [], []
    for fold in range(5):
        seed_all(args.seed + fold); split = frame[f"unseen_compound_fold{fold}"]
        indices = {name: np.flatnonzero(split.eq(name)) for name in ("train", "val", "test")}
        mean = targets[indices["train"]].mean(0); std = np.maximum(targets[indices["train"]].std(0), 1e-6)
        fold_graphs = []
        for i, item in enumerate(raw_graphs):
            copied = item.clone(); copied.y = torch.tensor(((targets[i] - mean) / std)[None], dtype=torch.float32); fold_graphs.append(copied)
        loaders = {name: DataLoader([fold_graphs[i] for i in idx], batch_size=args.batch_size,
                                    shuffle=(name == "train"), num_workers=0) for name, idx in indices.items()}
        first = fold_graphs[0]
        model = GIN(first.x.shape[1], first.edge_attr.shape[1], protein.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=2e-4)
        best, best_state, waiting, fold_history = float("inf"), None, 0, []
        for epoch in range(1, args.epochs + 1):
            model.train(); losses = []
            for batch in loaders["train"]:
                batch = batch.to(device); prediction = model(batch); loss = ((prediction - batch.y) ** 2).mean()
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step(); losses.append(loss.item())
            model.eval(); vp, vy = [], []
            with torch.inference_mode():
                for batch in loaders["val"]:
                    batch = batch.to(device); vp.append(model(batch).cpu().numpy()); vy.append(batch.y.cpu().numpy())
            vp = np.concatenate(vp) * std + mean; vy = np.concatenate(vy) * std + mean
            score = float(np.mean(np.sqrt(np.mean((vp - vy) ** 2, axis=0))))
            fold_history.append({"fold": fold, "epoch": epoch, "train_loss": np.mean(losses), "val_mean_rmse": score})
            if score < best - 1e-5:
                best, waiting = score, 0; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                waiting += 1
                if waiting >= args.patience: break
        histories.extend(fold_history); model.load_state_dict(best_state); model.eval(); tp = []
        with torch.inference_mode():
            for batch in loaders["test"]: tp.append(model(batch.to(device)).cpu().numpy())
        tp = np.concatenate(tp) * std + mean; ty = targets[indices["test"]]
        for column, target in enumerate(TARGETS):
            fold_rows.append({"fingerprint": "learned_2d_graph", "protein_representation": args.protein,
                              "model": "GIN_multitask", "fold": fold, "target": target,
                              "best_epoch": int(np.argmin([x["val_mean_rmse"] for x in fold_history]) + 1),
                              **metrics(ty[:, column], tp[:, column])})
            prediction_rows.extend({"row_index": int(i), "fold": fold, "target": target,
                                    "observed": float(y), "predicted": float(pred)}
                                   for i, y, pred in zip(indices["test"], ty[:, column], tp[:, column]))
        torch.save(best_state, out / f"fold{fold}_best.pt"); print(json.dumps(fold_rows[-1]), flush=True)
    folds = pd.DataFrame(fold_rows); folds.to_csv(out / "fold_metrics.csv", index=False)
    folds.groupby(["fingerprint", "protein_representation", "model", "target"], as_index=False).agg(
        rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"), mae_mean=("mae", "mean"),
        pearson_mean=("pearson", "mean"), spearman_mean=("spearman", "mean")).to_csv(out / "aggregate_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(out / "test_predictions.tsv.gz", sep="\t", index=False, compression="gzip")
    pd.DataFrame(histories).to_csv(out / "history.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({"seed": args.seed, "device": str(device), "torch": torch.__version__,
        "rdkit": rdBase.rdkitVersion, "model": "4-layer GINE", "protein": args.protein,
        "wt_input": "2D graph+endpoint only"}, indent=2) + "\n")


if __name__ == "__main__": main()
