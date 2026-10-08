#!/usr/bin/env python3
"""Predict a small custom SMILES set with the frozen seed-104 ensemble."""

# Resolve shared modules across the six workflow stages.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(p) for p in _Path(__file__).resolve().parents[1].glob("[0-9][0-9]_*") if p.is_dir()]

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

from variant_context import variant_esm
from screen_taosu_seed104_ensemble import contexts, predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--ensemble", type=Path, required=True)
    parser.add_argument("--esm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bundle = joblib.load(args.ensemble)
    if bundle.get("seed") != 104 or len(bundle["fold_models"]) != 5:
        raise ValueError("Expected frozen seed-104 five-fold ensemble")
    fold_contexts, _ = contexts(bundle, args.esm)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)

    rows, arrays = [], []
    with args.input.open(newline="") as handle:
        for row in csv.DictReader(handle):
            input_smiles = row.get("smiles") or row.get("SMILES") or ""
            mol = Chem.MolFromSmiles(input_smiles)
            if mol is None:
                row["canonical_smiles"] = ""
                row["status"] = "invalid_smiles"
                rows.append(row)
                continue
            row["canonical_smiles"] = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            row["status"] = "predicted"
            array = np.zeros(2048, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), array)
            arrays.append(array)
            rows.append(row)

    valid_indices = [i for i, row in enumerate(rows) if row["status"] == "predicted"]
    if arrays:
        raw = predict(np.stack(arrays), fold_contexts, workers=5)
        mut = np.stack([raw[(fold, "mut")] for fold in range(5)], axis=1)
        wt = np.stack([raw[(fold, "wt")] for fold in range(5)], axis=1)
        delta = np.stack([raw[(fold, "delta")] for fold in range(5)], axis=1)
        derived = mut - wt
        for j, i in enumerate(valid_indices):
            mut_mean, wt_mean = mut[j].mean(), wt[j].mean()
            direct_mean, derived_mean = delta[j].mean(), derived[j].mean()
            mut_ic50, wt_ic50 = 10 ** (9-mut_mean), 10 ** (9-wt_mean)
            direct_sel, derived_sel = 10 ** direct_mean, 10 ** derived_mean
            positive = mut_ic50 < 50 and wt_ic50 > 100 and direct_sel > 5 and derived_sel > 5
            rows[i].update({
                "p_activity_mut_mean": f"{mut_mean:.6f}",
                "p_activity_mut_sd": f"{mut[j].std(ddof=1):.6f}",
                "mut_ic50_nM": f"{mut_ic50:.6f}",
                "p_activity_wt_mean": f"{wt_mean:.6f}",
                "p_activity_wt_sd": f"{wt[j].std(ddof=1):.6f}",
                "wt_ic50_nM": f"{wt_ic50:.6f}",
                "delta_pactivity_direct_mean": f"{direct_mean:.6f}",
                "delta_pactivity_direct_sd": f"{delta[j].std(ddof=1):.6f}",
                "direct_selectivity_fold": f"{direct_sel:.6f}",
                "derived_delta_mean": f"{derived_mean:.6f}",
                "derived_selectivity_fold": f"{derived_sel:.6f}",
                "mut_ic50_lt50_votes": int((10 ** (9-mut[j]) < 50).sum()),
                "wt_ic50_gt100_votes": int((10 ** (9-wt[j]) > 100).sum()),
                "direct_selectivity_gt5_votes": int((10 ** delta[j] > 5).sum()),
                "derived_selectivity_gt5_votes": int((10 ** derived[j] > 5).sum()),
                "all_four_filters_votes": int(((10 ** (9-mut[j]) < 50) &
                                                (10 ** (9-wt[j]) > 100) &
                                                (10 ** delta[j] > 5) &
                                                (10 ** derived[j] > 5)).sum()),
                "potential_positive": int(positive),
            })

    fields = list(rows[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} rows; RDKit {rdBase.rdkitVersion}; {args.output}")


if __name__ == "__main__":
    main()
