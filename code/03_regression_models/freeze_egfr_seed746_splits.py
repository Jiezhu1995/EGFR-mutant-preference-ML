#!/usr/bin/env python3
"""Freeze a reproducible EGFR paired scaffold 5-fold split for any seed."""

# Resolve shared modules across the six workflow stages.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(p) for p in _Path(__file__).resolve().parents[1].glob("[0-9][0-9]_*") if p.is_dir()]
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


LEAD_KEYS = {"HRYNCLKDKCPYBF", "PBCZSGKMGDDXIJ"}


def balanced_mapping(frame, seed, n_folds=5):
    sizes = frame.groupby("scaffold_group").size().reset_index(name="n")
    rng = np.random.default_rng(seed)
    sizes["tie"] = rng.random(len(sizes))
    sizes = sizes.sort_values(["n", "tie"], ascending=[False, True])
    loads = [0] * n_folds
    mapping = {}
    for row in sizes.itertuples(index=False):
        fold = min(range(n_folds), key=lambda value: (loads[value], value))
        mapping[row.scaffold_group] = fold
        loads[fold] += int(row.n)
    return mapping, loads


def overlap(a, b):
    return len(set(a.dropna()) & set(b.dropna()))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=746)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.data, sep="\t", low_memory=False)
    if frame.scaffold_group.isna().any():
        raise ValueError("Missing scaffold_group values")
    mapping, loads = balanced_mapping(frame, args.seed)
    fold_column = f"seed{args.seed}_scaffold_fold"
    frame[fold_column] = frame.scaffold_group.map(mapping).astype(int)
    for fold in range(5):
        val_fold = (fold + 1) % 5
        frame[f"unseen_compound_fold{fold}"] = np.where(
            frame[fold_column].eq(fold),
            "test",
            np.where(frame[fold_column].eq(val_fold), "val", "train"),
        )

    data_path = out / f"egfr_pairs_scaffold5_seed{args.seed}.tsv.gz"
    frame.to_csv(data_path, sep="\t", index=False, compression="gzip")
    map_path = out / f"scaffold_fold_map_seed{args.seed}.csv"
    pd.DataFrame(
        sorted(mapping.items()), columns=["scaffold_group", fold_column]
    ).to_csv(map_path, index=False)

    split_rows, audit_rows = [], []
    for fold in range(5):
        col = f"unseen_compound_fold{fold}"
        parts = {name: frame[frame[col].eq(name)] for name in ("train", "val", "test")}
        split_rows.append({"fold": fold, **{name: len(part) for name, part in parts.items()}})
        audit_rows.append({
            "fold": fold,
            "scaffold_train_val_overlap": overlap(parts["train"].scaffold_group, parts["val"].scaffold_group),
            "scaffold_train_test_overlap": overlap(parts["train"].scaffold_group, parts["test"].scaffold_group),
            "scaffold_val_test_overlap": overlap(parts["val"].scaffold_group, parts["test"].scaffold_group),
            "compound_train_val_overlap": overlap(parts["train"].inchikey_connectivity, parts["val"].inchikey_connectivity),
            "compound_train_test_overlap": overlap(parts["train"].inchikey_connectivity, parts["test"].inchikey_connectivity),
            "compound_val_test_overlap": overlap(parts["val"].inchikey_connectivity, parts["test"].inchikey_connectivity),
        })
    split_path = out / f"split_sizes_seed{args.seed}.csv"
    audit_path = out / f"leakage_audit_seed{args.seed}.csv"
    pd.DataFrame(split_rows).to_csv(split_path, index=False)
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(audit_path, index=False)
    if audit.drop(columns="fold").to_numpy().sum() != 0:
        raise RuntimeError("Leakage audit failed")

    observed_leads = sorted(set(frame.inchikey_connectivity) & LEAD_KEYS)
    manifest = {
        "schema_version": "egfr_paired_repeated_scaffold_v1",
        "source_data": str(Path(args.data).resolve()),
        "seed": args.seed,
        "n_folds": 5,
        "assignment_algorithm": "size-descending greedy scaffold balancing; seeded random tie order",
        "validation_fold_rule": "(test_fold + 1) mod 5",
        "rows": len(frame),
        "unique_compounds": int(frame.inchikey_connectivity.nunique()),
        "unique_scaffold_groups": int(frame.scaffold_group.nunique()),
        "fold_loads": loads,
        "external_lead_connectivity_keys_found": observed_leads,
        "external_leads_absent": len(observed_leads) == 0,
        "leakage_checks_passed": True,
    }
    manifest_path = out / f"frozen_manifest_seed{args.seed}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    checksum_files = [data_path, map_path, split_path, audit_path, manifest_path]
    checksum_path = out / f"SHA256SUMS_seed{args.seed}.txt"
    checksum_path.write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in checksum_files)
    )
    print(json.dumps(manifest, indent=2))
    print(pd.DataFrame(split_rows).to_string(index=False))
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
