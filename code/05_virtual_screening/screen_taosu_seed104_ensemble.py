#!/usr/bin/env python3
"""Checkpointed TaoSu screening with the seed-104 three-SVR five-fold ensemble."""

# Resolve shared modules across the six workflow stages.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(p) for p in _Path(__file__).resolve().parents[1].glob("[0-9][0-9]_*") if p.is_dir()]

import argparse
import csv
import gzip
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

from variant_context import variant_esm


MUT_IC50_MAX_NM = 50.0
WT_IC50_MIN_NM = 100.0
SELECTIVITY_MIN_FOLD = 5.0


def fingerprint_batch(records, generator):
    valid, invalid, arrays = [], [], []
    for record in records:
        mol = Chem.MolFromSmiles(record[2])
        if mol is None:
            invalid.append(record); continue
        array = np.zeros(2048, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), array)
        arrays.append(array); valid.append(record)
    return valid, invalid, np.stack(arrays) if arrays else None


def contexts(bundle, esm_path):
    raw, quality, sequence_quality = variant_esm(esm_path)
    endpoint = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    values = []
    for fold in bundle["fold_models"]:
        projected = fold["esm_pca"].transform(raw[None, :])
        projected = fold["esm_scaler"].transform(projected).astype(np.float32)[0]
        mutant_context = np.concatenate([endpoint, projected, quality]).astype(np.float32)
        values.append((fold, mutant_context, endpoint))
    return values, sequence_quality


def predict(fp, fold_contexts, workers):
    tasks = []
    for fold, mutant_context, wt_context in fold_contexts:
        x_mut = np.concatenate([fp, np.repeat(mutant_context[None, :], len(fp), axis=0)], axis=1)
        x_wt = np.concatenate([fp, np.repeat(wt_context[None, :], len(fp), axis=0)], axis=1)
        tasks.extend([
            (fold["fold"], "mut", fold["mutant_model"], x_mut),
            (fold["fold"], "delta", fold["delta_model"], x_mut),
            (fold["fold"], "wt", fold["wt_model"], x_wt),
        ])
    output = {}
    def one(task):
        fold, target, model, matrix = task
        return fold, target, model.predict(matrix)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, task) for task in tasks]
        for future in as_completed(futures):
            fold, target, value = future.result(); output[(fold, target)] = value
    return output


def write_checkpoint(path, state):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--database", type=Path, required=True)
    ap.add_argument("--ensemble", type=Path, required=True)
    ap.add_argument("--esm", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--positives", type=Path, required=True)
    ap.add_argument("--invalid", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--max-molecules", type=int)
    args = ap.parse_args()
    for path in (args.output, args.positives, args.invalid, args.checkpoint, args.audit):
        path.parent.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(args.ensemble)
    if bundle.get("seed") != 104 or len(bundle["fold_models"]) != 5:
        raise ValueError("Expected the frozen seed-104 five-fold ensemble")
    fold_contexts, sequence_quality = contexts(bundle, args.esm)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
    fields = [
        "hit_id", "smiles", "source_file", "library_filename",
        "p_activity_mut_mean", "p_activity_mut_sd", "mut_ic50_nM",
        "p_activity_wt_mean", "p_activity_wt_sd", "wt_ic50_nM",
        "delta_pactivity_direct_mean", "delta_pactivity_direct_sd",
        "direct_selectivity_fold", "derived_delta_mean", "derived_selectivity_fold",
        "mut_ic50_lt50_votes", "wt_ic50_gt100_votes", "direct_selectivity_gt5_votes",
        "derived_selectivity_gt5_votes", "all_four_filters_votes", "potential_positive",
    ]
    state = {"last_rowid": 0, "rows_read": 0, "valid_scored": 0, "invalid_smiles": 0,
             "potential_positives": 0, "completed": False}
    if args.checkpoint.exists():
        state.update(json.loads(args.checkpoint.read_text()))
    output_exists = args.output.exists() and args.output.stat().st_size > 0
    positive_exists = args.positives.exists() and args.positives.stat().st_size > 0
    invalid_exists = args.invalid.exists() and args.invalid.stat().st_size > 0
    mode = "at" if state["last_rowid"] else "wt"
    started = time.time(); processed_this_run = 0

    database_path = args.database.resolve()
    connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    with gzip.open(args.output, mode, newline="") as all_handle, \
         gzip.open(args.positives, mode, newline="") as positive_handle, \
         gzip.open(args.invalid, mode, newline="") as invalid_handle:
        all_writer = csv.DictWriter(all_handle, fieldnames=fields)
        positive_writer = csv.DictWriter(positive_handle, fieldnames=fields)
        invalid_writer = csv.writer(invalid_handle)
        if not output_exists: all_writer.writeheader()
        if not positive_exists: positive_writer.writeheader()
        if not invalid_exists: invalid_writer.writerow(["rowid", "hit_id", "smiles", "source_file", "library_filename"])
        while True:
            limit = args.batch_size
            if args.max_molecules is not None:
                remaining = args.max_molecules - processed_this_run
                if remaining <= 0: break
                limit = min(limit, remaining)
            records = connection.execute(
                "SELECT rowid, hit_id, smiles, source_file, filename FROM molecules "
                "WHERE rowid > ? ORDER BY rowid LIMIT ?", (state["last_rowid"], limit)
            ).fetchall()
            if not records:
                state["completed"] = True; break
            valid, invalid, fp = fingerprint_batch(records, generator)
            for row in invalid: invalid_writer.writerow(row)
            if valid:
                raw = predict(fp, fold_contexts, args.workers)
                mut = np.stack([raw[(fold, "mut")] for fold in range(5)], axis=1)
                wt = np.stack([raw[(fold, "wt")] for fold in range(5)], axis=1)
                delta = np.stack([raw[(fold, "delta")] for fold in range(5)], axis=1)
                derived = mut - wt
                mut_mean, wt_mean, delta_mean, derived_mean = (
                    mut.mean(1), wt.mean(1), delta.mean(1), derived.mean(1))
                mut_ic50, wt_ic50 = 10 ** (9 - mut_mean), 10 ** (9 - wt_mean)
                direct_sel, derived_sel = 10 ** delta_mean, 10 ** derived_mean
                positive = ((mut_ic50 < MUT_IC50_MAX_NM) & (wt_ic50 > WT_IC50_MIN_NM) &
                            (direct_sel > SELECTIVITY_MIN_FOLD) &
                            (derived_sel > SELECTIVITY_MIN_FOLD))
                for i, record in enumerate(valid):
                    row = {
                        "hit_id": record[1], "smiles": record[2], "source_file": record[3],
                        "library_filename": record[4],
                        "p_activity_mut_mean": f"{mut_mean[i]:.6f}", "p_activity_mut_sd": f"{mut[i].std(ddof=1):.6f}",
                        "mut_ic50_nM": f"{mut_ic50[i]:.6f}",
                        "p_activity_wt_mean": f"{wt_mean[i]:.6f}", "p_activity_wt_sd": f"{wt[i].std(ddof=1):.6f}",
                        "wt_ic50_nM": f"{wt_ic50[i]:.6f}",
                        "delta_pactivity_direct_mean": f"{delta_mean[i]:.6f}",
                        "delta_pactivity_direct_sd": f"{delta[i].std(ddof=1):.6f}",
                        "direct_selectivity_fold": f"{direct_sel[i]:.6f}",
                        "derived_delta_mean": f"{derived_mean[i]:.6f}",
                        "derived_selectivity_fold": f"{derived_sel[i]:.6f}",
                        "mut_ic50_lt50_votes": int((10 ** (9-mut[i]) < 50).sum()),
                        "wt_ic50_gt100_votes": int((10 ** (9-wt[i]) > 100).sum()),
                        "direct_selectivity_gt5_votes": int((10 ** delta[i] > 5).sum()),
                        "derived_selectivity_gt5_votes": int((10 ** derived[i] > 5).sum()),
                        "all_four_filters_votes": int(((10 ** (9-mut[i]) < 50) & (10 ** (9-wt[i]) > 100) &
                                                       (10 ** delta[i] > 5) & (10 ** derived[i] > 5)).sum()),
                        "potential_positive": int(positive[i]),
                    }
                    all_writer.writerow(row)
                    if positive[i]: positive_writer.writerow(row)
                state["valid_scored"] += len(valid)
                state["potential_positives"] += int(positive.sum())
            state["invalid_smiles"] += len(invalid)
            state["rows_read"] += len(records); processed_this_run += len(records)
            state["last_rowid"] = records[-1][0]
            all_handle.flush(); positive_handle.flush(); invalid_handle.flush()
            elapsed = time.time() - started
            state["elapsed_seconds_this_run"] = elapsed
            state["molecules_per_second_this_run"] = processed_this_run / elapsed
            write_checkpoint(args.checkpoint, state)
            print(json.dumps(state), flush=True)
    connection.close(); write_checkpoint(args.checkpoint, state)
    audit = {**state, "database": str(args.database), "ensemble": str(args.ensemble),
             "seed": 104, "folds": 5, "models_per_fold": ["mutant SVR", "WT SVR", "direct-delta SVR"],
             "mutation": "SUB:T790M;SUB:C797S;SUB:L858R", "endpoint": "IC50",
             "esm_sequence_quality": sequence_quality, "rdkit_version": rdBase.rdkitVersion,
             "thresholds": {"mut_ic50_nM_lt": 50, "wt_ic50_nM_gt": 100,
                            "direct_selectivity_fold_gt": 5, "derived_selectivity_fold_gt": 5},
             "ensemble_rule": "mean on pActivity/delta scale, then exponentiate"}
    args.audit.write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
