#!/usr/bin/env python3
"""Deduplicate strict v2 evidence and exclude cross-document conflicts."""
from __future__ import annotations

# Resolve shared modules across the six workflow stages.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(p) for p in _Path(__file__).resolve().parents[1].glob("[0-9][0-9]_*") if p.is_dir()]

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def join_unique(values) -> str:
    return ";".join(sorted({str(v) for v in values if pd.notna(v) and str(v)}))


def direction_consistency(values, minimum_signal: float = 0.3) -> float:
    values = np.asarray(values, dtype=float)
    signal = values[np.abs(values) >= minimum_signal]
    if not len(signal):
        return 1.0
    return float(max(np.mean(signal > 0), np.mean(signal < 0)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = pd.read_csv(args.pairs, sep="\t", low_memory=False)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    identity = [
        "inchikey_connectivity", "uniprot_accession", "v2_variant_token",
        "v2_variant_types", "endpoint", "v2_assay_class",
    ]
    # Same paper may be present in ChEMBL and BindingDB. Collapse source copies
    # before estimating cross-document variability.
    document_key = identity + ["document_key"]
    evidence = data.groupby(document_key, as_index=False).agg(
        p_activity_wt=("p_activity_wt", "median"),
        p_activity_mut=("p_activity_mut", "median"),
        delta_pactivity=("delta_pactivity_mut_minus_wt", "median"),
        source_databases=("source_database", join_unique),
        n_source_rows=("source_database", "size"),
        n_wt_records=("n_wt_records", "sum"),
        n_mut_records=("n_mut_records", "sum"),
        wt_record_ids=("wt_record_ids", join_unique),
        mutant_record_ids=("mutant_record_ids", join_unique),
        wt_assay_ids=("wt_assay_ids", join_unique),
        mutant_assay_ids=("mutant_assay_ids", join_unique),
        wt_label_reasons=("wt_label_reasons", join_unique),
        mutant_label_reasons=("mutant_label_reasons", join_unique),
        standardized_smiles=("standardized_smiles", "first"),
    )
    evidence.to_csv(out / "document_evidence_deduplicated_v2.tsv.gz", sep="\t", index=False, compression="gzip")

    aggregate = evidence.groupby(identity, as_index=False).agg(
        p_activity_wt=("p_activity_wt", "median"),
        p_activity_mut=("p_activity_mut", "median"),
        delta_pactivity=("delta_pactivity", "median"),
        wt_min=("p_activity_wt", "min"),
        wt_max=("p_activity_wt", "max"),
        mutant_min=("p_activity_mut", "min"),
        mutant_max=("p_activity_mut", "max"),
        delta_min=("delta_pactivity", "min"),
        delta_max=("delta_pactivity", "max"),
        delta_std=("delta_pactivity", "std"),
        direction_consistency=("delta_pactivity", direction_consistency),
        n_documents=("document_key", "nunique"),
        documents=("document_key", join_unique),
        source_databases=("source_databases", join_unique),
        n_document_evidence_rows=("document_key", "size"),
        n_wt_records=("n_wt_records", "sum"),
        n_mut_records=("n_mut_records", "sum"),
        standardized_smiles=("standardized_smiles", "first"),
    )
    aggregate["wt_range"] = aggregate.wt_max - aggregate.wt_min
    aggregate["mutant_range"] = aggregate.mutant_max - aggregate.mutant_min
    aggregate["delta_range"] = aggregate.delta_max - aggregate.delta_min
    aggregate["delta_std"] = aggregate.delta_std.fillna(0.0)

    severe = (
        (aggregate.delta_range > 1.0)
        | (aggregate.direction_consistency < 0.67)
        | (aggregate.wt_range > 1.5)
        | (aggregate.mutant_range > 1.5)
    )
    high = (
        ~severe
        & (aggregate.n_documents >= 2)
        & (aggregate.delta_range <= 0.5)
        & (aggregate.direction_consistency >= 0.8)
        & (aggregate.wt_range <= 1.0)
        & (aggregate.mutant_range <= 1.0)
    )
    aggregate["confidence"] = np.select([high, ~severe], ["high", "medium"], default="conflict")
    aggregate["pair_weight"] = aggregate.confidence.map({"high": 1.0, "medium": 0.6, "conflict": 0.0})
    aggregate["conflict_reason"] = ""
    aggregate.loc[aggregate.delta_range > 1.0, "conflict_reason"] += "delta_range_gt_1;"
    aggregate.loc[aggregate.direction_consistency < 0.67, "conflict_reason"] += "direction_inconsistent;"
    aggregate.loc[aggregate.wt_range > 1.5, "conflict_reason"] += "wt_range_gt_1_5;"
    aggregate.loc[aggregate.mutant_range > 1.5, "conflict_reason"] += "mutant_range_gt_1_5;"

    accepted = aggregate[~severe].copy()
    conflicts = aggregate[severe].copy()
    accepted.to_csv(out / "final_training_pairs_v2.tsv.gz", sep="\t", index=False, compression="gzip")
    conflicts.to_csv(out / "conflict_excluded_pairs_v2.tsv.gz", sep="\t", index=False, compression="gzip")
    aggregate.to_csv(out / "all_aggregated_pairs_v2.tsv.gz", sep="\t", index=False, compression="gzip")

    audit = {
        "input_strict_rows": int(len(data)),
        "document_evidence_after_cross_source_dedup": int(len(evidence)),
        "aggregated_pair_tasks": int(len(aggregate)),
        "accepted_training_pairs": int(len(accepted)),
        "conflict_excluded": int(len(conflicts)),
        "accepted_unique_compounds": int(accepted.inchikey_connectivity.nunique()),
        "accepted_kinases": int(accepted.uniprot_accession.nunique()),
        "accepted_mutation_tasks": int(accepted.groupby(["uniprot_accession", "v2_variant_token"]).ngroups),
        "accepted_egfr_pairs": int(accepted.uniprot_accession.eq("P00533").sum()),
        "confidence": accepted.confidence.value_counts().to_dict(),
        "conflict_reasons": conflicts.conflict_reason.value_counts().to_dict(),
    }
    (out / "aggregation_audit_v2.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
