#!/usr/bin/env python3
"""Conservative BindingDB target relabeling for mutation-aware v2."""
from __future__ import annotations

# Resolve shared modules across the six workflow stages.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(p) for p in _Path(__file__).resolve().parents[1].glob("[0-9][0-9]_*") if p.is_dir()]

import argparse
import json
from pathlib import Path

import pandas as pd

from relabel_mutation_assays_v2 import parse_variants


def label_target(name: str) -> pd.Series:
    parsed = parse_variants(name)
    tokens = [
        *("SUB:" + x for x in parsed["points"]),
        *parsed["deletions"],
        *parsed["insertions"],
    ]
    if parsed["fusion"]:
        tokens.append("FUSION:MENTIONED")
    types = []
    if parsed["points"]:
        types.append("substitution")
    if parsed["deletions"]:
        types.append("deletion")
    if parsed["insertions"]:
        types.append("insertion")
    if parsed["fusion"]:
        types.append("fusion")
    if tokens and parsed["explicit_wt"]:
        state, reason = "AMBIGUOUS", "target_name_mentions_wt_and_variant"
    elif tokens:
        state, reason = "MUTANT", "explicit_variant_in_target_name"
    elif parsed["explicit_wt"]:
        state, reason = "WT", "explicit_wt_in_target_name"
    else:
        state, reason = "UNKNOWN", "no_explicit_wt_or_variant_evidence"
    return pd.Series({
        "v2_state": state,
        "v2_variant_token": ";".join(tokens),
        "v2_variant_types": ";".join(types),
        "v2_label_reason": reason,
    })


def endpoint_class(endpoint: str) -> str:
    endpoint = str(endpoint)
    if endpoint in {"Kd", "Ki"}:
        return "biochemical_binding"
    # BindingDB IC50 rows may be biochemical, target engagement, or cellular.
    # Without an assay description they must not be assigned a more specific class.
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--canonical-reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = pd.read_csv(args.input, sep="\t", low_memory=False)
    labels = data.target_name.fillna("").apply(label_target)
    data = pd.concat([data, labels], axis=1)
    reference_frame = pd.read_csv(
        args.canonical_reference,
        sep="\t",
        usecols=["uniprot_accession", "canonical_protein_sequence"],
        low_memory=False,
    ).dropna().drop_duplicates("uniprot_accession")
    reference = dict(zip(reference_frame.uniprot_accession, reference_frame.canonical_protein_sequence))

    def sequence_relation(row: pd.Series) -> str:
        sequence = "" if pd.isna(row.protein_sequence) else str(row.protein_sequence)
        canonical = reference.get(row.uniprot_accession, "")
        if not sequence or not canonical:
            return "missing_reference_or_sequence"
        if sequence == canonical:
            return "exact_canonical"
        if len(sequence) >= 150 and sequence in canonical:
            return "canonical_contiguous_subsequence"
        return "not_canonical_subsequence"

    data["v2_sequence_relation"] = data.apply(sequence_relation, axis=1)
    sequence_wt = (
        data.v2_state.eq("UNKNOWN")
        & data.v2_sequence_relation.isin(["exact_canonical", "canonical_contiguous_subsequence"])
    )
    data.loc[sequence_wt, "v2_state"] = "WT"
    data.loc[sequence_wt, "v2_label_reason"] = "sequence_confirmed_wt_construct"
    data["v2_assay_class"] = data.endpoint.map(endpoint_class)
    data["v2_regression_eligible"] = (
        data.is_exact.eq(True)
        & data.eligible_regression.eq(True)
        & data.endpoint.isin(["IC50", "Ki", "Kd"])
        & data.v2_state.isin(["WT", "MUTANT"])
    )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    data.to_csv(out / "bindingdb_assays_relabelled_v2.tsv.gz", sep="\t", index=False, compression="gzip")
    eligible = data[data.is_exact.eq(True) & data.eligible_regression.eq(True)]
    audit = {
        "rows": int(len(data)),
        "exact_originally_regression_eligible": int(len(eligible)),
        "states_all": data.v2_state.value_counts(dropna=False).to_dict(),
        "states_originally_eligible": eligible.v2_state.value_counts(dropna=False).to_dict(),
        "variant_types_originally_eligible": eligible.v2_variant_types.replace("", "none").value_counts().to_dict(),
        "sequence_relations_originally_eligible": eligible.v2_sequence_relation.value_counts().to_dict(),
        "accepted_v2_regression_rows": int(data.v2_regression_eligible.sum()),
        "accepted_v2_wt_rows": int((data.v2_regression_eligible & data.v2_state.eq("WT")).sum()),
        "accepted_v2_mutant_rows": int((data.v2_regression_eligible & data.v2_state.eq("MUTANT")).sum()),
    }
    (out / "bindingdb_relabel_audit_v2.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
