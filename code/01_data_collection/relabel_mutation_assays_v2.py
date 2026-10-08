#!/usr/bin/env python3
"""Conservative assay-level WT/mutant relabeling for mutation-aware v2 data.

Unknown constructs are never promoted to WT.  The output retains the raw text,
normalized variant tokens, evidence source, assay class, and exclusion reason so
every accepted WT-mutant pair can be traced back to an assay record.
"""
from __future__ import annotations

# Resolve shared modules across the six workflow stages.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(p) for p in _Path(__file__).resolve().parents[1].glob("[0-9][0-9]_*") if p.is_dir()]

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


POINT_RE = re.compile(r"(?<![A-Za-z])([A-Z][0-9]{1,5}[A-Z])(?![A-Za-z])", re.I)
WT_RE = re.compile(r"\b(?:wild[ -]?type|WT|non[ -]?mutant)\b", re.I)
DEL_PATTERNS = [
    re.compile(r"\b([A-Z]\d{2,5})[_-]([A-Z]?\d{2,5})del\b", re.I),
    re.compile(r"\bdel(?:etion)?[ /_-]?(\d{1,5})(?:[ /_-]?(?:to|-|_)[ /_-]?(\d{1,5}))?\b", re.I),
    re.compile(r"\b(?:exon|ex|e)[ -]?(\d{1,2})[ -]?(?:del|deletion)\b", re.I),
]
INS_PATTERNS = [
    re.compile(r"\b([A-Z]\d{2,5})[_-]([A-Z]?\d{2,5})ins([A-Z]+)\b", re.I),
    re.compile(r"\b(?:exon|ex|e)[ -]?(\d{1,2})[ -]?(?:ins|insertion)\b", re.I),
    re.compile(r"\b([A-Z]?\d{2,5})[ -]?(?:ins|insertion)[ -]?([A-Z]+)?\b", re.I),
]
FUSION_RE = re.compile(r"\b(?:fusion|rearrangement|translocation|EML4[- :]ALK|BCR[- :]ABL)\b", re.I)


def clean(value) -> str:
    return "" if pd.isna(value) else str(value).strip()


def normalize_token(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", value.upper())


def canonical_deletion_token(token: str) -> str:
    raw = token.upper().removeprefix("DEL:")
    exon = re.search(r"EX(?:ON)?(\d{1,2})", raw)
    if exon:
        return f"DEL:EXON{exon.group(1)}"
    # DEL19/19DEL and DEL18/18DEL are conventional EGFR exon shorthands.
    short = re.fullmatch(r"(?:DEL)?(18|19)(?:DEL(?:ETION)?)?", raw)
    if short:
        return f"DEL:EXON{short.group(1)}"
    numbers = [int(x) for x in re.findall(r"\d{3,5}", raw)]
    if len(numbers) >= 2:
        return f"DEL:{numbers[0]}-{numbers[1]}"
    if len(numbers) == 1:
        return f"DEL:{numbers[0]}"
    return "DEL:" + raw


def parse_variants(text: str) -> dict:
    text = clean(text)
    points = sorted({x.upper() for x in POINT_RE.findall(text)}, key=lambda x: (int(x[1:-1]), x))
    deletions = set()
    insertions = set()
    for pattern in DEL_PATTERNS:
        for match in pattern.finditer(text):
            token = normalize_token(match.group(0))
            if token:
                deletions.add("DEL:" + token)
    # Common shorthand that does not always match the generic expressions.
    for match in re.finditer(r"\b(?:(\d{1,2})\s*del|del\s*(\d{1,2}))\b", text, re.I):
        exon = match.group(1) or match.group(2)
        deletions.add(f"DEL:EXON{exon}DEL")
    for pattern in INS_PATTERNS:
        for match in pattern.finditer(text):
            token = normalize_token(match.group(0))
            if token:
                insertions.add("INS:" + token)
    # BindingDB often describes engineered constructs as retained coordinate
    # blocks, e.g. [1-745,751-1210,T790M].  Internal gaps are deletions; terminal
    # truncations are construct boundaries and must not be called deletions.
    bracket = re.search(r"\[([^\]]+)\]", text)
    if bracket:
        content = bracket.group(1)
        ranges = [(int(a), int(b)) for a, b in re.findall(r"(?<![A-Za-z0-9])(\d{1,5})\s*-\s*(\d{1,5})(?![A-Za-z0-9])", content)]
        ranges = sorted((a, b) for a, b in ranges if a <= b)
        for (_, left_end), (right_start, _) in zip(ranges, ranges[1:]):
            if right_start > left_end + 1:
                deletions.add(f"DEL:{left_end + 1}-{right_start - 1}")
        # Quoted residues between adjacent coordinate blocks denote insertions,
        # e.g. [1-770,'NPG',771-1210].
        for seq in re.findall(r"['\"]([A-Z]{1,30})['\"]", content, re.I):
            if ranges:
                insertions.add(f"INS:CONSTRUCT:{seq.upper()}")
    deletions = {canonical_deletion_token(x) for x in deletions}
    return {
        "points": points,
        "deletions": sorted(deletions),
        "insertions": sorted(insertions),
        "fusion": bool(FUSION_RE.search(text)),
        "explicit_wt": bool(WT_RE.search(text)),
    }


def assay_class(description: str) -> str:
    text = clean(description).lower()
    if re.search(r"prolifer|cell viability|cell growth|antiprolifer|celltiter|\bmts\b|\bmtt\b", text):
        return "cellular_proliferation"
    if re.search(r"phosphorylation.*(?:cell|cells)|(?:cell|cells).*phosphorylation|western blot|sandwich elisa", text):
        return "cellular_target_engagement"
    if re.search(r"binding constant|binding affinity|\b(?:kd|ki)\b|radioligand displacement|competition binding", text):
        return "biochemical_binding"
    if re.search(r"kinase assay|kinase activity|recombinant|atp|adp-glo|z.?lyte|htrf|mobility shift|peptide substrate", text):
        return "biochemical_activity"
    if re.search(r"\bcell(?:s|ular)?\b|ba/f3|pc-?9|h1975|a431", text):
        return "other_cellular"
    return "unknown"


def label_row(row: pd.Series) -> pd.Series:
    annotated = clean(row.get("annotated_mutation"))
    construct_mut = clean(row.get("mutation_string"))
    description = clean(row.get("assay_description"))
    # Structured annotation is highest priority, followed by verified construct
    # mutation and finally assay text.
    structured_text = " ; ".join(x for x in (annotated, construct_mut) if x and x.upper() != "UNDEFINED MUTATION")
    structured = parse_variants(structured_text)
    narrative = parse_variants(description)
    merged = {
        "points": sorted(set(structured["points"] + narrative["points"]), key=lambda x: (int(x[1:-1]), x)),
        "deletions": sorted(set(structured["deletions"] + narrative["deletions"])),
        "insertions": sorted(set(structured["insertions"] + narrative["insertions"])),
        "fusion": structured["fusion"] or narrative["fusion"],
    }
    tokens = [*("SUB:" + x for x in merged["points"]), *merged["deletions"], *merged["insertions"]]
    if merged["fusion"]:
        tokens.append("FUSION:MENTIONED")
    has_variant = bool(tokens)
    verified_mutant = bool(row.get("is_mutant") is True or row.get("is_mutant") == 1)
    verified_wt = bool(row.get("is_wild_type") is True or row.get("is_wild_type") == 1)
    explicit_wt = narrative["explicit_wt"]

    if has_variant and explicit_wt and not structured_text:
        state, reason = "AMBIGUOUS", "assay_text_mentions_wt_and_variant_without_structured_assignment"
    elif has_variant:
        state, reason = "MUTANT", "structured_or_assay_variant_evidence"
    elif verified_mutant:
        state, reason = "AMBIGUOUS", "verified_mutant_but_variant_identity_missing"
    elif verified_wt or explicit_wt:
        state, reason = "WT", "verified_construct_wt" if verified_wt else "explicit_wt_assay_text"
    else:
        state, reason = "UNKNOWN", "no_explicit_wt_or_variant_evidence"

    variant_types = []
    if merged["points"]:
        variant_types.append("substitution")
    if merged["deletions"]:
        variant_types.append("deletion")
    if merged["insertions"]:
        variant_types.append("insertion")
    if merged["fusion"]:
        variant_types.append("fusion")
    return pd.Series({
        "v2_state": state,
        "v2_variant_token": ";".join(tokens),
        "v2_variant_types": ";".join(variant_types),
        "v2_assay_class": assay_class(description),
        "v2_label_reason": reason,
        "v2_explicit_wt_text": explicit_wt,
        "v2_has_structured_annotation": bool(structured_text),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurements", required=True)
    parser.add_argument("--constructs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    measurements = pd.read_csv(args.measurements, sep="\t", low_memory=False)
    constructs = pd.read_csv(args.constructs, sep="\t", low_memory=False)
    construct_cols = ["construct_id", "uniprot_accession", "mutation_string", "is_wild_type", "is_mutant"]
    data = measurements.merge(constructs[construct_cols], on="construct_id", how="left", validate="many_to_one")
    labels = data.apply(label_row, axis=1)
    data = pd.concat([data, labels], axis=1)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    data.to_csv(out / "chembl37_assays_relabelled_v2.tsv.gz", sep="\t", index=False, compression="gzip")
    audit = {
        "rows": int(len(data)),
        "states": data.v2_state.value_counts(dropna=False).to_dict(),
        "variant_types": data.v2_variant_types.replace("", "none").value_counts().to_dict(),
        "assay_classes": data.v2_assay_class.value_counts().to_dict(),
        "label_reasons": data.v2_label_reason.value_counts().to_dict(),
        "accepted_wt_rows": int(data.v2_state.eq("WT").sum()),
        "accepted_mutant_rows": int(data.v2_state.eq("MUTANT").sum()),
        "excluded_unknown_or_ambiguous": int(data.v2_state.isin(["UNKNOWN", "AMBIGUOUS"]).sum()),
    }
    (out / "chembl37_relabel_audit_v2.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
