#!/usr/bin/env python3
"""Prepare auditable EGFR kinase-domain WT/mutant sequence pairs for ESM encoding."""

# Resolve shared modules across the six workflow stages.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(p) for p in _Path(__file__).resolve().parents[1].glob("[0-9][0-9]_*") if p.is_dir()]

import argparse
import json
import re
from pathlib import Path

import pandas as pd


SUB = re.compile(r"SUB:([A-Z])(\d+)([A-Z])$")
DEL_RANGE = re.compile(r"DEL:(\d+)-(\d+)$")
INS_BETWEEN = re.compile(r"INS:[A-Z](\d+)[A-Z](\d+)INS([A-Z]+)$")
INS_AFTER = re.compile(r"INS:[A-Z](\d+)INS([A-Z]+)$")


def apply_token(sequence, token):
    """Apply curated EGFR edits in canonical UniProt numbering; return sequence and audit."""
    edits, unresolved, positions = [], [], []
    parts = str(token).split(";")
    for part in parts:
        if match := SUB.fullmatch(part):
            wt, pos, mut = match.groups(); pos = int(pos)
            if sequence[pos - 1] != wt:
                unresolved.append(f"{part}:reference_is_{sequence[pos - 1]}")
            else:
                edits.append((pos, pos, mut, part)); positions.append(pos)
        elif match := DEL_RANGE.fullmatch(part):
            start, end = map(int, match.groups())
            edits.append((start, end, "", part)); positions.extend([start, end])
        elif part == "DEL:EXON19":
            # Coarse source annotation: use the canonical EGFR E746_A750 deletion surrogate.
            edits.append((746, 750, "", part)); positions.extend([746, 750])
        elif part == "DEL:EXON18":
            unresolved.append(f"{part}:coarse_exon_annotation")
        elif match := INS_BETWEEN.fullmatch(part):
            left, right, inserted = match.groups(); left, right = int(left), int(right)
            if right != left + 1:
                unresolved.append(f"{part}:nonadjacent_flanks")
            else:
                edits.append((right, right - 1, inserted, part)); positions.extend([left, right])
        elif match := INS_AFTER.fullmatch(part):
            pos, inserted = match.groups(); pos = int(pos)
            edits.append((pos + 1, pos, inserted, part)); positions.append(pos)
        else:
            unresolved.append(f"{part}:unparsed")
    mutant = sequence
    for start, end, replacement, _ in sorted(edits, key=lambda x: x[0], reverse=True):
        mutant = mutant[:start - 1] + replacement + mutant[end:]
    quality = "exact" if not unresolved and "EXON" not in str(token) else (
        "surrogate" if not unresolved else "unresolved")
    return mutant, sorted(set(positions)), quality, unresolved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--constructs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--domain-start", type=int, default=650)
    parser.add_argument("--domain-end", type=int, default=1000)
    args = parser.parse_args()
    pairs = pd.read_csv(args.pairs, sep="\t", low_memory=False)
    constructs = pd.read_csv(args.constructs, sep="\t", low_memory=False)
    wt = constructs[(constructs.sequence_length.eq(1210)) & constructs.explicit_wt.fillna(False) &
                    constructs.sequence_annotation_reliable.fillna(False)]
    if len(wt) != 1:
        raise RuntimeError(f"Expected one reliable canonical WT sequence, found {len(wt)}")
    sequence = wt.iloc[0].protein_sequence
    rows = []
    for token, n_rows in pairs.v2_variant_token.value_counts().items():
        mutant, positions, quality, unresolved = apply_token(sequence, token)
        # Slice after edits using a conservative domain window; sequence edits are all within this window.
        start, end = args.domain_start, args.domain_end
        wt_domain = sequence[start - 1:end]
        # Preserve a comparable C-terminal boundary after indels by slicing the edited full sequence.
        mut_domain = mutant[start - 1:end + (len(mutant) - len(sequence))]
        rows.append({
            "v2_variant_token": token, "n_pair_rows": int(n_rows),
            "sequence_quality": quality, "unresolved_reason": ";".join(unresolved),
            "mutation_positions": ";".join(map(str, positions)),
            "domain_start": start, "domain_end": end,
            "wt_domain_sequence": wt_domain, "mutant_domain_sequence": mut_domain,
            "wt_length": len(wt_domain), "mutant_length": len(mut_domain),
        })
    out = pd.DataFrame(rows).sort_values(["n_pair_rows", "v2_variant_token"], ascending=[False, True])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, sep="\t", index=False)
    audit = {
        "canonical_construct_id": wt.iloc[0].canonical_construct_id,
        "canonical_length": len(sequence), "domain": [args.domain_start, args.domain_end],
        "unique_variant_tokens": len(out), "paired_rows": int(out.n_pair_rows.sum()),
        "quality_by_token": out.sequence_quality.value_counts().to_dict(),
        "coverage_by_pair_row": out.groupby("sequence_quality").n_pair_rows.sum().to_dict(),
        "exon19_surrogate": "E746_A750del",
    }
    args.output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
