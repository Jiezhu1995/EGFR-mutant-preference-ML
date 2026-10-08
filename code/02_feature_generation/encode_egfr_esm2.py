#!/usr/bin/env python3
"""Encode frozen EGFR WT/mutant kinase-domain sequence pairs with ESM-2."""

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
from transformers import AutoModel, AutoTokenizer


def pool(hidden, mask):
    values = hidden[mask]
    return values.mean(0) if len(values) else hidden.mean(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="facebook/esm2_t30_150M_UR50D")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--local-radius", type=int, default=16)
    args = parser.parse_args()
    data = pd.read_csv(args.sequences, sep="\t").fillna("")
    unique_sequences = pd.unique(pd.concat([data.wt_domain_sequence, data.mutant_domain_sequence], ignore_index=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache)
    model = AutoModel.from_pretrained(args.model, cache_dir=args.cache).to(device).eval()
    encoded = {}
    with torch.inference_mode():
        for start in range(0, len(unique_sequences), args.batch_size):
            seqs = list(unique_sequences[start:start + args.batch_size])
            tokens = tokenizer(seqs, return_tensors="pt", padding=True, add_special_tokens=True)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            hidden = model(**tokens).last_hidden_state.float().cpu()
            attention = tokens["attention_mask"].cpu().bool()
            for i, sequence in enumerate(seqs):
                # ESM adds BOS/EOS. Retain residue positions only.
                encoded[sequence] = hidden[i, 1:1 + len(sequence)].numpy()
            print(f"encoded {min(start + args.batch_size, len(unique_sequences))}/{len(unique_sequences)}", flush=True)
    wt_global, mut_global, diff_global = [], [], []
    wt_local, mut_local, diff_local = [], [], []
    for row in data.itertuples(index=False):
        wt = encoded[row.wt_domain_sequence]
        mut = encoded[row.mutant_domain_sequence]
        wg, mg = wt.mean(0), mut.mean(0)
        positions = [int(x) for x in str(row.mutation_positions).split(";") if x]
        relative = [p - int(row.domain_start) for p in positions]
        wmask = np.zeros(len(wt), dtype=bool); mmask = np.zeros(len(mut), dtype=bool)
        for pos in relative:
            wmask[max(0, pos - args.local_radius):min(len(wt), pos + args.local_radius + 1)] = True
            mmask[max(0, pos - args.local_radius):min(len(mut), pos + args.local_radius + 1)] = True
        wl, ml = pool(wt, wmask), pool(mut, mmask)
        if row.sequence_quality == "unresolved":
            mg, ml = wg.copy(), wl.copy()
        wt_global.append(wg); mut_global.append(mg); diff_global.append(mg - wg)
        wt_local.append(wl); mut_local.append(ml); diff_local.append(ml - wl)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        variant_tokens=data.v2_variant_token.to_numpy(object),
        sequence_quality=data.sequence_quality.to_numpy(object),
        wt_global=np.asarray(wt_global, dtype=np.float16),
        mutant_global=np.asarray(mut_global, dtype=np.float16),
        difference_global=np.asarray(diff_global, dtype=np.float16),
        wt_local=np.asarray(wt_local, dtype=np.float16),
        mutant_local=np.asarray(mut_local, dtype=np.float16),
        difference_local=np.asarray(diff_local, dtype=np.float16),
    )
    audit = {
        "model": args.model, "device": str(device), "torch": torch.__version__,
        "unique_sequences": len(unique_sequences), "variant_tokens": len(data),
        "embedding_dimension": int(np.asarray(wt_global).shape[1]),
        "local_radius": args.local_radius,
        "unresolved_variants_zero_difference": int(data.sequence_quality.eq("unresolved").sum()),
    }
    args.output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
