#!/usr/bin/env python3
"""Extract auditable human kinase measurements from a ChEMBL SQLite release."""

from __future__ import annotations

# Resolve shared modules across the six workflow stages.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(p) for p in _Path(__file__).resolve().parents[1].glob("[0-9][0-9]_*") if p.is_dir()]

import argparse
import gzip
import json
import math
import sqlite3
import time
from pathlib import Path


ENDPOINTS = ("IC50", "KI", "KD")


def kinase_targets(con: sqlite3.Connection) -> list[int]:
    query = """
    WITH RECURSIVE kinase_classes(protein_class_id) AS (
      SELECT 6
      UNION ALL
      SELECT pc.protein_class_id
      FROM protein_classification pc
      JOIN kinase_classes kc ON pc.parent_id = kc.protein_class_id
    )
    SELECT DISTINCT td.tid
    FROM target_dictionary td
    JOIN target_components tc ON tc.tid = td.tid
    JOIN component_class cc ON cc.component_id = tc.component_id
    WHERE td.tax_id = 9606
      AND td.target_type = 'SINGLE PROTEIN'
      AND cc.protein_class_id IN (SELECT protein_class_id FROM kinase_classes)
    ORDER BY td.tid
    """
    return [row[0] for row in con.execute(query)]


def batches(values: list[int], size: int):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-batch-size", type=int, default=20)
    parser.add_argument("--limit-targets", type=int)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / "chembl37_human_kinase_measurements.tsv.gz"
    audit_path = args.output / "chembl37_extraction_audit.json"
    uri = f"file:{args.database.resolve()}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.execute("PRAGMA query_only=ON")
    con.execute("PRAGMA temp_store=MEMORY")

    targets = kinase_targets(con)
    if args.limit_targets:
        targets = targets[:args.limit_targets]
    if not targets:
        raise RuntimeError("No human single-protein kinase targets found")

    select_sql = """
    SELECT
      a.activity_id, a.assay_id, a.doc_id, a.record_id AS chembl_record_id,
      a.molregno, md.chembl_id AS compound_chembl_id,
      cs.canonical_smiles, cs.standard_inchi, cs.standard_inchi_key AS inchikey,
      UPPER(a.standard_type) AS endpoint, a.standard_relation AS relation,
      a.standard_value AS value_nM, a.standard_units,
      CASE WHEN a.standard_value > 0
           THEN 9.0 - log(a.standard_value) / log(10.0) END AS p_activity_calculated,
      a.pchembl_value, a.standard_flag, a.data_validity_comment,
      a.potential_duplicate, a.activity_comment, a.action_type, a.modality,
      ass.assay_type, ass.assay_test_type, ass.assay_category,
      ass.description AS assay_description, ass.assay_organism, ass.assay_tax_id,
      ass.assay_cell_type, ass.assay_tissue, ass.assay_subcellular_fraction,
      ass.relationship_type, ass.confidence_score, ass.bao_format,
      ass.variant_id, vs.mutation AS annotated_mutation,
      vs.accession AS variant_accession, vs.sequence AS variant_sequence,
      td.tid, td.chembl_id AS target_chembl_id, td.pref_name AS target_name,
      td.organism AS target_organism, comp.accession AS uniprot_accession,
      comp.sequence AS canonical_protein_sequence,
      COALESCE(vs.sequence, comp.sequence) AS protein_sequence,
      CASE WHEN ass.variant_id IS NULL THEN 'unannotated_or_WT'
           ELSE 'annotated_variant' END AS construct_status,
      d.year, d.pubmed_id, d.doi, d.patent_id, d.chembl_id AS document_chembl_id,
      d.doc_type, src.src_short_name AS source_name,
      CASE WHEN UPPER(a.standard_type) IN ('IC50','KI','KD')
                 AND a.standard_units = 'nM' AND a.standard_value > 0
                 AND a.standard_relation = '=' AND a.standard_flag = 1
                 AND a.data_validity_comment IS NULL
                 AND COALESCE(a.potential_duplicate, 0) = 0
                 AND ass.confidence_score >= 8
                 AND cs.canonical_smiles IS NOT NULL
                 AND COALESCE(vs.sequence, comp.sequence) IS NOT NULL
           THEN 1 ELSE 0 END AS eligible_regression
    FROM assays ass
    JOIN activities a ON a.assay_id = ass.assay_id
    JOIN target_dictionary td ON td.tid = ass.tid
    JOIN target_components tc ON tc.tid = td.tid
    JOIN component_sequences comp ON comp.component_id = tc.component_id
    JOIN molecule_dictionary md ON md.molregno = a.molregno
    LEFT JOIN compound_structures cs ON cs.molregno = a.molregno
    LEFT JOIN variant_sequences vs ON vs.variant_id = ass.variant_id
    LEFT JOIN docs d ON d.doc_id = a.doc_id
    LEFT JOIN source src ON src.src_id = a.src_id
    WHERE ass.tid IN ({placeholders})
      AND UPPER(a.standard_type) IN ('IC50','KI','KD')
      AND a.standard_value IS NOT NULL
      AND a.standard_units = 'nM'
    ORDER BY ass.tid, a.activity_id
    """

    columns: list[str] | None = None
    total = eligible = annotated_variants = 0
    endpoint_counts = {key: 0 for key in ENDPOINTS}
    start = time.time()
    with gzip.open(out_path, "wt", encoding="utf-8", newline="") as handle:
        for batch_no, target_batch in enumerate(batches(targets, args.target_batch_size), 1):
            placeholders = ",".join("?" for _ in target_batch)
            cursor = con.execute(select_sql.format(placeholders=placeholders), target_batch)
            if columns is None:
                columns = [item[0] for item in cursor.description]
                handle.write("\t".join(columns) + "\n")
            for row in cursor:
                values = []
                record = dict(zip(columns, row))
                for value in row:
                    if value is None:
                        values.append("")
                    elif isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                        values.append("")
                    else:
                        values.append(str(value).replace("\t", " ").replace("\r", " ").replace("\n", " "))
                handle.write("\t".join(values) + "\n")
                total += 1
                eligible += int(record["eligible_regression"] or 0)
                annotated_variants += int(record["variant_id"] is not None)
                endpoint_counts[record["endpoint"]] += 1
            print(
                f"batch={batch_no} targets={min(batch_no * args.target_batch_size, len(targets))}/"
                f"{len(targets)} rows={total:,} eligible={eligible:,}", flush=True
            )

    audit = {
        "chembl_release": 37,
        "database": str(args.database),
        "human_single_protein_kinase_targets": len(targets),
        "output_rows": total,
        "eligible_regression_rows": eligible,
        "annotated_variant_rows": annotated_variants,
        "endpoint_counts": endpoint_counts,
        "target_batch_size": args.target_batch_size,
        "elapsed_seconds": round(time.time() - start, 1),
        "filters": {
            "organism_tax_id": 9606,
            "target_type": "SINGLE PROTEIN",
            "endpoints": list(ENDPOINTS),
            "standard_units": "nM",
            "eligible_relation": "=",
            "minimum_confidence_score": 8,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    con.close()
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
