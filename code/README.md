# Code organized in manuscript workflow order

| Folder | Purpose |
|---|---|
| `01_data_collection` | Database extraction, assay curation and WT-mutant pairing |
| `02_feature_generation` | EGFR sequence reconstruction, ESM features and regression fingerprints |
| `03_regression_models` | Frozen partitions, conventional and neural regression, generic five-fold SVR training |
| `04_triple_mutant_classifier` | Supplied fingerprint matrix loading, classifier training and matched validation |
| `05_virtual_screening` | Library inference, selection filters, connectivity and literature audits |
| `06_figures_and_statistics` | Computational figures, seed-level summaries and statistical comparisons |

Use `reproduce.py` in the release root for supported commands; the root
`README.md` lists each task. Classification uses the provided fingerprint
matrices and fixed model configurations.

Candidate audits and Supplementary Figure 7 document the screening hits and
experimental prioritization. Supplementary Figure 6 is reproduced from the
provided source-control metrics using the plotting and statistical verification
scripts.
