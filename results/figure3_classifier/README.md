# Triple-mutant classifier results

This directory contains the numerical inputs for Figure 3 and Supplementary
Figures 5 and 6. Classification uses the supplied 1,024-bit ECFP6 matrices and
an RBF-SVM (`C=10`, `gamma=scale`). The source-controlled analysis uses ten
scaffold-split seeds, five folds per seed, and ChEMBL (968 compounds),
patent-derived records (1,039) and BindingDB (92). The label-randomization
analysis contains 20 permutations per null design.

## Source-controlled results

| Analysis | Dataset | ROC-AUC, mean ± SD | MCC, mean ± SD | Balanced accuracy |
|---|---|---:|---:|---:|
| Within-source scaffold CV | ChEMBL | 0.957 ± 0.037 | 0.734 ± 0.109 | 0.832 |
| Within-source scaffold CV | Patent | 0.981 ± 0.020 | 0.866 ± 0.065 | 0.925 |
| Within-source scaffold CV | BindingDB | 0.981 ± 0.032 | 0.852 ± 0.131 | 0.922 |
| Source-balanced scaffold CV | All sources | 0.964 ± 0.015 | 0.817 ± 0.060 | 0.908 |
| Source-balanced label randomization | All sources | 0.496 ± 0.026 | −0.012 ± 0.048 | 0.494 |

Cross-source ROC-AUC values were 0.839 for Patent→ChEMBL, 0.755 for
ChEMBL→Patent and 0.419 for ChEMBL+Patent→BindingDB. These analyses report
within-source performance, balanced-source performance and cross-source
transfer as separate evaluations. BindingDB contributes 92 compounds, including
22 active labels.

## Files

- `all_model_fold_metrics.csv`: fingerprint and classifier fold-level metrics.
- `model_performance_summary.csv`: grouped model-performance metrics.
- `paired_validation_strategy_metrics.csv`: matched random/scaffold validation.
- `source_control_fold_metrics.csv`: source-controlled and permutation metrics.
- `source_control_summary.csv`: grouped source-controlled metrics.
- `source_label_counts.csv`: compound counts by source and label.
- `run_metadata.json`: analysis settings and seed identifiers.

`python reproducibility/check_sf6.py` verifies the Supplementary Figure 6
statistics. `python reproduce.py figures` generates the corresponding figure.
