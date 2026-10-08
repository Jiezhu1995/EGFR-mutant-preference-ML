# Numerical results by manuscript figure

| Manuscript display | Numerical source in this release |
|---|---|
| Figure 1 and Supplementary Figure 7: screening funnel and candidate identities | `figure1_screening/corrected_full_library/` |
| Figure 2: regression fingerprints, variant representations and model comparisons | `figure2_regression/raw/`, `figure2_regression/repeated_scaffold/` and `figure2_regression/summary/` |
| Supplementary Figures 2 and 3: additional regression endpoints and metrics | Figure 2 seed-level and fold-level results |
| Supplementary Figure 4: paired out-of-fold predictions | `figure2_regression/raw/protein_representation/svr/test_predictions.tsv.gz` and `figure2_regression/repeated_scaffold/seed*/protein_svr/test_predictions.tsv.gz` |
| Figure 3: fingerprint, classifier and validation-strategy comparisons | `figure3_classifier/all_model_fold_metrics.csv` and `figure3_classifier/paired_validation_strategy_metrics.csv` |
| Supplementary Figure 5: secondary classification metrics | `figure3_classifier/all_model_fold_metrics.csv` |
| Supplementary Figure 6: source-controlled classification and label randomization | `figure3_classifier/source_control_fold_metrics.csv`, `source_control_summary.csv` and `source_label_counts.csv` |

Supplementary Figure 1 uses the model-development records in `data/`.
Figures 4 and 5 and Supplementary Figure 8 present experimental results.

The directory contains fold-level metrics, aggregate metrics and the individual
predictions used for out-of-fold displays. `python reproduce.py figures`
generates figure exports under `generated/figure_rebuild/figures/`.
