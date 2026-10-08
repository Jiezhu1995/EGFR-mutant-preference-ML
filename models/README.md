# Frozen screening models

The package contains the seed-104 three-task five-fold SVR ensemble and the
full-data ECFP6-SVM triple-mutant classifier. Their specifications and scope
are documented in `MODEL_CARD.md`. Refit commands write separate models to
`generated/`.

| Directory | Frozen artifact | Contents |
|---|---|---|
| `regression/` | `seed104_three_task_fivefold_ensemble.joblib` | 15 SVRs: five for each of mutant activity, WT activity and direct delta; five fitted ESM PCA/scaler pairs |
| `classification/` | `mutant_ECFP6_SVM_full_data.joblib` | Full-data triple-mutant ECFP6-SVM fitted on 2,099 compounds |

Run `python reproduce.py verify` from the release root to check the packaged
hashes and model loading/inference. File hashes are in
`reproducibility/SHA256SUMS.txt`. Numerical refit comparisons are recorded in
`reproducibility/refit_comparison.json`.

The regression artifact records scikit-learn 1.9.0; numerical refit differences
are recorded in `reproducibility/refit_comparison.json`. Load joblib files only
from trusted sources because deserialization can execute code.
The model binaries are included directly in this package; Git LFS is not required.
