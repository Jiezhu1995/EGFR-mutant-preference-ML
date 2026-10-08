# EGFR mutant-preference modelling

## What this project does

This repository contains the computational workflow used to prioritize
compounds for EGFR L858R/T790M/C797S. Separate regression models predict
mutant activity, wild-type (WT) activity and their activity difference. A
triple-mutant activity classifier provides an additional screening filter.
The release includes the processed model-development data, scaffold splits,
code, fitted screening models and numerical results underlying the
computational figures.

The paired-regression dataset contains 1,223 records representing 810
compound-connectivity groups and 30 EGFR variant settings. The classifier
dataset contains 2,099 compounds: 997 active labels and 1,102 negative labels,
including 763 ChEMBL-sampled presumed negatives. The screening records trace
the candidate set from 529 regression-positive structures to 16 consensus
hits, five compounds selected for initial biochemical evaluation and two
compounds selected for detailed follow-up.

## Folder guide

| Folder | Contents |
|---|---|
| `data/` | Curated EGFR records, scaffold splits, molecular fingerprints and EGFR sequence features |
| `code/` | Data curation, feature generation, model training, screening and figure/statistics scripts, ordered by workflow stage |
| `models/` | Fitted seed-104 regression ensemble and full-data triple-mutant classifier |
| `results/` | Numerical results organized by Figure 1 screening, Figure 2 regression and Figure 3 classification; see `results/README.md` for supplementary-figure mappings |
| `reproducibility/` | File hashes, verification scripts, refit comparison and external-asset list |

## How to use it

Run commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python reproduce.py verify
python reproduce.py figures
```

`verify` checks file integrity, the ten scaffold partitions and model inference.
`figures` rebuilds the supported computational figures from the supplied
numerical results. Both write temporary output under the ignored `generated/`
directory and leave the supplied data and models unchanged.

The principal additional commands are:

| Task | Command |
|---|---|
| Refit the seed-104 regression ensemble | `python reproduce.py train-regression --seed 104` |
| Refit the triple-mutant classifier | `python reproduce.py train-classifier` |
| Rerun classifier benchmarks | `python reproduce.py classifier-benchmark` |
| Compare random and scaffold validation | `python reproduce.py classifier-validation` |
| Rerun regression fingerprint or variant benchmarks | `python reproduce.py benchmark-fingerprints --seed 104` or `python reproduce.py benchmark-protein --seed 104` |
| Check Supplementary Figure 6 statistics | `python reproducibility/check_sf6.py` |

Regression screening uses the seed-104 five-fold ensemble. Predictions are
averaged on the pActivity scale, with the endpoint set to IC₅₀, before
conversion to concentration or selectivity. The triple-mutant classifier uses
the supplied 1,024-bit R/CDK ECFP6 matrices and calls scores ≥0 active;
regression uses 2,048-bit RDKit ECFP6. The full commercial-library screen
requires an authorized copy of that library and is not rerun by `verify` or
`figures`. Experimental measurements and crystallographic files are outside
this computational package.

## Reproduction environment

Use Python with the packages listed in `requirements.txt`: NumPy, pandas,
SciPy, scikit-learn, RDKit, joblib, Matplotlib, XGBoost, PyYAML, statsmodels
and ReportLab. For multitask MLP or GIN training, install
`requirements-deep.txt`, which adds PyTorch, PyTorch Geometric and
Transformers. On macOS, XGBoost also needs an OpenMP runtime.

The fitted regression model records scikit-learn 1.9.0; numerical comparisons
with a local refit are in `reproducibility/refit_comparison.json`. Load joblib
model files only from trusted sources. Public distribution is subject to
source-data redistribution terms and an approved repository license.
