# Frozen paired-regression model

`seed104_three_task_fivefold_ensemble.joblib` is the seed-104 screening
ensemble. Load its `fold_models` entries to obtain the mutant, WT and direct-delta
SVRs together with fitted ESM transforms. Do not average converted IC50 values:
average predicted pActivity first, then calculate `10 ** (9 - mean_pActivity)`.

The general trainer is `code/03_regression_models/train_svr_ensemble.py`.
Use `python reproduce.py train-regression --seed 104`; refits are written to
`generated/retrained_seed104/`, separately from this frozen artifact.
