# Model card

## Paired-regression ensemble

- File: `regression/seed104_three_task_fivefold_ensemble.joblib`
- Compound representation: RDKit ECFP6, Morgan radius 3, 2,048 bits
- Mutation representation: ΔESM-2 for mutant and direct ΔpActivity models;
  no mutation representation for the WT model
- Estimator: radial-basis-function support vector regression
- Outputs: mutant pActivity, WT pActivity and direct ΔpActivity
- Ensemble: five scaffold-fold estimators from seed 104
- Each fold contains three separately trained SVRs and its fitted ESM PCA/scaler.
- WT inputs: compound fingerprint plus endpoint; mutant/delta inputs additionally use transformed ESM differences and reconstruction-quality indicators.
- Screening endpoint: IC50; endpoint order: IC50, Ki, Kd.
- Average predictions across folds on the pActivity/delta scale before converting to nM or selectivity ratios.

## Triple-mutant classifier

- File: `classification/mutant_ECFP6_SVM_full_data.joblib`
- Compound representation: archived R/CDK ECFP6, 1,024 bits; use the supplied matrices for reproduction.
- Estimator: support vector machine with radial-basis-function kernel
- Output: binary activity assessment for EGFR L858R/T790M/C797S
- Parameters: C=10, gamma=scale, class_weight=balanced, random_state=2026.
- Positive classification: decision-function score >=0; scores are not calibrated probabilities.
- Training records: 997 measured active, 339 measured inactive and 763 presumed negative compounds.
- Label definition: measured IC50 below 50 nM is active; measured IC50 above
  500 nM is inactive; an additional 763 ChEMBL compounds were included as
  presumed-negative examples

Classifier hyperparameters were selected by ten-fold cross-validation within
the original training set. They were subsequently applied consistently across
the repeated scaffold partitions. The classifier does not predict WT activity
and must not be interpreted as a direct model of selectivity.

## Intended use

The models prioritize compounds for experimental evaluation. They do not
establish biochemical potency, cellular efficacy, allele selectivity, safety
or clinical activity without experimental testing.
