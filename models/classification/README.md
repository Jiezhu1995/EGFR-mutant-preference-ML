# Frozen triple-mutant classifier

`mutant_ECFP6_SVM_full_data.joblib` is the full-data RBF-SVM.
Its inputs are the supplied 1,024-bit ECFP6 matrices. A decision-function score
>=0 defines an active prediction. This classifier evaluates triple-mutant
activity; WT preference is assessed by the regression stage.

Use `python reproduce.py train-classifier` to refit from the supplied matrices.
The new model is written to `generated/retrained_classifier.joblib`, without
overwriting the supplied model. Classification uses the supplied R/CDK matrices.
