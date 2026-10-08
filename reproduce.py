#!/usr/bin/env python3
"""Portable entry points for the archived EGFR analyses; run from any directory."""
import argparse, os, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def run(script,*args):
    env=os.environ.copy();env['MPLBACKEND']='Agg';env['MPLCONFIGDIR']=str(ROOT/'generated/.matplotlib')
    env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
    subprocess.run([sys.executable,str(ROOT/script),*map(str,args)],cwd=ROOT,env=env,check=True)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('task',choices=['verify','figures','train-regression','train-classifier','classifier-validation','classifier-benchmark','benchmark-protein','benchmark-fingerprints'])
    p.add_argument('--seed',type=int,default=104)
    a=p.parse_args(); out=ROOT/'generated';out.mkdir(exist_ok=True)
    data=ROOT/f'data/repeated_scaffold_splits/seed{a.seed}/egfr_pairs_scaffold5_seed{a.seed}.tsv.gz'
    esm=ROOT/'data/features/egfr_esm2_150m_features.npz'
    if a.task=='verify':run('reproducibility/verify.py')
    elif a.task=='train-regression':
        run('code/03_regression_models/train_svr_ensemble.py','--data',data,'--esm',esm,'--seed',a.seed,'--output',out/f'retrained_seed{a.seed}')
    elif a.task=='train-classifier':run('code/04_triple_mutant_classifier/train_release_classifier.py','--output',out/'retrained_classifier.joblib')
    elif a.task=='classifier-validation':run('code/04_triple_mutant_classifier/run_paired_validation_strategy_comparison.py','--legacy-dir',ROOT/'data/classification/archived_fingerprints/ECFP6/mutation','--output',out/'paired_validation_strategy_metrics.csv')
    elif a.task=='classifier-benchmark':run('code/04_triple_mutant_classifier/manuscript_baseline_reproduction.py','--legacy-root',ROOT/'data/classification/archived_fingerprints','--output-dir',out/'classifier_benchmark','--all-fingerprints','--seed',2026,'--repeats',10,'--n-jobs',1)
    elif a.task in ('benchmark-protein','benchmark-fingerprints'):
        script='compare_esm_protein_representations.py' if a.task=='benchmark-protein' else 'compare_regression_representations.py'
        args=['--data',data,'--cdk',ROOT/'data/features/cdk_fingerprints.csv.gz','--seed',a.seed,'--output',out/f'{a.task}_{a.seed}']
        if a.task=='benchmark-protein':args+=['--esm',esm]
        run('code/03_regression_models/'+script,*args)
    elif a.task=='figures':
        # Execute against a disposable output tree so bundled reference data remain immutable.
        import shutil
        scratch=out/'figure_rebuild'
        scratch.mkdir(exist_ok=True)
        shutil.copytree(ROOT/'code',scratch/'code',dirs_exist_ok=True)
        shutil.copytree(ROOT/'results',scratch/'results',dirs_exist_ok=True)
        (scratch/'figures').mkdir(exist_ok=True)
        for script in ['06_figures_and_statistics/build_benchmark_master.py','06_figures_and_statistics/make_figure2_representation_benchmark.py','06_figures_and_statistics/create_figure3_mutant_classifier_benchmark.py','06_figures_and_statistics/create_supplementary_ecfp6_mutant_secondary_metrics.py','06_figures_and_statistics/create_supplementary_source_control_figure.py','06_figures_and_statistics/draw_consensus_16_structures.py','06_figures_and_statistics/create_sfig7_evidence_prioritization.py']:
            run(scratch/'code'/script)
if __name__=='__main__':main()
