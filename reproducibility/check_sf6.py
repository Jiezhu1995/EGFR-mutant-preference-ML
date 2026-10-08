"""Verify archived SF6 provenance, observation units and current statistics."""
from pathlib import Path
import hashlib, json
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]

def main():
    src = ROOT/'results/figure3_classifier/source_control_fold_metrics.csv'
    expected = 'f5a003a49ba16368cf0ac82fabf2271776e732a837a5d84e1142a3d3c70d6b80'
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    assert digest == expected, 'Historical result file differs from verified source'
    meta = json.loads((ROOT/'results/figure3_classifier/run_metadata.json').read_text())
    assert meta['n_compounds'] == 2099 and meta['n_features'] == 1024
    assert meta['y_randomizations_per_null_design'] == 20
    df = pd.read_csv(src)
    observed = df[df.experiment.eq('source_balanced_scaffold_cv')]
    assert set(observed.seed) == set(meta['seeds'])
    assert observed.groupby('seed').size().eq(5).all()
    null = df[df.experiment.eq('source_balanced_y_randomization')]
    assert len(null) == 20
    rows=[]
    for metric in ['roc_auc','mcc']:
        values = observed.groupby('seed')[metric].mean()
        test = mannwhitneyu(values, null[metric], alternative='two-sided', method='exact')
        rows.append({'metric':metric,'raw_p':float(test.pvalue)})
    adjusted=multipletests([r['raw_p'] for r in rows],method='holm')[1]
    # The reference statistics are stored independently of rendered figures.
    expected_adjusted={'roc_auc':1.3313356641692475e-07,
                       'mcc':1.3313356641692475e-07}
    for row,p in zip(rows,adjusted):
        row['holm_p']=float(p)
        assert abs(expected_adjusted[row['metric']]-p)<1e-15
    report={'status':'passed','historical_results_sha256':digest,'metadata':meta,
            'observed_seeds':10,'folds_per_seed':5,'null_permutations':20,'statistics':rows,
            'scope':'Archived provenance and statistics; not historical training reproduction',
            'not_recorded_in_historical_metadata':['execution_host','CSC_job_id','class_weight']}
    out=ROOT/'generated';out.mkdir(exist_ok=True)
    (out/'sf6_consistency.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
