"""Check release integrity, training data, frozen splits and model inference."""
from pathlib import Path
import sys,json,hashlib,ast,importlib.metadata as md
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[*[str(p) for p in (ROOT/'code').glob('[0-9][0-9]_*') if p.is_dir()]]
def main():
    import numpy as np,pandas as pd,joblib
    from screen_taosu_seed104_ensemble import contexts,predict
    from compare_regression_representations import compound_features,mols
    from run_paired_validation_strategy_comparison import load_data
    report={'checks':{},'environment':{}}
    for name in ['numpy','pandas','scipy','scikit-learn','rdkit','joblib','matplotlib','xgboost']:
        try:report['environment'][name]=md.version(name)
        except md.PackageNotFoundError:report['environment'][name]='not installed'
    for p in ROOT.joinpath('code').rglob('*.py'):ast.parse(p.read_text(),filename=str(p.relative_to(ROOT)))
    report['checks']['python_syntax']='passed'
    integrity=ROOT/'reproducibility/SHA256SUMS.txt'
    if integrity.exists():
        for line in integrity.read_text().splitlines():
            expected,rel=line.split('  ',1);p=ROOT/rel
            assert p.is_file(),rel
            assert hashlib.sha256(p.read_bytes()).hexdigest()==expected,rel
        report['checks']['sha256']='passed'
    rows=[]; discrepancies=[]
    paths=list((ROOT/'data/repeated_scaffold_splits').glob('seed*/egfr_pairs*.gz'))
    assert len(paths)==10,len(paths)
    for p in paths:
        df=pd.read_csv(p,sep='\t');assert len(df)==1223
        assert df.inchikey_connectivity.nunique()==810
        assert df.v2_variant_token.nunique()==30
        error=df.delta_pactivity-(df.p_activity_mut-df.p_activity_wt)
        for i in np.flatnonzero(error.abs().to_numpy()>1e-6):
            discrepancies.append({'file':str(p.relative_to(ROOT)),'row_index':int(i),'inchikey_connectivity':df.inchikey_connectivity.iloc[i],'stored_delta':float(df.delta_pactivity.iloc[i]),'difference_of_stored_activities':float(df.p_activity_mut.iloc[i]-df.p_activity_wt.iloc[i]),'difference':float(error.iloc[i])})
        group='scaffold_group' if 'scaffold_group' in df else None
        for fold in range(5):
            col=f'unseen_compound_fold{fold}'
            assert df.groupby('inchikey_connectivity')[col].nunique().max()==1
            if group:assert df.groupby(group)[col].nunique().max()==1
        rows.append({'file':str(p.relative_to(ROOT)),'rows':len(df),'compound_groups':810,'variant_settings':30,'scaffold_column':group})
    report['checks']['frozen_regression_data']=rows
    report['checks']['delta_aggregation_discrepancies']=discrepancies
    x,y,g=load_data(ROOT/'data/classification/archived_fingerprints/ECFP6/mutation')
    model=joblib.load(ROOT/'models/classification/mutant_ECFP6_SVM_full_data.joblib')
    scores=model.decision_function(x[:8]);assert np.isfinite(scores).all()
    report['checks']['classifier']={'rows':len(y),'active':int(y.sum()),'negative_labels':int((y==0).sum()),'features':x.shape[1],'sample_scores':scores.tolist(),'parameters':model.get_params()}
    bundle=joblib.load(ROOT/'models/regression/seed104_three_task_fivefold_ensemble.joblib')
    assert bundle['seed']==104 and len(bundle['fold_models'])==5
    context,quality=contexts(bundle,ROOT/'data/features/egfr_esm2_150m_features.npz')
    # General inference smoke test using records from the packaged dataset.
    fp=compound_features(mols(df.standardized_smiles.iloc[:8]),'ecfp6').astype(np.float32)
    values=predict(fp,context,workers=1)
    report['checks']['regression_inference']={}
    for target in ['mut','wt','delta']:
        result=np.stack([values[(f,target)] for f in range(5)]).mean(axis=0)
        assert np.isfinite(result).all()
        report['checks']['regression_inference'][target]=result.tolist()
    report['checks']['esm_quality']=quality
    report['status']='passed_with_data_definition_warning' if discrepancies else 'passed'
    dest=ROOT/'generated';dest.mkdir(exist_ok=True)
    (dest/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
