"""Record refit checks, inventory files and build a portable release ZIP."""
from pathlib import Path
import hashlib,json,csv,re,zipfile,sys,shutil
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[*[str(p) for p in (ROOT/'code').glob('[0-9][0-9]_*') if p.is_dir()]]
def compare_refits():
    import joblib,numpy as np,pandas as pd
    from run_paired_validation_strategy_comparison import load_data
    from screen_taosu_seed104_ensemble import contexts,predict
    from compare_regression_representations import compound_features,mols
    report={}
    X,y,_=load_data(ROOT/'data/classification/archived_fingerprints/ECFP6/mutation')
    old=joblib.load(ROOT/'models/classification/mutant_ECFP6_SVM_full_data.joblib')
    new=joblib.load(ROOT/'generated/retrained_classifier.joblib')
    a=old.decision_function(X);b=new.decision_function(X)
    report['classifier_refit']={'n_predictions':len(X),'max_absolute_score_difference':float(np.max(np.abs(a-b))),'identical_positive_labels':bool(np.array_equal(a>=0,b>=0))}
    frame=pd.read_csv(ROOT/'data/repeated_scaffold_splits/seed104/egfr_pairs_scaffold5_seed104.tsv.gz',sep='\t')
    fp=compound_features(mols(frame.standardized_smiles),'ecfp6').astype(np.float32)
    predictions=[]
    for path in [ROOT/'models/regression/seed104_three_task_fivefold_ensemble.joblib',ROOT/'generated/retrained_seed104/three_svr_fivefold_ensemble.joblib']:
        bundle=joblib.load(path);context,_=contexts(bundle,ROOT/'data/features/egfr_esm2_150m_features.npz')
        raw=predict(fp,context,workers=1)
        predictions.append({t:np.stack([raw[(f,t)] for f in range(5)]).mean(axis=0) for t in ['mut','wt','delta']})
    report['regression_refit']={t:{'n_predictions':len(fp),'max_absolute_pactivity_difference':float(np.max(np.abs(predictions[0][t]-predictions[1][t]))),'mean_absolute_pactivity_difference':float(np.mean(np.abs(predictions[0][t]-predictions[1][t])))} for t in ['mut','wt','delta']}
    report['regression_refit']['note']='All 1,223 structures scored at the fixed triple-mutant/IC50 context to test numerical reproduction, not an accuracy evaluation. Archived estimators identify sklearn 1.9.0; local refit uses 1.7.1. Packaged original models are preserved.'
    (ROOT/'reproducibility/refit_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    shutil.copy2(ROOT/'generated/verification.json',ROOT/'reproducibility/verification_snapshot.json')
    return report

def main():
    report=json.loads((ROOT/'reproducibility/refit_comparison.json').read_text()) if '--package-only' in sys.argv else compare_refits()
    # Scan concrete credential patterns without copying matched values into reports.
    patterns=[r'-----BEGIN (?:OPENSSH|RSA|EC) PRIVATE KEY-----',r'gh[pousr]_[A-Za-z0-9]{25,}',r'AKIA[A-Z0-9]{16}',r'(?i)(?:password|passwd|api_key|access_token)\s*[:=]\s*[\"\x27][^\"\x27]{6,}[\"\x27]']
    hits=[];absolute=[]
    files=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name not in {'.DS_Store','.Rhistory'} and not any(x in {'generated','__pycache__','.git'} for x in p.relative_to(ROOT).parts)]
    for p in files:
        if p.suffix in {'.py','.R','.sh','.json','.md','.yaml','.yml','.txt'}:
            text=p.read_text(errors='replace')
            if any(re.search(pattern,text) for pattern in patterns) and p.name!='finalize.py':hits.append(str(p.relative_to(ROOT)))
            if '/Users/' in text or '/root/' in text or 'C:\\' in text:absolute.append(str(p.relative_to(ROOT)))
    (ROOT/'reproducibility/release_scan.json').write_text(json.dumps({'credential_pattern_files':hits,'absolute_path_reference_files':absolute,'note':'Path references in archived manifests document original runs; portable entry points take local paths. Pattern scan is not a legal or exhaustive security review.'},indent=2)+'\n')
    if hits:raise RuntimeError('Review credential-pattern files before packaging: '+str(hits))
    inventory=[]
    for p in sorted(ROOT.rglob('*')):
        rel=p.relative_to(ROOT)
        if not p.is_file() or any(x in {'generated','__pycache__','.git'} for x in rel.parts) or p.name in {'SHA256SUMS.txt','FILE_INVENTORY.csv','.DS_Store','.Rhistory'}:continue
        inventory.append((str(rel),p.stat().st_size,hashlib.sha256(p.read_bytes()).hexdigest()))
    with (ROOT/'reproducibility/FILE_INVENTORY.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['path','bytes','sha256']);w.writerows(inventory)
    inv=ROOT/'reproducibility/FILE_INVENTORY.csv'
    lines=[f'{h}  {p}' for p,_,h in inventory]+[f'{hashlib.sha256(inv.read_bytes()).hexdigest()}  reproducibility/FILE_INVENTORY.csv']
    (ROOT/'reproducibility/SHA256SUMS.txt').write_text('\n'.join(lines)+'\n')
    dest=ROOT.parent/(ROOT.name+'.zip')
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel,_,_ in inventory:z.write(ROOT/rel,ROOT.name+'/'+rel)
        for name in ['FILE_INVENTORY.csv','SHA256SUMS.txt']:z.write(ROOT/'reproducibility'/name,ROOT.name+'/reproducibility/'+name)
    with zipfile.ZipFile(dest) as z:assert z.testzip() is None
    print(json.dumps({'files':len(inventory)+2,'zip_bytes':dest.stat().st_size,'zip':str(dest),'refit_comparison':report},indent=2))
if __name__=='__main__':main()
