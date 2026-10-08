"""Train three target-specific SVRs per frozen fold, without compound-specific tests."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from compare_esm_protein_representations import endpoint_features,load_esm
from compare_regression_representations import compound_features,make_model,mols

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','esm','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--seed',type=int,default=104)
    p.add_argument('--esm-components',type=int,default=16)
    a=p.parse_args()
    frame=pd.read_csv(a.data,sep='\t',low_memory=False)
    fp=compound_features(mols(frame.standardized_smiles),'ecfp6').astype(np.float32)
    endpoint=endpoint_features(frame);esm,quality=load_esm(a.esm,frame)
    bundles=[]
    for fold in range(5):
        train=np.flatnonzero(frame[f'unseen_compound_fold{fold}'].eq('train'))
        pca=PCA(n_components=min(a.esm_components,len(train)-1,esm.shape[1]),random_state=a.seed+fold)
        transformed=pca.fit_transform(esm[train]);scaler=StandardScaler()
        transformed=scaler.fit_transform(transformed).astype(np.float32)
        features=np.concatenate([fp[train],endpoint[train],transformed,quality[train]],axis=1)
        wt_features=np.concatenate([fp[train],endpoint[train]],axis=1)
        models={}
        for name in ['p_activity_mut','delta_pactivity']:
            models[name]=make_model('svr',a.seed);models[name].fit(features,frame[name].to_numpy()[train])
        wt=make_model('svr',a.seed);wt.fit(wt_features,frame.p_activity_wt.to_numpy()[train])
        bundles.append(dict(fold=fold,esm_pca=pca,esm_scaler=scaler,mutant_model=models['p_activity_mut'],delta_model=models['delta_pactivity'],wt_model=wt,n_training_rows=len(train)))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    target=out/'three_svr_fivefold_ensemble.joblib'
    joblib.dump(dict(fold_models=bundles,seed=a.seed,compound_representation='ECFP6 radius=3 2048 bits',mutation_representation='ESM difference PCA(16) + quality one-hot',endpoint_order=['IC50','Ki','Kd']),target,compress=3)
    metadata=dict(seed=a.seed,folds=5,models_per_fold=3,training_rows=[x['n_training_rows'] for x in bundles],data_sha256=hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),ensemble_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    (out/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(metadata,indent=2))

if __name__=='__main__':main()
