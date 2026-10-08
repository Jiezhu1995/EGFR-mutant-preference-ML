"""Refit the archived 1,024-bit R/CDK mutant classifier on all 2,099 records."""

# Resolve shared modules across the six workflow stages.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(p) for p in _Path(__file__).resolve().parents[1].glob("[0-9][0-9]_*") if p.is_dir()]
from pathlib import Path
import argparse,json,joblib,numpy as np
from sklearn.svm import SVC
from run_paired_validation_strategy_comparison import load_data
ROOT=Path(__file__).resolve().parents[2]
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    X,y,groups=load_data(ROOT/'data/classification/archived_fingerprints/ECFP6/mutation')
    model=SVC(C=10,kernel='rbf',gamma='scale',class_weight='balanced',cache_size=1000,random_state=2026)
    model.fit(X.astype(np.float32),y)
    a.output.parent.mkdir(parents=True,exist_ok=True);joblib.dump(model,a.output)
    a.output.with_suffix('.json').write_text(json.dumps({'rows':len(y),'features':X.shape[1],'active':int(y.sum()),'parameters':model.get_params(),'fingerprint':'archived R/CDK ECFP6; NOT RDKit Morgan','positive_rule':'decision_function >= 0'},indent=2)+'\n')
