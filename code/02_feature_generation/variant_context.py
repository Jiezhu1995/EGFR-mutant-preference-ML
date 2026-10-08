"""Load the normalized triple-mutant embedding and reconstruction-quality flag."""
import numpy as np

def variant_esm(path):
    cache=np.load(path,allow_pickle=True)
    index=[str(x) for x in cache['variant_tokens']].index('SUB:T790M;SUB:C797S;SUB:L858R')
    value=np.concatenate([cache['difference_global'][index].astype(np.float32),cache['difference_local'][index].astype(np.float32)])
    norm=np.linalg.norm(value)
    if norm:value/=norm
    quality=np.zeros(3,dtype=np.float32)
    label=str(cache['sequence_quality'][index])
    quality[{'exact':0,'surrogate':1,'unresolved':2}[label]]=1
    return value,quality,label
