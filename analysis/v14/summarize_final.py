"""Summarize frozen predictions; conditional paired component bootstrap, post-test."""
import argparse,csv,json
from pathlib import Path
import numpy as np
def run(root,out):
    m=json.loads((root/'results/final/metrics.json').read_text()); rows=[]
    for name in ['majority','structured_logistic','text_tfidf_logistic','feature_fusion_logistic','text','concat','gated']:
        keys=[name] if name in m else [name+'_seed'+str(s) for s in (62,63,64)]
        row={'model':name,'runs':len(keys)}
        for metric in ['macro_f1','accuracy']:
            v=[m[k][metric] for k in keys];row[metric+'_mean']=float(np.mean(v));row[metric+'_sample_sd']=float(np.std(v,ddof=1)) if len(v)>1 else None
        rows.append(row)
    out.mkdir(exist_ok=True,parents=True)
    with (out/'final_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (root/'results/development_baselines/split_manifest.csv').open() as f: components={r['incident_id']:r['duplicate_component'] for r in csv.DictReader(f)}
    codes=[];ref=None
    for model in ['text','gated']:
        for seed in (62,63,64):
            with (root/f'results/final/{model}_seed{seed}_predictions.csv').open() as f: p=list(csv.DictReader(f))
            ids=[(r['incident_id'],r['true']) for r in p]
            if ref is None:ref=ids
            assert ids==ref
            codes.append(np.array([int(r['true'])*3+int(r['predicted']) for r in p]))
    _,inv=np.unique([components[g] for g,y in ref],return_inverse=True);n=int(inv.max())+1
    def score(code,w):
        c=np.bincount(code,weights=w,minlength=9).reshape(3,3);d=c.sum(0)+c.sum(1)
        return np.divide(2*np.diag(c),d,out=np.zeros(3),where=d!=0).mean()
    rng=np.random.default_rng(62);diff=[]
    for _ in range(2000):
        w=np.bincount(rng.integers(0,n,size=n),minlength=n)[inv]
        v=[score(c,w) for c in codes];diff.append(float(np.mean(v[3:])-np.mean(v[:3])))
    result={'resamples':2000,'rng_seed':62,'components':n,'cases':len(ref),'contrast':'mean gated F1 across three fixed seeds minus mean text F1 across three fixed seeds','point_difference':rows[-1]['macro_f1_mean']-rows[-3]['macro_f1_mean'],'percentile_95_interval':np.quantile(diff,[.025,.975]).tolist(),'scope':'Post-test analysis conditional on fitted models. Does not quantify training uncertainty or establish equivalence.'}
    (out/'bootstrap.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.repo,a.out)
