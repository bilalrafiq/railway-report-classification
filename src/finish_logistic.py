"""Finish the development logistic fit; never reads the reserved test set."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ.setdefault(k,'2')
import argparse, csv, hashlib, json, warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix,hstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('Use a fresh output directory')
    receipt=json.loads((a.source/'completion.json').read_text())
    for name,key in [('train.jsonl','train_sha256'),('validation.jsonl','validation_sha256'),('protocol.json','protocol_sha256')]:
        if sha(a.source/name)!=receipt[key]:raise ValueError('Changed input: '+name)
    if receipt['test_evaluated'] is not False:raise ValueError('Unexpected source receipt')
    datasets=[]
    for name in ['train','validation']:
        with (a.source/(name+'.jsonl')).open() as f:datasets.append([json.loads(l) for l in f if l.strip()])
    train,val=datasets
    bundle=joblib.load(a.source/'feature_fusion_bundle.joblib')
    matrices=[]
    for rows in datasets:
        frame=pd.DataFrame([r['features'] for r in rows])
        for k in ['TEMP','TRNSPD','TONS','CARS']:frame[k]=pd.to_numeric(frame[k],errors='coerce')
        matrices.append(hstack([csr_matrix(bundle['preprocess'].transform(frame)),bundle['tfidf'].transform([r['text'] for r in rows])],format='csr'))
    y=np.array([r['label'] for r in train]);yv=np.array([r['label'] for r in val])
    model=bundle['classifier'];before=model.get_params().copy();model.set_params(max_iter=5000,warm_start=True)
    print('Continuing the existing logistic fit on training data only',flush=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');model.fit(matrices[0],y)
    if list(model.classes_)!=[0,1,2]:raise ValueError('Unexpected class order')
    prob=model.predict_proba(matrices[1]);pred=prob.argmax(1)
    converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
    a.out.mkdir(parents=True)
    result={'status':'CONVERGENCE_RETRY_COMPLETE' if converged else 'CONVERGENCE_UNRESOLVED','test_evaluated':False,'macro_f1':float(f1_score(yv,pred,average='macro')),'accuracy':float(accuracy_score(yv,pred)),'confusion_matrix':confusion_matrix(yv,pred).tolist(),'classification_report':classification_report(yv,pred,output_dict=True,zero_division=0),'iterations_this_fit':model.n_iter_.tolist(),'warnings':[str(w.message) for w in caught],'source_hashes':{n:sha(a.source/n) for n in ['train.jsonl','validation.jsonl','protocol.json','feature_fusion_bundle.joblib']},'original_parameters':before,'retry_parameters':model.get_params(),'script_sha256':sha(Path(__file__))}
    (a.out/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    with (a.out/'predictions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['incident_id','true','predicted','p0','p1','p2'])
        w.writerows([r['incident_id'],int(t),int(q)]+pr.tolist() for r,t,q,pr in zip(val,yv,pred,prob))
    joblib.dump(bundle,a.out/'feature_fusion_bundle.joblib')
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
