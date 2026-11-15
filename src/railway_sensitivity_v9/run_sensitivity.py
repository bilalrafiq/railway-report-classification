"""Broad outcome-related sentence deletion sensitivity; not leakage removal certification."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'2')
import argparse,csv,hashlib,json,re,warnings,platform
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import sklearn,joblib
from scipy.sparse import csr_matrix,hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix

RULES={
 'derailment':r'\bd\s*e\s*r\s*a\s*i\s*l(?:\s*(?:e\s*d|i\s*n\s*g|m\s*e\s*n\s*t(?:s)?|s))?\b',
 'casualties':r'\b(?:injur\w*|fatal\w*|kill(?:ed|ing)?|death\w*|casualt\w*)\b|\bPI[\x27\u2019]?S\b',
 'damage':r'\bdamag\s*e\w*\b|\bdestroy\w*\b|\bloss(?:es)?\b|\bcost(?:s)?\b',
 'money':r'[$¥￥]|\b(?:USD|CNY|RMB|DOLLARS?)\b',
 'physical_outcome':r'\boverturn\w*\b|\bturn(?:ed|ing)?\s+over\b|\bwheels?\s+on\s+(?:the\s+)?ground\b',
}
COMPILED={k:re.compile(v,re.I) for k,v in RULES.items()}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj):p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def transform(text,gid):
    # Conservative punctuation segmentation; do not split decimal numbers.
    spans=[(m.start(),m.end(),m.group()) for m in re.finditer(r'.+?(?:[.!?](?=\s|$)|$)',text,re.S)]
    hits=[]
    for i,(start,end,s) in enumerate(spans):
        reasons=[k for k,p in COMPILED.items() if p.search(s)]
        if reasons:hits.append({'sentence_index':i,'start':start,'end':end,'text':s,'rules':reasons})
    target={h['sentence_index'] for h in hits}
    order=sorted(range(len(spans)),key=lambda i:hashlib.sha256(('control62|'+gid+'|'+str(i)).encode()).hexdigest())
    control=set(order[:len(target)])
    def kept(excluded):return ' '.join(s.strip() for i,(_,_,s) in enumerate(spans) if i not in excluded).strip()
    return {'full':text,'targeted':kept(target),'random_control':kept(control)}, {
        'incident_id':gid,'original_text':text,'sentence_count':len(spans),'removed_sentences':hits,
        'random_removed_sentence_indices':sorted(control),
        'targeted_empty':not bool(kept(target)), 'random_empty':not bool(kept(control)),
        'targeted_removed_characters':sum(end-start for i,(start,end,s) in enumerate(spans) if i in target),
        'random_removed_characters':sum(end-start for i,(start,end,s) in enumerate(spans) if i in control)}

def run(a):
    if a.out.exists():raise ValueError('Use a new output directory')
    protocol=json.loads((a.source/'protocol.json').read_text())
    complete=json.loads((a.source/'completion.json').read_text())
    assert complete['test_evaluated'] is False
    assert sha(a.source/'protocol.json')==complete['protocol_sha256']
    assert sha(a.source/'metrics.json')==complete['metrics_sha256']
    records={}
    for split in ('train','validation'):
        records[split]=[json.loads(l) for l in (a.source/(split+'.jsonl')).read_text().splitlines() if l.strip()]
        ids=[r['incident_id'] for r in records[split]]
        assert len(ids)==len(set(ids))==protocol['selected_counts'][split]
        assert all((int(r['date'][:4])<=2017 if split=='train' else 2018<=int(r['date'][:4])<=2020) for r in records[split])
    assert not {r['incident_id'] for r in records['train']}&{r['incident_id'] for r in records['validation']}
    a.out.mkdir(parents=True)
    transformed={};summary={}
    for split,data in records.items():
        transformed[split]={k:[] for k in ('full','targeted','random_control')};stats=Counter()
        with (a.out/(split+'_changes.jsonl')).open('w',encoding='utf-8') as f:
            for r in data:
                variants,change=transform(r['text'],r['incident_id'])
                for k,v in variants.items():transformed[split][k].append(v)
                change['transformed_text']=variants
                f.write(json.dumps(change,ensure_ascii=False)+'\n')
                stats['records']+=1;stats['changed']+=bool(change['removed_sentences'])
                stats['targeted_empty']+=change['targeted_empty'];stats['random_empty']+=change['random_empty']
                stats['original_characters']+=len(r['text'])
                for k in ('targeted_removed_characters','random_removed_characters'):stats[k]+=change[k]
        summary[split]=dict(stats)
    save(a.out/'transformation_summary.json',summary)
    save(a.out/'protocol.json',{'status':'EXPLORATORY_DEVELOPMENT_ONLY','rules':RULES,
        'conditions':['full','targeted','random_control'],'seed':62,'test_evaluated':False,
        'hashes':{n:sha(a.source/n) for n in ['train.jsonl','validation.jsonl','protocol.json','structured_logistic.joblib']},
        'script_sha256':sha(Path(__file__)),'versions':{'python':platform.python_version(),'sklearn':sklearn.__version__},
        'limitations':['Broad sentence deletion removes context, includes negation and may match derail devices',
        'Unmatched outcome disclosures remain; not a causal leakage estimate or prospective task',
        'Control matches removed sentence count per record, not character count or semantics',
        'Single-sentence narratives may become empty in both conditions; retained, never dropped',
        'Rules informed by training-only review; validation used previously for model development',
        'Provisional composite labels and selected-report policy remain unresolved']})
    print(json.dumps(summary,indent=2),flush=True)
    if a.preview_only:return
    pre=joblib.load(a.source/'structured_logistic.joblib').named_steps['preprocess']
    xs={};ys={}
    for split,data in records.items():
        frame=pd.DataFrame([r['features'] for r in data])
        for k in protocol['numeric_features']:frame[k]=pd.to_numeric(frame[k],errors='coerce')
        xs[split]=csr_matrix(pre.transform(frame));ys[split]=np.asarray([r['label'] for r in data])
    results={}
    for condition in ('full','targeted','random_control'):
        tf=TfidfVectorizer(max_features=20000,min_df=2,ngram_range=(1,2),sublinear_tf=True)
        try:
            xt=tf.fit_transform(transformed['train'][condition]);xv=tf.transform(transformed['validation'][condition])
        except ValueError as error:
            raise RuntimeError('No usable training vocabulary in '+condition+'; review transformation') from error
        for mode in ('text','feature_fusion'):
            train_x=xt if mode=='text' else hstack([xs['train'],xt],format='csr')
            val_x=xv if mode=='text' else hstack([xs['validation'],xv],format='csr')
            model=LogisticRegression(C=1.0,max_iter=1000,solver='lbfgs',random_state=62)
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always');model.fit(train_x,ys['train'])
            pred=model.predict(val_x);prob=model.predict_proba(val_x);assert list(model.classes_)==[0,1,2]
            name=condition+'_'+mode
            results[name]={'accuracy':float(accuracy_score(ys['validation'],pred)),
                'macro_f1':float(f1_score(ys['validation'],pred,labels=[0,1,2],average='macro',zero_division=0)),
                'classification_report':classification_report(ys['validation'],pred,labels=[0,1,2],output_dict=True,zero_division=0),
                'confusion_matrix':confusion_matrix(ys['validation'],pred,labels=[0,1,2]).tolist(),
                'warnings':[str(w.message) for w in captured]}
            with (a.out/(name+'_predictions.csv')).open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['incident_id','true','predicted','p0','p1','p2'])
                for r,y,p,pv in zip(records['validation'],ys['validation'],pred,prob):w.writerow([r['incident_id'],int(y),int(p)]+pv.tolist())
            joblib.dump({'vectorizer':tf,'structured_preprocessor':pre if mode=='feature_fusion' else None,'classifier':model},a.out/(name+'.joblib'))
            save(a.out/'metrics.json',results)
            print(name+': macro-F1={:.4f}, accuracy={:.4f}'.format(results[name]['macro_f1'],results[name]['accuracy']),flush=True)
    save(a.out/'completion.json',{'status':'SENSITIVITY_COMPLETE','test_evaluated':False,'metrics_sha256':sha(a.out/'metrics.json')})
    print('SENSITIVITY_COMPLETE: '+str(a.out.resolve()),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--preview-only',action='store_true');run(p.parse_args())
