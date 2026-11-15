"""Provisional development baselines. No final test evaluation or neural training."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'2')
import argparse, csv, hashlib, json, math, platform, re, warnings
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import joblib
from scipy.sparse import csr_matrix,hstack
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM=['TEMP','TRNSPD','TONS','CARS']
CAT=['STATE','VISIBLTY','WEATHER','TRKCLAS','TYPTRK','PASSTRN','TRNDIR']
RAW_HASH='9271f04027d907ae9c47d1de2d38a1b1ddf4eb64fcae177b9f62faa21be02e19'
COHORT_HASH='0cac4a6348008bc10685d51c96cc48fc7046556af85762d6ed35ac2941ed3668'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def number(value):
    try:
        x=float(str(value).replace(',',''))
        return x if math.isfinite(x) else None
    except (ValueError,TypeError):return None

def text_of(row):
    text=' '.join(row.get('NARR'+str(i),'') for i in range(1,16))
    return ' '.join(x for x in text.split() if x.lower() not in ('nan','none'))

def normalized(text):return ' '.join(re.findall(r'\w+',text.casefold()))

def run(a):
    if a.train_cap<3 or a.val_cap<3:raise ValueError('Caps must be >=3')
    if a.out.exists():raise ValueError('Output already exists; choose a fresh --out')
    if sha(a.raw)!=RAW_HASH or sha(a.cohort)!=COHORT_HASH:
        raise ValueError('Input hash mismatch; use the verified raw CSV and v6 cohort')
    meta={}
    with a.cohort.open(newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['status']=='stable_label_under_overlap_bounds':meta[r['inc3_group_id']]=r
    parent={g:g for g in meta}
    def find(g):
        while parent[g]!=g:
            parent[g]=parent[parent[g]];g=parent[g]
        return g
    def union(x,y):
        x,y=find(x),find(y)
        if x!=y:parent[max(x,y)]=min(x,y)
    with a.duplicates.open(encoding='utf-8') as f:
        for line in f:
            group=json.loads(line)['group_ids']
            if not all(g in meta for g in group):raise ValueError('Duplicate file/cohort mismatch')
            for g in group[1:]:union(group[0],g)
    # Also detect all nonempty selected narratives, including short strings.
    source_to_group={int(r['reference_narrative_source_row']):g for g,r in meta.items()}
    if len(source_to_group)!=len(meta):raise ValueError('Nonunique source rows')
    records={};seen_text={}
    with a.raw.open(encoding='utf-8-sig',newline='') as f:
        for rownum,row in enumerate(csv.DictReader(f),1):
            if rownum not in source_to_group:continue
            g=source_to_group[rownum];r=meta[g];text=text_of(row)
            if not text:raise ValueError('Empty selected text')
            key=normalized(text)
            if key:
                if key in seen_text:union(g,seen_text[key])
                else:seen_text[key]=g
            year=int(r['incident_date'][:4])
            if year!=int(row['YEAR4']):raise ValueError('Source date mismatch')
            split='train' if year<=2017 else ('validation' if year<=2020 else 'reserved_future')
            features={k:number(row.get(k)) for k in NUM}
            features.update({k:row.get(k,'').strip() or '<MISSING>' for k in CAT})
            records[g]={'incident_id':g,'source_row':rownum,'date':r['incident_date'],
                'label':int(r['candidate_label']),'text':text,'features':features,'initial_split':split}
    if len(records)!=len(meta):raise ValueError('Missing source rows')
    priority={'train':0,'validation':1,'reserved_future':2}
    component_latest={}
    for g,r in records.items():
        c=find(g);component_latest[c]=max(component_latest.get(c,0),priority[r['initial_split']])
    pools={'train':[],'validation':[]};dispositions={}
    for g,r in records.items():
        split=r['initial_split']
        if priority[split]<component_latest[find(g)]:status='excluded_earlier_duplicate'
        else:status=split
        dispositions[g]=status
        if status in pools:pools[status].append(g)
    def sample(ids,cap):
        return sorted(ids,key=lambda g:hashlib.sha256(('62|'+g).encode()).hexdigest())[:cap]
    selected={k:sample(v,a.train_cap if k=='train' else a.val_cap) for k,v in pools.items()}
    train=[records[g] for g in selected['train']];val=[records[g] for g in selected['validation']]
    if not train or not val:raise ValueError('Empty development partition')
    if {find(r['incident_id']) for r in train}&{find(r['incident_id']) for r in val}:
        raise ValueError('Duplicate component crosses development partitions')
    for data in (train,val):
        if {r['label'] for r in data}!={0,1,2}:raise ValueError('Missing class; increase cap before training')
    a.out.mkdir(parents=True)
    with (a.out/'split_manifest.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['incident_id','date','disposition','selected_for_development','duplicate_component'])
        chosen=set(selected['train']+selected['validation'])
        for g in sorted(records):w.writerow([g,records[g]['date'],dispositions[g],int(g in chosen),find(g)])
    for name,data in [('train',train),('validation',val)]:
        with (a.out/(name+'.jsonl')).open('w',encoding='utf-8') as f:
            for r in data:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    protocol={'status':'PROVISIONAL_DEVELOPMENT_ONLY','seed':62,
        'target':'v6 adapted nominal-CNY composite; not official severity grades',
        'train_years':'1982-2017','validation_years':'2018-2020','reserved_years':'2021-2025; no evaluation',
        'source_rule':'v6 longest cleaned narrative source row; all structured values from that same report',
        'duplicate_rule':'v7 all-report exact normalized links plus all nonempty selected texts; transitive components; exclude earlier partitions, never move future records backwards',
        'numeric_features':NUM,'categorical_features':CAT,
        'selection':'label-independent SHA256(seed|incident_id) ranking within each partition',
        'caps':{'train':a.train_cap,'validation':a.val_cap},
        'dispositions':dict(Counter(dispositions.values())),
        'eligible_pool_counts':{k:len(v) for k,v in pools.items()},
        'selected_counts':{'train':len(train),'validation':len(val)},
        'selected_class_counts':{k:dict(Counter(records[g]['label'] for g in ids)) for k,ids in selected.items()},
        'hashes':{'raw':sha(a.raw),'cohort':sha(a.cohort),'duplicates':sha(a.duplicates),'script':sha(Path(__file__))},
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__},
        'limitations':['Development target and cohort are provisional; exclusions may bias population',
            'Narratives retain outcomes; this is retrospective report classification',
            'Selected report is not a verified aggregate of every involved train',
            'Unknown incident links and near-duplicate narratives may remain',
            'Old zero-filled raw codes are not universally reinterpreted as missing',
            'No hyperparameter search, statistical comparison or final test evaluation']}
    write_json(a.out/'protocol.json',protocol)
    xtrain=pd.DataFrame([r['features'] for r in train]);xval=pd.DataFrame([r['features'] for r in val])
    for frame in (xtrain,xval):
        for k in NUM:frame[k]=pd.to_numeric(frame[k],errors='coerce')
    yt=np.array([r['label'] for r in train]);yv=np.array([r['label'] for r in val])
    numeric=Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler(with_mean=False))])
    pre=ColumnTransformer([('numeric',numeric,NUM),('category',OneHotEncoder(handle_unknown='ignore'),CAT)])
    models=[('majority',DummyClassifier(strategy='most_frequent'),np.zeros((len(train),1)),np.zeros((len(val),1))),
        ('structured_logistic',Pipeline([('preprocess',pre),('classifier',LogisticRegression(C=1.0,max_iter=1000,solver='lbfgs',random_state=62))]),xtrain,xval),
        ('text_tfidf_logistic',Pipeline([('tfidf',TfidfVectorizer(max_features=20000,min_df=2,ngram_range=(1,2),sublinear_tf=True)),
        ('classifier',LogisticRegression(C=1.0,max_iter=1000,solver='lbfgs',random_state=62))]),[r['text'] for r in train],[r['text'] for r in val])]
    results={}
    fitted={}
    for name,model,xt,xv in models:
        print('Fitting '+name,flush=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');model.fit(xt,yt)
        pred=model.predict(xv);prob=model.predict_proba(xv)
        if list(model.classes_)!=[0,1,2]:raise ValueError('Unexpected probability column order')
        results[name]={'accuracy':float(accuracy_score(yv,pred)),
            'macro_f1':float(f1_score(yv,pred,labels=[0,1,2],average='macro',zero_division=0)),
            'report':classification_report(yv,pred,labels=[0,1,2],output_dict=True,zero_division=0),
            'confusion_matrix':confusion_matrix(yv,pred,labels=[0,1,2]).tolist(),
            'warnings':[str(w.message) for w in caught]}
        with (a.out/(name+'_predictions.csv')).open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['incident_id','true','predicted','p0','p1','p2'])
            for r,y,predicted,p in zip(val,yv,pred,prob):w.writerow([r['incident_id'],int(y),int(predicted)]+p.tolist())
        joblib.dump(model,a.out/(name+'.joblib'))
        fitted[name]=model
        if name=='text_tfidf_logistic':
            fitted_pre=fitted['structured_logistic'].named_steps['preprocess']
            fitted_tf=model.named_steps['tfidf']
            joint_train=hstack([csr_matrix(fitted_pre.transform(xtrain)),fitted_tf.transform([r['text'] for r in train])],format='csr')
            joint_val=hstack([csr_matrix(fitted_pre.transform(xval)),fitted_tf.transform([r['text'] for r in val])],format='csr')
            models.append(('feature_fusion_logistic',LogisticRegression(C=1.0,max_iter=1000,solver='lbfgs',random_state=62),joint_train,joint_val))
        if name=='feature_fusion_logistic':
            joblib.dump({'preprocess':fitted_pre,'tfidf':fitted_tf,'classifier':model},a.out/'feature_fusion_bundle.joblib')
        write_json(a.out/'metrics.json',results)
        print(name+': validation macro-F1={:.4f}'.format(results[name]['macro_f1']),flush=True)
    write_json(a.out/'completion.json',{'status':'DEVELOPMENT_RUN_COMPLETE','test_evaluated':False,
        'metrics_sha256':sha(a.out/'metrics.json'),'protocol_sha256':sha(a.out/'protocol.json'),
        'train_sha256':sha(a.out/'train.jsonl'),'validation_sha256':sha(a.out/'validation.jsonl')})
    print('DEVELOPMENT_RUN_COMPLETE: '+str(a.out.resolve()),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('raw','cohort','duplicates','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--train-cap',type=int,default=3000);p.add_argument('--val-cap',type=int,default=1000)
    run(p.parse_args())
