"""Frozen future-period evaluation of existing checkpoints. No fitting or selection."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'2')
import csv,json,hashlib,sys,argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import joblib,torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from scipy.sparse import csr_matrix,hstack
from sklearn.metrics import f1_score,accuracy_score,classification_report,confusion_matrix
from data_definition import NUM,CAT,number,text_of
from neural_definition import Reports

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def require(ok,msg):
 if not ok:raise ValueError(msg)
def main(a):
 root=a.root.resolve();source=root/'results/development_v10_20261115_132741_435856'
 out=root/('results/validation_replay_v12' if a.validation_check else 'results/final_evaluation_v12')
 expected_count=6188 if a.validation_check else 8935
 require(not out.exists(),'Output exists. Preserve it; do not repeat evaluation blindly.')
 if not a.validation_check:
  replay=root/'results/validation_replay_v12/completion.json'
  require(replay.is_file() and read(replay)['status']=='VALIDATION_REPLAY_COMPLETE' and read(replay).get('script_sha256')==sha(Path(__file__)),'Run --validation-check first')
 protocol=read(source/'protocol.json');receipt=read(source/'completion.json')
 for n,k in [('protocol.json','protocol_sha256'),('train.jsonl','train_sha256'),('validation.jsonl','validation_sha256')]:require(sha(source/n)==receipt[k],'Source changed: '+n)
 raw=root/'data/raw_original/datanew.csv';cohort=root/'src/cohort_v6/local_results/incident_label_candidates.csv'
 require(sha(raw)==protocol['hashes']['raw'],'Raw changed');require(sha(cohort)==protocol['hashes']['cohort'],'Cohort changed')
 manifest=list(csv.DictReader((source/'split_manifest.csv').open()))
 future={r['incident_id']:r for r in manifest if r['disposition']==('validation' if a.validation_check else 'reserved_future')}
 require(len(future)==expected_count,'Unexpected future cohort')
 other={r['duplicate_component'] for r in manifest if r['disposition'] in ['train','validation']}
 if not a.validation_check:require(not other.intersection(r['duplicate_component'] for r in future.values()),'Component overlap')
 runs={62:root/'results/neural_development_v10_seed62_20261115_143518',63:root/'results/neural_repeat_v10_20261115_170240/seed_63',64:root/'results/neural_repeat_v10_20261115_170240/seed_64'}
 original=root/'src/training_original/zyrailway/a_roberta_co_con_res_ve';plm=root/'pretrained_models/roberta-base-v10'
 conv=root/'results/logistic_completion_v11_20261115_221643'
 require(read(conv/'metrics.json')['status']=='CONVERGENCE_RETRY_COMPLETE','Convergence unresolved')
 hashes={};base=None
 for seed,run in runs.items():
  p=read(run/'protocol.json');require(p['seed']==seed,'Wrong seed')
  require(read(run/'completion.json')['status']=='NEURAL_DEVELOPMENT_COMPLETE','Incomplete neural run')
  comparable={k:v for k,v in p.items() if k!='seed'}
  if base is None:base=comparable
  require(base==comparable,'Training protocol differs')
  for n,h in p['source_hashes'].items():require(sha(source/n)==h,'Neural source differs')
  for n,h in p['original_hashes'].items():require(sha(original/n)==h,'Model source differs')
  for n,h in p['pretrained_hashes'].items():require(sha(plm/n)==h,'Pretrained file differs')
  for mode in ['text','concat','gated']:hashes[str(seed)+'/'+mode]=sha(run/(mode+'_best.pt'))
  require(read(run/'processor.json')==read(runs[62]/'processor.json'),'Preprocessors differ')
 for name in ['majority','structured_logistic','text_tfidf_logistic']:hashes[name]=sha(source/(name+'.joblib'))
 hashes['feature_fusion_logistic']=sha(conv/'feature_fusion_bundle.joblib')
 require(torch.cuda.is_available(),'GPU required for neural evaluation')
 out.mkdir(parents=True)
 save(out/'frozen_protocol.json',{'status':'FROZEN_BEFORE_TEST_PREDICTIONS','target':protocol['target'],'interpretation':'Retrospective adapted-index classification; not official grades or prospective prediction','primary_metric':'macro_f1','seeds':[62,63,64],'selection':'Existing maximum validation macro-F1 checkpoint per model and seed','evaluation_years':'2018-2020 validation replay' if a.validation_check else '2021-2025 as available in the supplied extract; not necessarily complete 2025','expected_cases':expected_count,'validation_replay':a.validation_check,'checkpoint_hashes':hashes,'split_manifest_sha256':sha(source/'split_manifest.csv'),'source_protocol_sha256':sha(source/'protocol.json'),'script_sha256':sha(Path(__file__)),'no_post_test_tuning':True})
 meta={}
 with cohort.open() as f:
  for r in csv.DictReader(f):
   if r['inc3_group_id'] in future:
    require(r['status']=='stable_label_under_overlap_bounds','Unstable selected label')
    meta[int(r['reference_narrative_source_row'])]=r
 require(len(meta)==expected_count,'Nonunique reference rows')
 records=[]
 with raw.open(encoding='utf-8-sig',newline='') as f:
  for i,row in enumerate(csv.DictReader(f),1):
   if i not in meta:continue
   m=meta[i];g=m['inc3_group_id'];date=m['incident_date']
   require(date==future[g]['date'] and ((2018<=int(date[:4])<=2020) if a.validation_check else (2021<=int(date[:4])<=2025)),'Date mismatch')
   features={k:number(row.get(k)) for k in NUM};features.update({k:row.get(k,'').strip() or '<MISSING>' for k in CAT})
   records.append({'incident_id':g,'date':date,'text':text_of(row),'label':int(m['candidate_label']),'features':features})
 records.sort(key=lambda r:r['incident_id']);require(len(records)==expected_count,'Missing future records')
 truth=np.array([r['label'] for r in records]);results={}
 def score(y,p):
  q=p.argmax(1)
  return {'macro_f1':float(f1_score(y,q,labels=[0,1,2],average='macro',zero_division=0)),'accuracy':float(accuracy_score(y,q)),'confusion_matrix':confusion_matrix(y,q,labels=[0,1,2]).tolist(),'classification_report':classification_report(y,q,labels=[0,1,2],output_dict=True,zero_division=0)}
 def record(name,prob):
  require(prob.shape==(expected_count,3) and np.isfinite(prob).all() and (prob>=0).all() and (prob<=1).all() and np.allclose(prob.sum(1),1,atol=1e-5),'Invalid probabilities')
  result=score(truth,prob);result['by_year']={}
  for year in sorted(set(r['date'][:4] for r in records)):
   idx=np.array([r['date'][:4]==year for r in records]);result['by_year'][year]=score(truth[idx],prob[idx])
  if a.validation_check:
   if '_seed' in name:
    mode,seed=name.split('_seed');expected=read(runs[int(seed)]/(mode+'_best_metrics.json'))
   elif name=='feature_fusion_logistic':expected=read(conv/'metrics.json')
   else:expected=read(source/'metrics.json')[name]
   require(abs(result['macro_f1']-expected['macro_f1'])<1e-10 and result['confusion_matrix']==expected['confusion_matrix'],'Validation replay differs: '+name)
  results[name]=result
  with (out/(name+'_predictions.csv')).open('w',newline='') as f:
   w=csv.writer(f);w.writerow(['incident_id','date','true','predicted','p0','p1','p2'])
   w.writerows([r['incident_id'],r['date'],int(y),int(q)]+p.tolist() for r,y,q,p in zip(records,truth,prob.argmax(1),prob))
  save(out/'metrics.json',results);print(name,result['macro_f1'],flush=True)
 frame=pd.DataFrame([r['features'] for r in records])
 for k in NUM:frame[k]=pd.to_numeric(frame[k],errors='coerce')
 for name in ['majority','structured_logistic','text_tfidf_logistic']:
  model=joblib.load(source/(name+'.joblib'));require(list(model.classes_)==[0,1,2],'Class order mismatch')
  x=np.zeros((len(records),1)) if name=='majority' else (frame if name=='structured_logistic' else [r['text'] for r in records])
  record(name,model.predict_proba(x))
 bundle=joblib.load(conv/'feature_fusion_bundle.joblib');require(list(bundle['classifier'].classes_)==[0,1,2],'Fusion class order mismatch')
 x=hstack([csr_matrix(bundle['preprocess'].transform(frame)),bundle['tfidf'].transform([r['text'] for r in records])],format='csr')
 record('feature_fusion_logistic',bundle['classifier'].predict_proba(x))
 sys.path.insert(0,str(original));from model import RailwayDualExpertModel
 tokenizer=AutoTokenizer.from_pretrained(str(plm),local_files_only=True)
 def collate(batch):
  nums,cats,texts,ys,ids=zip(*batch);enc=tokenizer(list(texts),padding=True,truncation=True,max_length=256,return_tensors='pt')
  return torch.stack(nums),torch.stack(cats),enc['input_ids'],enc['attention_mask']
 cfg=SimpleNamespace(D_MODEL=128,STRUCT_DROPOUT=0.1,TRANSFORMER_LAYERS=2,PLM_PATH=str(plm),GATE_BIAS=-3.0)
 for seed,run in runs.items():
  processor=read(run/'processor.json');loader=DataLoader(Reports(records,processor),batch_size=8,shuffle=False,num_workers=0,collate_fn=collate)
  for mode in ['text','concat','gated']:
   model=RailwayDualExpertModel(len(NUM),{k:len(processor['maps'][k]) for k in CAT},cfg)
   model.load_state_dict(torch.load(run/(mode+'_best.pt'),map_location='cpu'),strict=True);model=model.cuda().eval();probs=[]
   with torch.no_grad():
    for b in loader:
     nums,cats,ids,mask=[v.cuda() for v in b]
     h=model.text_adapter(model.plm(input_ids=ids,attention_mask=mask).last_hidden_state);vt=h[:,0]
     vs=torch.zeros_like(vt) if mode=='text' else model.struct_expert(nums,cats)
     fused=model.fusion(h,vs,vt,mask)[0] if mode=='gated' else model.fusion.norm(torch.cat([vs,vt],-1))
     probs.append(model.classifier(fused).softmax(-1).cpu().numpy())
   record(mode+'_seed'+str(seed),np.concatenate(probs));del model;torch.cuda.empty_cache()
 save(out/'completion.json',{'status':'VALIDATION_REPLAY_COMPLETE' if a.validation_check else 'FINAL_EVALUATION_COMPLETE','test_evaluated':not a.validation_check,'models':len(results),'cases':expected_count,'metrics_sha256':sha(out/'metrics.json'),'script_sha256':sha(Path(__file__))})
 print('RESULTS:',out,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--validation-check',action='store_true');main(p.parse_args())
