"""Development neural comparison; only train/validation files are accepted."""
import argparse, csv, hashlib, json, random, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

NUM=['TEMP','TRNSPD','TONS','CARS']
CAT=['STATE','VISIBLTY','WEATHER','TRKCLAS','TYPTRK','PASSTRN','TRNDIR']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf8')
def read(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def numeric(rows):return np.asarray([[np.nan if r['features'][k] is None else float(r['features'][k]) for k in NUM] for r in rows],dtype=np.float64)

class Reports(Dataset):
    def __init__(self,rows,processor):
        self.rows=rows
        a=numeric(rows);a=np.where(np.isfinite(a),a,np.asarray(processor['median']))
        self.nums=torch.tensor((a-np.asarray(processor['mean']))/np.asarray(processor['std']),dtype=torch.float32)
        self.cats=torch.tensor([[processor['maps'][k].get(str(r['features'][k]),0) for k in CAT] for r in rows],dtype=torch.long)
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return self.nums[i],self.cats[i],self.rows[i]['text'],self.rows[i]['label'],self.rows[i]['incident_id']

def main(a):
    if a.out.exists():raise ValueError('Output exists; use a fresh directory')
    if not a.plm.is_dir():raise ValueError('Local pretrained model missing; download the RoBERTa model and tokenizer files to the --plm directory as described in docs/REPRODUCIBILITY.md')
    source=a.source
    protocol=json.loads((source/'protocol.json').read_text())
    completion=json.loads((source/'completion.json').read_text())
    if completion['test_evaluated'] is not False or completion['protocol_sha256']!=sha(source/'protocol.json'):raise ValueError('Protocol/completion mismatch')
    train=read(source/'train.jsonl');val=read(source/'validation.jsonl')
    for split,rows in [('train',train),('validation',val)]:
        ids=[r['incident_id'] for r in rows]
        if len(ids)!=len(set(ids)) or len(rows)!=protocol['selected_counts'][split]:raise ValueError('Count or ID mismatch')
        if not all((int(r['date'][:4])<=2017 if split=='train' else 2018<=int(r['date'][:4])<=2020) for r in rows):raise ValueError('Unexpected date')
        expected=completion.get(split+'_sha256')
        if expected and expected!=sha(source/(split+'.jsonl')):raise ValueError('Data checksum mismatch')
    if set(r['incident_id'] for r in train)&set(r['incident_id'] for r in val):raise ValueError('Split overlap')
    sys.path.insert(0,str(a.original.resolve()))
    from model import RailwayDualExpertModel
    a.out.mkdir(parents=True)
    arr=numeric(train)
    med=[float(np.median(arr[np.isfinite(arr[:,j]),j])) if np.isfinite(arr[:,j]).any() else 0.0 for j in range(len(NUM))]
    clean=np.where(np.isfinite(arr),arr,np.asarray(med));mean=clean.mean(0);std=clean.std(0);std[std<1e-8]=1
    maps={k:{v:i+1 for i,v in enumerate(sorted(set(str(r['features'][k]) for r in train)))} for k in CAT}
    processor={'numeric':NUM,'categorical':CAT,'median':med,'mean':mean.tolist(),'std':std.tolist(),'maps':maps}
    save(a.out/'processor.json',processor)
    tokenizer=AutoTokenizer.from_pretrained(str(a.plm),local_files_only=True)
    def collate(batch):
        nums,cats,texts,labels,ids=zip(*batch)
        encoded=tokenizer(list(texts),padding=True,truncation=True,max_length=256,return_tensors='pt')
        return torch.stack(nums),torch.stack(cats),encoded['input_ids'],encoded['attention_mask'],torch.tensor(labels),list(ids)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda' and not a.smoke:raise RuntimeError('Full neural training requires GPU')
    cfg=SimpleNamespace(D_MODEL=128,STRUCT_DROPOUT=0.1,TRANSFORMER_LAYERS=2,PLM_PATH=str(a.plm.resolve()),GATE_BIAS=-3.0)
    manifest={'status':'SMOKE_ONLY' if a.smoke else 'PROVISIONAL_DEVELOPMENT','test_evaluated':False,'seed':a.seed,'epochs':a.epochs,'batch_size':a.batch_size,'max_length':256,'learning_rate':0.0001,'weight_decay':0.01,'loss':'unweighted cross entropy, no smoothing; revised common comparison setting','trainable_plm_layers':[10,11], 'numeric_preprocessing':'train median and mean/std; all four retained; no missing indicators','target':protocol['target'],'source_hashes':{n:sha(source/n) for n in ['train.jsonl','validation.jsonl','protocol.json']},'original_hashes':{n:sha(a.original/n) for n in ['model.py','model_zoo.py']},'pretrained_hashes':{p.name:sha(p) for p in a.plm.iterdir() if p.is_file()},'torch_version':torch.__version__,'device':str(device),'script_sha256':sha(Path(__file__)),'limitations':['Development target remains adapted, not official grades','Text-only head uses zero structural half to match classifier dimensions','Smoke does not estimate predictive performance','No claim of bitwise deterministic GPU training']}
    save(a.out/'protocol.json',manifest)
    for mode in ['text','concat','gated']:
        random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
        model=RailwayDualExpertModel(len(NUM),{k:len(maps[k]) for k in CAT},cfg).to(device)
        if mode!='gated':
            for n,p in model.fusion.named_parameters():
                if not n.startswith('norm.'):p.requires_grad=False
        if mode=='text':
            for p in model.struct_expert.parameters():p.requires_grad=False
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=0.0001,weight_decay=0.01)
        def forward(nums,cats,ids,mask):
            h=model.text_adapter(model.plm(input_ids=ids,attention_mask=mask).last_hidden_state);vt=h[:,0]
            vs=torch.zeros_like(vt) if mode=='text' else model.struct_expert(nums,cats)
            fused=model.fusion(h,vs,vt,mask)[0] if mode=='gated' else model.fusion.norm(torch.cat([vs,vt],-1))
            return model.classifier(fused)
        data=train[:32] if a.smoke else train
        loader=DataLoader(Reports(data,processor),batch_size=a.batch_size,shuffle=True,num_workers=0,collate_fn=collate,generator=torch.Generator().manual_seed(a.seed))
        vloader=DataLoader(Reports(val,processor),batch_size=a.batch_size,shuffle=False,num_workers=0,collate_fn=collate) if not a.smoke else None
        history=[];best=-1
        for epoch in range(1,(1 if a.smoke else a.epochs)+1):
            model.train();loss_sum=0;count=0
            for step,b in enumerate(loader):
                nums,cats,ids,mask,y=[x.to(device) for x in b[:5]]
                optimizer.zero_grad(set_to_none=True);logits=forward(nums,cats,ids,mask);loss=nn.functional.cross_entropy(logits,y)
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                loss.backward()
                if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise ValueError('Nonfinite gradients')
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();loss_sum+=loss.item()*len(y);count+=len(y)
                if a.smoke and step>=1:break
            entry={'epoch':epoch,'training_loss':loss_sum/count}
            if not a.smoke:
                model.eval();truth=[];probs=[];identifiers=[]
                with torch.no_grad():
                    for b in vloader:
                        nums,cats,ids,mask,y=[x.to(device) for x in b[:5]]
                        p=forward(nums,cats,ids,mask).softmax(-1).cpu().numpy();probs.extend(p.tolist());truth.extend(y.cpu().tolist());identifiers.extend(b[5])
                pred=np.asarray(probs).argmax(1);score=float(f1_score(truth,pred,labels=[0,1,2],average='macro',zero_division=0))
                entry.update(macro_f1=score,accuracy=float(accuracy_score(truth,pred)),confusion_matrix=confusion_matrix(truth,pred,labels=[0,1,2]).tolist())
                if score>best:
                    best=score;torch.save(model.state_dict(),a.out/(mode+'_best.pt'))
                    with (a.out/(mode+'_predictions.csv')).open('w',newline='') as f:
                        w=csv.writer(f);w.writerow(['incident_id','true','predicted','p0','p1','p2']);w.writerows([gid,y,int(pr)]+p for gid,y,pr,p in zip(identifiers,truth,pred,probs))
                    save(a.out/(mode+'_best_metrics.json'),entry)
            history.append(entry);save(a.out/(mode+'_history.json'),history);print(mode,entry,flush=True)
        del optimizer,model
        if torch.cuda.is_available():torch.cuda.empty_cache()
    save(a.out/'completion.json',{'status':'REAL_DATA_SMOKE_COMPLETE' if a.smoke else 'NEURAL_DEVELOPMENT_COMPLETE','test_evaluated':False})

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['source','original','plm','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--epochs',type=int,default=5);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--seed',type=int,default=62)
    a=p.parse_args()
    if a.epochs<1 or a.batch_size<1:raise ValueError('Positive epochs/batch size required')
    main(a)
