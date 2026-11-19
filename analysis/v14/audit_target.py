"""Post-test descriptive audit. No fitting, checkpoint selection, or prediction changes.

Keyword flags are screening indicators, not validated outcome disclosures.
Negated mentions are included. No model attribution or causal inference is made.
"""
import argparse, csv, hashlib, io, json, re, zipfile
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

PATTERNS = {
    'casualty_mention': r'\b(?:kill(?:ed)?|fatal(?:ity|ities)?|death(?:s)?|died|injur\w*|casualt\w*)\b',
    'damage_mention': r'\b(?:damag\w*|dollars?|cost\w*)\b|\$',
    'derailment_mention': r'\bderail\w*\b',
}
def sha(b): return hashlib.sha256(b).hexdigest()
def table(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def run(root, rawzip, out):
    if out.exists(): raise ValueError('Choose a fresh output directory')
    expected=json.loads((root/'data/raw_snapshot.json').read_text())
    with zipfile.ZipFile(rawzip) as z: raw=z.read('datanew.csv')
    assert sha(raw)==expected['sha256'] and len(raw)==expected['bytes']
    cohort_path=root/'data/incident_label_candidates.csv'
    split_path=root/'results/development_baselines/split_manifest.csv'
    protocol=json.loads((root/'results/development_baselines/protocol.json').read_text())
    assert sha(cohort_path.read_bytes())==protocol['hashes']['cohort']
    frozen=json.loads((root/'results/final/frozen_protocol.json').read_text())
    assert sha(split_path.read_bytes())==frozen['split_manifest_sha256']
    with cohort_path.open() as f: cohort={r['inc3_group_id']:r for r in csv.DictReader(f) if r['status']=='stable_label_under_overlap_bounds'}
    with split_path.open() as f: splits={r['incident_id']:r for r in csv.DictReader(f)}
    assert set(cohort)==set(splits)
    rate_path=root/'src/railway_cohort_v6/fred_aexchus.csv'
    with rate_path.open() as f: rates={int(r['year']):Decimal(r['cny_per_usd']) for r in csv.DictReader(f)}
    source={int(r['reference_narrative_source_row']):g for g,r in cohort.items()}
    records={}; counts=defaultdict(Counter); trigger_counts=Counter(); mismatches=[]
    for idx,r in enumerate(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))),1):
        if idx not in source: continue
        g=source[idx]; c=cohort[g]; split=splits[g]['disposition']
        text=' '.join(t for t in ' '.join(r.get('NARR'+str(i),'') for i in range(1,16)).split() if t.lower() not in ('nan','none'))
        k=Decimal(r['CASKLD'].replace(',','')); inj=Decimal(r['CASINJ'].replace(',','')); loss=Decimal(r['ACCDMG'].replace(',',''))*rates[int(r['YEAR4'])]
        p=int(c['passenger_lower']); freight=int(c['freight_lower'])
        levels={'fatalities':2 if k>=3 else 1 if k>=1 else 0,
                'injuries':2 if inj>=10 else 1 if inj>=1 else 0,
                'nominal_cny_damage':2 if loss>=10000000 else 1 if loss>=1000000 else 0,
                'passenger_derailment_lower':2 if p>=2 else 1 if p>=1 else 0,
                'freight_derailment_lower':2 if freight>=6 else 1 if freight>=2 else 0}
        y=int(c['candidate_label']); computed=max(levels.values())
        if computed!=y: mismatches.append(g)
        flags={name:bool(re.search(pattern,text,re.I)) for name,pattern in PATTERNS.items()}
        flags['any_mention']=any(flags.values())
        records[g]={'incident_id':g,'split':split,'label':y,**flags,'word_count':len(text.split())}
        counts[split]['n']+=1; counts[split]['class_'+str(y)]+=1
        for name,v in flags.items(): counts[split][name]+=int(v)
        if y>0:
            defining=[name for name,lev in levels.items() if lev==y]
            for name in defining: trigger_counts[(split,y,name)]+=1
            if len(defining)==1: trigger_counts[(split,y,'sole_'+defining[0])]+=1
    assert len(records)==len(cohort) and not mismatches
    summaries=[]
    for split,c in sorted(counts.items()):
        summaries.append({'split':split,**dict(c),**{k+'_pct':100*c[k]/c['n'] for k in list(PATTERNS)+['any_mention']}})
    # Fixed, descriptive subgroup metrics. Always average over classes 0,1,2.
    subgroup=[]; prediction_hashes={}; reference=None
    for path in sorted((root/'results/final').glob('*_predictions.csv')):
        prediction_hashes[path.name]=sha(path.read_bytes())
        with path.open() as f: pred=list(csv.DictReader(f))
        ids=[r['incident_id'] for r in pred]
        assert len(ids)==8935 and len(set(ids))==8935
        if reference is None: reference=ids
        assert ids==reference
        for r in pred:
            assert records[r['incident_id']]['label']==int(r['true']) and records[r['incident_id']]['split']=='reserved_future'
        for flag in ['any_mention']:
            for present in [False,True]:
                sub=[r for r in pred if records[r['incident_id']][flag]==present]
                cm=[[0]*3 for _ in range(3)]
                for r in sub: cm[int(r['true'])][int(r['predicted'])]+=1
                f1=[]
                for k in range(3):
                    den=sum(cm[k])+sum(row[k] for row in cm)
                    f1.append(2*cm[k][k]/den if den else 0.)
                subgroup.append({'model':path.stem.replace('_predictions',''),'screen':flag,'present':present,'n':len(sub),'class_0':sum(cm[0]),'class_1':sum(cm[1]),'class_2':sum(cm[2]),'macro_f1':sum(f1)/3,'accuracy':sum(cm[k][k] for k in range(3))/len(sub) if sub else None})
    assert len(prediction_hashes)==13
    out.mkdir(parents=True)
    table(out/'mention_summary.csv',summaries)
    table(out/'label_triggers.csv',[{'split':s,'label':y,'trigger':name,'count':v} for (s,y,name),v in sorted(trigger_counts.items())])
    table(out/'descriptive_subgroups.csv',subgroup)
    table(out/'incident_screen.csv',list(records.values()))
    audit={'status':'DESCRIPTIVE_POST_TEST_AUDIT_COMPLETE','records':len(records),'label_reconstruction_mismatches':len(mismatches),'training_performed':False,'predictions_changed':False,'patterns':PATTERNS,'limitations':['Keyword screening includes negations and non-outcome contexts; no manual validation performed.','No keyword match does not establish absence of outcome information.','Subgroup scores are post-test descriptive comparisons and are confounded by class composition and other differences.','Labels reconstructed using selected-row outcome fields and provided incident-level lower derailment bounds; this does not independently validate identity grouping, source fields, or regulatory validity.'],'hashes':{'raw':sha(raw),'cohort':sha(cohort_path.read_bytes()),'split':sha(split_path.read_bytes()),'rates':sha(rate_path.read_bytes()),'script':sha(Path(__file__).read_bytes()),'predictions':prediction_hashes}}
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps({'records':len(records),'mismatches':len(mismatches),'splits':summaries},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,required=True);p.add_argument('--raw-zip',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.repo,a.raw_zip,a.out)
