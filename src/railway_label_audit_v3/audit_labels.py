"""Read-only candidate-label sensitivity audit. Python 3.8+, standard library only.
All policies are research proxies, not official accident classifications.
No training, splitting, input changes or silent missing-rate fallback.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

LEGACY = {2025:'7.20',2024:'7.19',2023:'7.05',2022:'6.73',2021:'6.45',
          2020:'6.90',2019:'6.90',2018:'6.62',2017:'6.75',2016:'6.64'}
NAMES = ['legacy', 'sourced_fx', 'sourced_fx_without_injuries',
         'sourced_fx_without_derailment', 'without_monetary_criterion',
         'casualty_outcome_only']

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):
            h.update(chunk)
    return h.hexdigest()

def value(s):
    n = Decimal(str(s).replace(',', '').strip())
    if not n.is_finite() or n < 0:
        raise ValueError('Missing, invalid or negative label ingredient: '+str(s))
    return n

def label(k, i, loss, p, f):
    # Exact 3-class collapse of the supplied nonnegative-count legacy rules.
    if k >= 3 or i >= 10 or loss >= 10000000 or p >= 2 or f >= 6:
        return 2
    if k >= 1 or i >= 1 or loss >= 1000000 or p >= 1 or f >= 2:
        return 1
    return 0

def run(candidate, rates_path, out):
    with rates_path.open() as f:
        rates = {int(r['year']):value(r['cny_per_usd']) for r in csv.DictReader(f)}
    out.mkdir(parents=True, exist_ok=False)
    counts = {n:Counter() for n in NAMES}
    transitions = {n:[[0]*3 for _ in range(3)] for n in NAMES[1:]}
    years=Counter(); changes=Counter(); compared_fx=Counter(); triggers=Counter()
    total=0; old_mismatch=0; empty=0; fallback=0
    with candidate.open(encoding='utf-8') as src, (out/'record_policy_labels.csv').open('w',newline='') as dst:
        w=csv.writer(dst)
        w.writerow(['source_row_1based','year','empty_narrative']+NAMES)
        for line in src:
            r=json.loads(line); meta=r['provenance']; a=r['targets_for_audit_only']
            y=int(meta['raw_identifiers']['YEAR4']); rate=rates[y]
            v={k:value(v) for k,v in a['raw_outcomes'].items()}
            for key in ('CASKLD','CASINJ','LOADP2','EMPTYP2','LOADF2','EMPTYF2','CABOOSE2'):
                assert v[key] == v[key].to_integral_value(), (meta,key)
            k,i,d=v['CASKLD'],v['CASINJ'],v['ACCDMG']
            p=v['LOADP2']+v['EMPTYP2']; f=v['LOADF2']+v['EMPTYF2']+v['CABOOSE2']
            c=d*rate
            labels=[label(k,i,d*Decimal(LEGACY.get(y,'7.00')),p,f),
                    label(k,i,c,p,f), label(k,0,c,p,f),label(k,i,c,0,0),
                    label(k,i,0,p,f), 2 if k>0 else (1 if i>0 else 0)]
            old_mismatch += labels[0] != a['legacy_label3']
            is_empty=not r['inputs']['narrative'].strip()
            empty+=is_empty;total+=1;years[y]+=1;fallback+=y not in LEGACY
            for n,l in zip(NAMES,labels): counts[n][l]+=1
            for n,l in zip(NAMES[1:],labels[1:]):
                transitions[n][labels[0]][l]+=1
                changes[n]+=l!=labels[0]
                compared_fx[n]+=l!=labels[1]
            for name,l in [('fatalities',label(k,0,0,0,0)),('injuries',label(0,i,0,0,0)),
                           ('money',label(0,0,c,0,0)),('derailment',label(0,0,0,p,f))]:
                if l==2: triggers[name]+=1
            w.writerow([meta['source_row_1based'],y,int(is_empty)]+labels)
    assert old_mismatch==0, 'Collapsed labels differ from supplied legacy labels'
    summary={'status':'LABEL_SENSITIVITY_ONLY_NOT_FINAL_LABELS', 'records':total,
             'year_min':min(years),'year_max':max(years),'missing_rate_years':[],
             'records_using_legacy_fallback':fallback,'empty_narratives':empty,
             'legacy_reconstruction_mismatches':old_mismatch,
             'counts':{n:dict(c) for n,c in counts.items()},
             'changed_vs_legacy':dict(changes),'changed_vs_sourced_fx':dict(compared_fx),
             'transitions_rows_legacy_columns_policy':transitions,
             'high_class_triggers_sourced_fx_nonexclusive':dict(triggers),
             'year_counts':dict(sorted(years.items())),
             'sha256':{'candidate':digest(candidate),'rates':digest(rates_path),
                       'script':digest(Path(__file__))},
             'limitations':['Fixed nominal thresholds; no inflation adjustment',
                 'All injuries are not serious injuries',
                 'Derailment counts omit locomotives and do not establish train type',
                 'Report-level analysis; related incidents not yet grouped',
                 'Original cohort including one empty narrative retained for comparison',
                 'No policy selected using model performance']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in
                     ['transitions_rows_legacy_columns_policy','year_counts']},indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidate',type=Path,required=True)
    ap.add_argument('--rates',type=Path,default=Path(__file__).with_name('fred_aexchus.csv'))
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();run(args.candidate,args.rates,args.out)
