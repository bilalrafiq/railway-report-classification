"""Incident cohort and label-bound candidates. Not a final training split.

Max/sum bounds assume valid report counts of subsets of the same incident's
derailed passenger/freight cars, and complete reporting. They address unknown
overlap, not incorrect counts, missing reports, or legal-grade equivalence.
"""
import argparse,csv,json,hashlib,re
from pathlib import Path
from collections import defaultdict,Counter
from decimal import Decimal

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def n(s):
    x=Decimal(str(s).replace(',','').strip())
    if not x.is_finite() or x<0:raise ValueError('Invalid outcome: '+str(s))
    return x

def clean(r):
    s=' '.join(r.get('NARR'+str(i),'') for i in range(1,16))
    s=re.sub(r'([,.!?()])',r' \1 ',s)
    return ' '.join(t for t in s.split() if t.lower() not in ('nan','none'))

def label(k,i,loss,p,f):
    if k>=3 or i>=10 or loss>=10000000 or p>=2 or f>=6:return 2
    if k>=1 or i>=1 or loss>=1000000 or p>=1 or f>=2:return 1
    return 0

def bounds(rs):
    # Total unique affected cars lies between the largest report and the sum
    # only under the subset/coverage assumptions in the module docstring.
    return max(r['p'] for r in rs),sum(r['p'] for r in rs),max(r['f'] for r in rs),sum(r['f'] for r in rs)

def run(raw,identity,rates_path,out):
    manifest=json.loads((identity/'summary.json').read_text())
    assert sha(raw)==manifest['sha256']['raw'],'Raw file mismatch'
    assignments=identity/'inc3_assignments.csv'
    assert sha(assignments)==manifest['sha256']['assignments'],'Identity assignment mismatch'
    with assignments.open(newline='') as f:byrow={int(r['source_row_1based']):r for r in csv.DictReader(f)}
    with rates_path.open(newline='') as f:rates={int(r['year']):n(r['cny_per_usd']) for r in csv.DictReader(f)}
    groups=defaultdict(list);missing=[];raw_total=0
    with raw.open(encoding='utf-8-sig',newline='') as f:
        for row,r in enumerate(csv.DictReader(f),1):
            raw_total+=1
            if row not in byrow:missing.append(row);continue
            a=byrow[row];s=clean(r)
            vals={k:n(r[k]) for k in ['CASKLD','CASINJ','ACCDMG','LOADP2','EMPTYP2','LOADF2','EMPTYF2','CABOOSE2']}
            for k,v in vals.items():
                if k!='ACCDMG':assert v==int(v)
            groups[a['inc3_group_id']].append({'row':row,'year':int(r['YEAR4']),
                'date':'{}-{:02d}-{:02d}'.format(int(r['YEAR4']),int(r['MONTH']),int(r['DAY'])),
                'k':vals['CASKLD'],'i':vals['CASINJ'],'loss':vals['ACCDMG'],
                'p':int(vals['LOADP2']+vals['EMPTYP2']),
                'f':int(vals['LOADF2']+vals['EMPTYF2']+vals['CABOOSE2']),
                'narrative':s,'joint':r['JOINTCD'],'train':r['TRNNBR'],
                'reasons':a['review_reasons'],'old':int(a['in_old_candidate'])})
    assert missing==manifest['missing_inc3_rows'] and raw_total==manifest['counts']['raw_rows']
    out.mkdir(parents=True,exist_ok=False)
    status=Counter();classes=Counter();years=Counter();metrics=Counter();rows=[];reviews=[]
    for gid,rs in sorted(groups.items()):
        texts=[r for r in rs if r['narrative']]
        why=sorted({x for r in rs for x in r['reasons'].split(';') if x})
        rowids=[r['row'] for r in rs];low=high=chosen=None;pl=pu=fl=fu=None
        # Holdout here means pending data review, NOT a model evaluation set.
        if why:state='held_identity_exception'
        elif not texts:state='no_cleaned_narrative'
        elif rs[0]['year'] not in rates:state='held_missing_exchange_rate'
        else:
            assert len({(r['k'],r['i'],r['loss'],r['date']) for r in rs})==1
            pl,pu,fl,fu=bounds(rs);r=rs[0];loss=r['loss']*rates[r['year']]
            low=label(r['k'],r['i'],loss,pl,fl);high=label(r['k'],r['i'],loss,pu,fu)
            assert low<=high
            state='stable_label_under_overlap_bounds' if low==high else 'held_derailment_overlap_sensitive'
            if low==high:
                chosen=low;classes[chosen]+=1;years[r['year']]+=1
                metrics['stable_groups_with_multiple_reports']+=int(len(rs)>1)
                metrics['stable_groups_with_nonexact_counts']+=int(pl!=pu or fl!=fu)
                metrics['stable_groups_without_old_candidate']+=int(not any(r['old'] for r in rs))
            else:
                reviews.append({'group_id':gid,'source_rows':rowids,'label_lower':low,'label_upper':high,
                                'passenger_bounds':[pl,pu],'freight_bounds':[fl,fu],
                                'reports':[{'row':r['row'],'jointcd':r['joint'],'train':r['train'],
                                            'passenger':r['p'],'freight':r['f']} for r in rs]})
        status[state]+=1
        # A reference text source for review only; no model-input policy frozen.
        representative=max(texts,key=lambda r:(len(r['narrative']),-r['row']))['row'] if texts else None
        rows.append([gid,state,chosen,low,high,pl,pu,fl,fu,len(rs),representative,
                     rs[0]['date'] if not why else '',json.dumps(rowids),';'.join(why)])
    header=['inc3_group_id','status','candidate_label','label_lower','label_upper','passenger_lower',
            'passenger_upper','freight_lower','freight_upper','report_count','reference_narrative_source_row',
            'incident_date','all_source_rows','identity_review_reasons']
    with (out/'incident_label_candidates.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rows)
    with (out/'derailment_sensitive_groups.jsonl').open('w') as f:
        for r in reviews:f.write(json.dumps(r)+'\n')
    summary={'status':'COHORT_CANDIDATE_NOT_FINAL_TRAINING_DATA','raw_rows':raw_total,
             'incident_groups':len(groups),'unkeyed_source_rows':missing,'group_disposition':dict(status),
             'stable_candidate_class_counts':dict(classes),'stable_candidate_year_counts':dict(sorted(years.items())),
             **dict(metrics),'sha256':{'raw':sha(raw),'identity_assignments':sha(assignments),
                 'rates':sha(rates_path),'script':sha(Path(__file__)),
                 'label_candidates':sha(out/'incident_label_candidates.csv')},
             'assumptions':['Recorded derailment counts are valid subset counts and reports cover the incident',
                            'Passenger and freight car categories are consistent across reports',
                            'Adapted total-injury/nominal-CNY target; not official grades'],
             'pending':['Incomplete report coverage and missing links','Inflation sensitivity',
                        'Input selection and outcome disclosures in narrative','Narrative duplicates and final split',
                        'Selection bias from held groups; report exclusions explicitly']}
    assert sum(status.values())==len(groups)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='stable_candidate_year_counts'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True);p.add_argument('--identity',type=Path,required=True)
    p.add_argument('--rates',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.raw,a.identity,a.rates,a.out)
