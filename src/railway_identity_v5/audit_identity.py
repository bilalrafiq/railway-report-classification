"""Audit FRA INC3-style keys and JOINTCD; preserve all inputs. Python 3.8+."""
import argparse,csv,json,hashlib
from pathlib import Path
from collections import Counter,defaultdict
from datetime import date
from decimal import Decimal

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def num(s):
    n=Decimal(s.replace(',','').strip())
    assert n.is_finite() and n>=0
    return str(n.normalize())

def run(raw,candidate,out):
    out.mkdir(parents=True,exist_ok=False)
    selected=set()
    with candidate.open(encoding='utf-8') as f:
        for line in f:selected.add(json.loads(line)['provenance']['source_row_1based'])
    groups=defaultdict(list);missing=[];counts=Counter();old=defaultdict(list)
    with raw.open(encoding='utf-8-sig',newline='') as f:
        for row,r in enumerate(csv.DictReader(f),1):
            counts['raw_rows']+=1
            # FRA FAQ key uses IYR+IMO+RR3+INCDTNO3, not the reporting RR key.
            k=(r['IYR'].strip(),r['IMO'].strip().zfill(2),r['RR3'].strip().upper(),r['INCDTNO3'].strip().upper())
            if not all(k):missing.append(row);continue
            try:d=date(int(r['YEAR4']),int(r['MONTH']),int(r['DAY'])).isoformat()
            except (ValueError,TypeError):d=None
            item={'row':row,'joint':r['JOINTCD'].strip(),'date':d,
                  'outcomes':tuple(num(r[x]) for x in ['CASKLD','CASINJ','ACCDMG']),
                  'train':r['TRNNBR'].strip(),'typeq':r['TYPEQ'].strip(),
                  'equipment':(r['RRCAR1'].strip(),r['CARNBR1'].strip()),
                  'has_narrative':any(r['NARR'+str(i)].strip() for i in range(1,16)),
                  'candidate':row in selected,
                  'linked_date_mismatch':r['IYR3'].strip()!=r['IYR'].strip() or r['IMO3'].strip().zfill(2)!=r['IMO'].strip().zfill(2)}
            groups[k].append(item)
            old[(r['RAILROAD'].strip().upper(),r['INCDTNO'].strip().upper(),r['YEAR4'],r['MONTH'])].append(item)
            counts['jointcd_'+item['joint']+'_rows']+=1
    review=[];assign=[];hist=Counter();by_reason=Counter();eligible=Counter()
    for k,rs in sorted(groups.items()):
        gid='INC3_'+hashlib.sha256(json.dumps(k).encode()).hexdigest()[:20]
        j=Counter(r['joint'] for r in rs);hist[str(j['1'])]+=1
        reasons=[]
        if j['1']!=1:reasons.append('not_exactly_one_jointcd1')
        ds={r['date'] for r in rs}
        if len(ds)!=1 or None in ds:reasons.append('inconsistent_or_invalid_date')
        if len({r['outcomes'] for r in rs})!=1:reasons.append('incident_outcome_disagreement')
        if any(r['linked_date_mismatch'] for r in rs):reasons.append('linked_month_year_disagreement')
        for x in reasons:by_reason[x]+=1
        candidates=sum(r['candidate'] for r in rs)
        if candidates:counts['groups_with_old_candidates']+=1
        if len(rs)>1:counts['multi_report_inc3_groups']+=1
        if not reasons:
            eligible['consistent_inc3_groups']+=1
            if any(r['has_narrative'] for r in rs):eligible['consistent_groups_with_any_raw_narrative']+=1
            if candidates:eligible['consistent_groups_with_old_candidates']+=1
        else:
            review.append({'inc3_key':k,'group_id':gid,'reasons':reasons,'reports':rs})
        for r in rs:assign.append([r['row'],gid,r['joint'],int(r['candidate']),';'.join(reasons)])
    repeated=[rs for rs in old.values() if len(rs)>1]
    counts['repeated_reporting_keys']=len(repeated)
    counts['repeated_keys_containing_jointcd3']=sum(any(r['joint']=='3' for r in rs) for rs in repeated)
    counts['repeated_keys_with_multiple_nonblank_train_ids']=sum(len({r['train'] for r in rs if r['train']})>1 for rs in repeated)
    counts['repeated_keys_with_multiple_equipment_ids']=sum(len({r['equipment'] for r in rs if all(r['equipment'])})>1 for rs in repeated)
    with (out/'inc3_assignments.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['source_row_1based','inc3_group_id','jointcd','in_old_candidate','review_reasons']);w.writerows(sorted(assign))
    with (out/'inc3_review.jsonl').open('w') as f:
        for r in review:f.write(json.dumps(r)+'\n')
    summary={'status':'IDENTITY_AUDIT_NOT_FINAL_COHORT','counts':dict(counts),
             'inc3_groups':len(groups),'missing_inc3_rows':missing,
             'jointcd1_count_per_group_histogram':dict(hist),
             'flagged_groups':len(review),'flag_reasons':dict(by_reason),'consistency_screen':dict(eligible),
             'sha256':{'raw':sha(raw),'candidate':sha(candidate),'script':sha(Path(__file__)),
                       'assignments':sha(out/'inc3_assignments.csv')},
             'limitations':['No derailment aggregation or final label assignment',
                            'Matching keys are not independent verification of physical incidents',
                            'Raw narrative presence is not the final cleaned-text eligibility test',
                            'No training split, cohort exclusion, or input modification']}
    assert len(assign)+len(missing)==counts['raw_rows']
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.raw,a.candidate,a.out)
