"""Read-only input screening; keyword flags do not establish label leakage."""
import argparse, csv, hashlib, json, re
from collections import Counter, defaultdict
from pathlib import Path

PATTERNS = {
    'casualty_terms': r'\b(?:kill(?:ed)?|fatal(?:ity|ities)?|death(?:s)?|injur\w*|casualt\w*)\b',
    'derailment_terms': r'\bderail\w*\b',
    'damage_terms': r'\b(?:damage\w*|loss(?:es)?|cost(?:s)?)\b|\$\s*\d',
    'numeric_derailment': r'\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:\w+\s+){0,4}derail\w*\b',
}

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def run(raw, cohort, out):
    summary_path=cohort.parent/'summary.json'
    manifest=json.loads(summary_path.read_text())
    if sha(raw)!=manifest['sha256']['raw']: raise ValueError('Raw checksum mismatch')
    if sha(cohort)!=manifest['sha256']['label_candidates']: raise ValueError('Cohort checksum mismatch')
    byrow={}; labels={}; flags=defaultdict(set); examples={}; texts=defaultdict(set)
    with cohort.open(newline='') as f:
        for r in csv.DictReader(f):
            if r['status']!='stable_label_under_overlap_bounds': continue
            gid=r['inc3_group_id']; labels[gid]=int(r['candidate_label'])
            for row in json.loads(r['all_source_rows']):
                if row in byrow: raise ValueError('Row in multiple groups')
                byrow[row]=gid
    patterns={k:re.compile(v,re.I) for k,v in PATTERNS.items()}
    counts=Counter(); reviewed=0
    with raw.open(encoding='utf-8-sig',newline='') as f:
        for row,r in enumerate(csv.DictReader(f),1):
            if row not in byrow: continue
            reviewed+=1; gid=byrow[row]
            narrative=' '.join(r.get('NARR'+str(i),'') for i in range(1,16))
            narrative=' '.join(t for t in narrative.split() if t.lower() not in ('nan','none'))
            for k,p in patterns.items():
                match=p.search(narrative)
                if match:
                    flags[gid].add(k)
                    if (gid,k) not in examples:
                        examples[(gid,k)]={'group_id':gid,'source_row':row,'category':k,
                            'context':narrative[max(0,match.start()-80):match.end()+120]}
            for field in ('CARSDMG','CARSHZD'):
                try: positive=float(r[field].replace(',',''))>0
                except (ValueError,KeyError): positive=False
                if positive: flags[gid].add(field+'_positive')
            norm=' '.join(re.findall(r'\w+',narrative.casefold()))
            if len(norm)>=40:
                texts[hashlib.sha256(norm.encode()).hexdigest()].add(gid)
    for gid in labels:
        counts.update(flags[gid])
        if flags[gid]&set(patterns): counts['any_narrative_keyword']+=1
    dup=[{'text_sha256':h,'group_ids':sorted(g),'class_ids':sorted({labels[x] for x in g})}
         for h,g in sorted(texts.items()) if len(g)>1]
    out.mkdir(parents=True,exist_ok=False)
    with (out/'group_flags.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['group_id','candidate_label','flags'])
        for gid in sorted(labels): w.writerow([gid,labels[gid],';'.join(sorted(flags[gid]))])
    with (out/'review_examples.jsonl').open('w') as f:
        # Deterministic short review sample per category, not a prevalence sample.
        used=Counter()
        for key,e in sorted(examples.items()):
            if used[e['category']]<25:
                f.write(json.dumps(e)+'\n'); used[e['category']]+=1
    with (out/'duplicate_narrative_groups.jsonl').open('w') as f:
        for r in dup:f.write(json.dumps(r)+'\n')
    result={'status':'INPUT_SCREENING_NOT_LEAKAGE_CERTIFICATION','candidate_groups':len(labels),
        'raw_reports_screened':reviewed,'group_flags':dict(counts),
        'repeated_normalized_narrative_sets':len(dup),
        'distinct_groups_in_repeated_narratives':len(set(x for r in dup for x in r['group_ids'])),
        'repeated_sets_with_different_candidate_labels':sum(len(r['class_ids'])>1 for r in dup),
        'sha256':{'raw':sha(raw),'cohort':sha(cohort),'script':sha(Path(__file__))},
        'limitations':['Keyword matches include negations; no match does not prove no outcome disclosure',
        'All reports within candidate groups screened, not finalized model inputs',
        'Duplicate screen ignores case/punctuation and requires at least 40 normalized characters',
        'Duplicate texts can describe distinct incidents; do not merge incident identities automatically',
        'Training loader not inspected; serialized outcome metadata is not proof it reached the model']}
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.raw,a.cohort,a.out)
