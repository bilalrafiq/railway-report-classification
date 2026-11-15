#!/usr/bin/env python3
"""Create an auditable candidate dataset, NOT a frozen training dataset.

Python 3.8+, pandas and the supplied original processor are required.
Legacy labels/cohort selection are retained for controlled comparison.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from collections import Counter
from datetime import date
from pathlib import Path


def number(value):
    try:
        value = float(str(value).strip().replace(',', ''))
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def integer(value):
    value = number(value)
    return int(value) if value is not None and value.is_integer() else None


def category(value):
    value = str(value).strip()
    return 'UNKNOWN' if value.lower() in ('', 'nan', 'none', 'null', '.') else value


def code(value):
    value = integer(value)
    return str(value) if value is not None else 'UNKNOWN'


def temporal(row):
    y, m, d, h, minute = [integer(row.get(k)) for k in
                         ('YEAR4', 'MONTH', 'DAY', 'TIMEHR', 'TIMEMIN')]
    year_ok = y is not None and 1 <= y <= 9999
    month_ok = m is not None and 1 <= m <= 12
    try:
        day = date(y, m, d).isoformat()
    except (ValueError, TypeError):
        day = None
    ampm = category(row.get('AMPM', '')).upper()
    valid_time = (h is not None and 1 <= h <= 12 and minute is not None
                  and 0 <= minute <= 59 and ampm in ('AM', 'PM'))
    hour = (h % 12 + (12 if ampm == 'PM' else 0)) if valid_time else None
    fraction = hour + minute / 60 if valid_time else None
    features = {
        'Month_Sin': math.sin(2 * math.pi * m / 12) if month_ok else None,
        'Month_Cos': math.cos(2 * math.pi * m / 12) if month_ok else None,
        'Hour_Sin': math.sin(2 * math.pi * fraction / 24) if valid_time else None,
        'Hour_Cos': math.cos(2 * math.pi * fraction / 24) if valid_time else None,
        'Year_Norm': (y - 1975) / 50 if year_ok else None,
    }
    clock = '{:02d}:{:02d}'.format(hour, minute) if valid_time else 'unknown'
    return features, day, valid_time, 'Date: {}. Time: {}.'.format(day or 'unknown', clock)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--original-processor', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not args.raw.is_file() or not args.original_processor.is_file():
        parser.error('Input CSV or original processor is missing')
    # Refuse overwrites, including prior candidate output.
    args.out.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('legacy_processor', args.original_processor)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    processor = old.FinalProcessorV6(str(args.raw), '')
    counts, labels, missing = Counter(), Counter(), Counter()
    seen = set()
    outcomes = ['CASKLD', 'CASINJ', 'ACCDMG', 'LOADP2', 'EMPTYP2',
                'LOADF2', 'EMPTYF2', 'CABOOSE2']
    provenance = ['RAILROAD', 'INCDTNO', 'YEAR', 'YEAR4', 'MONTH', 'DAY',
                  'TIMEHR', 'TIMEMIN', 'AMPM', 'RR2', 'INCDTNO2', 'IYR2',
                  'IMO2', 'RR3', 'INCDTNO3', 'IYR3', 'IMO3']
    with args.raw.open(encoding='utf-8-sig', newline='') as source, \
            (args.out / 'candidate.jsonl').open('w', encoding='utf-8') as dest, \
            (args.out / 'time_issues.jsonl').open('w', encoding='utf-8') as issues:
        for row_number, row in enumerate(csv.DictReader(source), start=1):
            counts['raw_rows'] += 1
            if row.get('NARRLEN', '').strip().lower() in ('0', '', 'nan'):
                counts['excluded_legacy_narrlen'] += 1
                continue
            key = tuple(row.get(k, '') for k in ('RAILROAD', 'INCDTNO', 'YEAR'))
            if key in seen:
                counts['excluded_legacy_duplicate_key'] += 1
                continue
            seen.add(key)
            numeric = {k: number(row.get(k)) for k in processor.num_cols}
            # Zero coordinates are treated as missing under the provisional
            # US railway dataset policy, not as a universal geographic rule.
            for k, limit in [('Latitude', 90), ('Longitud', 180)]:
                value = numeric[k]
                if value is not None and (value == 0 or abs(value) > limit):
                    numeric[k] = None
            time, day, valid_time, prefix = temporal(row)
            numeric.update(time)
            speed, tons = numeric['TRNSPD'], numeric['TONS']
            numeric['Kinetic_Energy'] = (0.5 * tons * speed ** 2
                                         if tons is not None and speed is not None else None)
            cats = {k: category(row.get(k, '')) for k in processor.cat_cols}
            for k in ('TYPE', 'WEATHER', 'VISIBLTY'):
                cats[k] = code(row.get(k))
            # Reuse supplied descriptions only where an explicit mapping exists.
            # Codes 12/13 remain literal until checked against the data dictionary.
            typ = cats['TYPE']
            type_text = old.get_Type_accident(typ) if typ in {str(i) for i in range(1, 12)} else 'code ' + typ
            weather = {'1': 'clear', '2': 'cloudy', '3': 'rain', '4': 'fog',
                       '5': 'sleet', '6': 'snow'}.get(cats['WEATHER'], 'unknown')
            visibility = {'1': 'dawn', '2': 'day', '3': 'dusk', '4': 'dark'}.get(cats['VISIBLTY'], 'unknown')
            narrative = old.clean_text_for_plm(' '.join(row.get('NARR' + str(i), '') for i in range(1, 16)))
            # Labels are explicitly provisional and kept outside model inputs.
            label6, _ = processor._calculate_severity(row)
            label3 = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2}[label6]
            meta = {'source_row_1based': row_number, 'raw_identifiers':
                    {k: row.get(k, '') for k in provenance}, 'date': day}
            item = {
                'provenance': meta,
                'targets_for_audit_only': {'legacy_label6': label6, 'legacy_label3': label3,
                                          'raw_outcomes': {k: row.get(k, '') for k in outcomes}},
                'inputs': {'structured_data': {'numeric': numeric, 'categorical': cats},
                           'numeric_missing': {k: int(v is None) for k, v in numeric.items()},
                           'narrative': narrative,
                           'text_data': '{} Visibility: {}. Weather: {}. Accident type: {}. Narrative: {}'.format(
                               prefix, visibility, weather, type_text, narrative)},
            }
            dest.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')
            counts['candidate_rows'] += 1
            labels[str(label3)] += 1
            counts['invalid_or_missing_date'] += int(day is None)
            counts['invalid_or_missing_clock'] += int(not valid_time)
            counts['empty_cleaned_narrative'] += int(not narrative)
            for k, v in numeric.items():
                if v is None:
                    missing[k] += 1
            if day is None or not valid_time:
                issues.write(json.dumps(meta, ensure_ascii=False) + '\n')
    summary = {
        'status': 'CANDIDATE_ONLY_NOT_FOR_TRAINING',
        'counts': dict(counts), 'legacy_label3_counts': dict(labels),
        'numeric_missing_counts': dict(missing),
        'raw_csv_sha256': digest(args.raw),
        'original_processor_sha256': digest(args.original_processor),
        'candidate_script_sha256': digest(Path(__file__)),
        'candidate_jsonl_sha256': digest(args.out / 'candidate.jsonl'),
        'pending': ['Confirm label definition and currency policy',
                    'Validate all code descriptions against source data dictionary',
                    'Review duplicate records and linked incident reports',
                    'Freeze evaluation split and train-only imputation',
                    'Review post-event inputs and outcome-bearing narratives'],
    }
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print('Completed:', args.out.resolve())


if __name__ == '__main__':
    main()
