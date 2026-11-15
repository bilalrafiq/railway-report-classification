"""Recompute all published final prediction metrics, without raw data or GPU."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
root=Path(__file__).resolve().parents[1];d=root/'results/final';m=json.loads((d/'metrics.json').read_text());c=json.loads((d/'completion.json').read_text())
assert hashlib.sha256((d/'metrics.json').read_bytes()).hexdigest()==c['metrics_sha256']
ref=None
for name,v in m.items():
 with (d/(name+'_predictions.csv')).open() as f:r=list(csv.DictReader(f))
 ids=[(x['incident_id'],x['date'],int(x['true'])) for x in r]
 if ref is None:ref=ids
 assert ref==ids and len(ids)==8935 and len({x[0] for x in ids})==8935
 y=[x[2] for x in ids];q=[int(x['predicted']) for x in r];p=np.array([[float(x['p'+str(i)]) for i in range(3)] for x in r])
 assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all() and np.allclose(p.sum(1),1,atol=1e-5) and np.array_equal(q,p.argmax(1))
 assert abs(f1_score(y,q,average='macro')-v['macro_f1'])<1e-12
 assert abs(accuracy_score(y,q)-v['accuracy'])<1e-12
 assert confusion_matrix(y,q).tolist()==v['confusion_matrix']
 print(name,round(v['macro_f1'],6))
print('PASS: all 13 final prediction files')
