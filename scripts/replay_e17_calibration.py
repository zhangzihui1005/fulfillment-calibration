#!/usr/bin/env python3
"""Replay E17 Core calibration from a user-owned archive of private score records.

Outputs are private unless only the aggregate validation JSON/metrics are retained.
The original archive is never extracted or executed. No model is refitted.
"""
from __future__ import annotations
import argparse, hashlib, io, json, os, sys, zipfile
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(k,'1')
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fulfillment_recalibration.e17_calibration import ORIGINS,replay
METHODS=['Frozen','Expanding','Rolling60','Approval60','PoolRate','Platt','BrierOpt']

def read_csv(archive: zipfile.ZipFile, ending: str) -> pd.DataFrame:
    members=[n for n in archive.namelist() if n.replace('\\','/').endswith(ending)]
    if len(members)!=1:
        raise ValueError(f'Expected one archive member ending in {ending!r}; found {len(members)}')
    return pd.read_csv(io.BytesIO(archive.read(members[0])),compression='gzip' if ending.endswith('.gz') else None)

def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive',type=Path,required=True)
    ap.add_argument('--origin',choices=list(ORIGINS),default='O2')
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--atol',type=float,default=1e-7)
    ap.add_argument('--save-private-records',action='store_true')
    a=ap.parse_args()
    if not a.archive.is_file():ap.error('Archive does not exist.')
    if a.atol<=0:ap.error('--atol must be positive.')
    if a.out.exists() and any(a.out.iterdir()):ap.error('Use an empty/new output directory; existing results are never overwritten.')
    a.out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.archive) as z:
        h=read_csv(z,f'inputs/S1_{a.origin}_scored.csv.gz')
        ref=read_csv(z,f'predictions/Core_{a.origin}.csv.gz')
        pools=read_csv(z,f'Core_{a.origin}_pools.csv')
    required={'order_id','t','feedback','promise_end','eligibility','y','raw'}
    if not required.issubset(h):raise ValueError(f'Missing score columns: {sorted(required-set(h))}')
    for c in ['t','feedback','promise_end','eligibility']:h[c]=pd.to_datetime(h[c],errors='raise')
    if h.order_id.duplicated().any():raise ValueError('Duplicate history order ID.')
    if not h.y.isin([0,1]).all() or not h.raw.between(0,1).all():raise ValueError('Invalid binary labels/probabilities.')
    with threadpool_limits(limits=1):q,log=replay(h,a.origin,METHODS)
    q=q.set_index('order_id').sort_index();ref=ref.set_index('order_id').sort_index()
    if not q.index.equals(ref.index):raise AssertionError('Evaluation IDs differ.')
    diffs={m:float(np.max(np.abs(q[m].to_numpy()-ref[m].to_numpy()))) for m in ['Raw',*METHODS]}
    keys=['Date','Method'];log['Date']=pd.to_datetime(log.Date);pools['Date']=pd.to_datetime(pools.Date)
    got=log.sort_values(keys).reset_index(drop=True);expected=pools.sort_values(keys).reset_index(drop=True)
    if len(got)!=len(expected):raise AssertionError('Pool-row counts differ.')
    for k in ['Date','Method','N','Positives','PoolHash','Active','Solver']:
        if not got[k].equals(expected[k]):raise AssertionError(f'Pool mismatch in {k}.')
    errors={k:float(np.nanmax(np.abs(got[k].to_numpy(float)-expected[k].to_numpy(float)))) for k in ['Intercept','Slope','PoolRate','MeanOrderAge']}
    passed=max(diffs.values())<=a.atol and max(errors['Intercept'],errors['Slope'])<1e-5
    records=[]
    for m in ['Raw',*METHODS]:
        y=q.y.to_numpy();p=q[m].to_numpy();pc=np.clip(p,1e-7,1-1e-7)
        records.append({'Origin':a.origin,'Method':m,'N':len(q),'Positive':int(y.sum()),'Brier':float(np.mean((p-y)**2)),'LogLoss':float(np.mean(-y*np.log(pc)-(1-y)*np.log1p(-pc)))})
    pd.DataFrame(records).to_csv(a.out/'replayed_metrics.csv',index=False)
    result={'passed':bool(passed),'origin':a.origin,'rows':len(q),'pool_rows':len(got),'prediction_max_abs_error':diffs,'pool_parameter_max_abs_error':errors,'atol':a.atol,'scope':'Private-score calibration replay; no LR retraining or feature reconstruction','private_records_written':a.save_private_records}
    (a.out/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if a.save_private_records:
        q.reset_index().to_csv(a.out/'PRIVATE_predictions.csv.gz',index=False,compression='gzip')
        log.to_csv(a.out/'PRIVATE_pool_log.csv',index=False)
    print(json.dumps(result,indent=2))
    if not passed:raise SystemExit(1)
if __name__=='__main__':main()
