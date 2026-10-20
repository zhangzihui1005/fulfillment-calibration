#!/usr/bin/env python3
"""Rebuild the Olist S1 four-origin LR and E16-S1 window results from original Olist data.

The original source is NOT included. Outputs are private, record-level artifacts and
should not be committed to a public repository. Cross-platform floating-point parity
is not guaranteed. Original research used Python 3.12 + pinned numerical packages.
"""
from __future__ import annotations
import argparse, json, sys, platform
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import brentq
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'src'))
from fulfillment_recalibration import olist_helpers as base

ORIGINS={'O1':'2017-07-01','O2':'2017-10-01','O3':'2018-01-01','O4':'2018-04-01'}
CLIP=1e-7
SEED=20261009


def calc_scores(y,p):
    p=np.asarray(p,float); y=np.asarray(y,int)
    pc=np.clip(p,CLIP,1-CLIP)
    return {'Brier':float(np.mean((p-y)**2)),
            'LogLoss':float(np.mean(-y*np.log(pc)-(1-y)*np.log1p(-pc))),
            'AP':float(average_precision_score(y,p)),
            'MeanPred':float(np.mean(p)),
            'ActualRate':float(np.mean(y)),
            'OnTimeBrierContribution':float(np.mean((1-y)*p*p)),
            'LateBrierContribution':float(np.mean(y*(1-p)**2))}


def intercept(z,y):
    """Unpenalized Bernoulli intercept, as in original frozen study."""
    n=len(y); pos=int(np.sum(y))
    if n<1000 or pos<50 or n-pos<50:
        return 0., False
    prevalence=pos/n
    center=np.log(prevalence/(1-prevalence))
    b=brentq(lambda b:np.mean(expit(z+b))-prevalence,
             center-z.max()-1,center-z.min()+1,xtol=1e-12,rtol=1e-12)
    assert abs(np.mean(expit(z+b))-prevalence)<1e-9
    return float(b), True


def bootstrap_ci(d, metricA, metricB, evaluation_start, evaluation_end, num=2000):
    """Original frozen study's calendar-day noncircular paired block resampling."""
    byday=d.groupby('day').apply(lambda x: pd.Series({
        'N':len(x),
        'd':float(np.sum(x[metricA]-x[metricB]))}),include_groups=False)
    # Preserve calendar dates with zero evaluated orders (important for O3).
    calendar=pd.date_range(evaluation_start,evaluation_end,inclusive='left',freq='D')
    byday=byday.reindex(calendar,fill_value=0)
    T=len(byday)
    assert T==60, f'Expected 60 calendar days, saw {T}'
    rng=np.random.default_rng(SEED)
    ix=(rng.integers(0,T-6,size=(num,(T+6)//7))[:,:,None]+np.arange(7)).reshape(num,-1)[:,:T]
    block_counts=np.array([np.bincount(row,minlength=T) for row in ix])
    vals=(block_counts@byday.d.to_numpy())/(block_counts@byday.N.to_numpy())
    return tuple(map(float,np.quantile(vals,[0.025,0.975])))


def run_one(cohort:pd.DataFrame, nums:list[str], cats:list[str], key:str,
            outdir:Path, write_records:bool):
    cutoff=pd.Timestamp(ORIGINS[key]); start=cutoff+pd.Timedelta(days=60); end=start+pd.Timedelta(days=60)
    train=cohort.loc[cohort.order_approved_at.lt(cutoff) & cohort.label_observed_at.lt(cutoff)]
    cand=cohort.loc[cohort.order_approved_at.ge(cutoff)&cohort.order_approved_at.lt(end)].copy()
    assert not set(train.order_id)&set(cand.order_id)
    model=base.make_lr(nums,cats)
    model.fit(train[nums+cats],train.y)
    cand['raw']=model.predict_proba(cand[nums+cats])[:,1]
    h=cand[['order_id','order_approved_at','label_observed_at','promise_end','y','raw']].copy()
    h=h.rename(columns={'order_approved_at':'t','label_observed_at':'feedback'}).sort_values(['t','order_id']).reset_index(drop=True)
    h['eligibility']=h[['feedback','promise_end']].max(axis=1)
    assert h.eligibility.ge(h.promise_end).all()
    q=h.loc[h.t.ge(start)&h.t.lt(end)].copy().reset_index(drop=True)
    q['day']=q.t.dt.normalize()
    policies=['Raw','Expanding','Rolling60','Approval60','Frozen']
    z=np.log(np.clip(h.raw.to_numpy(),CLIP,1-CLIP)/(1-np.clip(h.raw.to_numpy(),CLIP,1-CLIP)))
    zq=np.log(np.clip(q.raw.to_numpy(),CLIP,1-CLIP)/(1-np.clip(q.raw.to_numpy(),CLIP,1-CLIP)))
    q['Raw']=q.raw
    for policy in policies[1:]:q[policy]=np.nan
    ys=h.y.to_numpy(int)
    pool_rows=[]; frozen=None
    for day in pd.date_range(start,end,inclusive='left'):
        trg=q.t.ge(day)&q.t.lt(day+pd.Timedelta(days=1))
        for policy in policies[1:]:
            fit=start if policy=='Frozen' else day
            mask=(h.t.lt(fit)&h.feedback.lt(fit)&h.eligibility.lt(fit)).to_numpy()
            if policy=='Rolling60':mask &= h.eligibility.ge(fit-pd.Timedelta(days=60)).to_numpy()
            if policy=='Approval60':mask &= h.t.ge(fit-pd.Timedelta(days=60)).to_numpy()
            y=ys[mask];zz=z[mask]
            if policy=='Frozen' and frozen is not None:
                b, active = frozen
            else:
                b, active=intercept(zz,y)
                if policy=='Frozen':frozen=(b,active)
            if mask.any():
                assert h.loc[mask,'eligibility'].max()<fit
            q.loc[trg,policy]=expit(zq[trg.to_numpy()]+b) if active else q.loc[trg,'Raw']
            pool_rows.append({'Origin':key,'Day':day,'Method':policy,'N':int(mask.sum()),
                              'Positives':int(y.sum()),'Active':bool(active),'Intercept':b,
                              'MeanOrderAgeDays':float(((fit-h.loc[mask,'t']).dt.total_seconds()/86400).mean()) if mask.any() else None})
    assert q[policies].notna().all().all()
    rows=[]
    for policy in policies:
        s=calc_scores(q.y,q[policy]);rows.append({'Origin':key,'Start':start.date(),'EndExclusive':end.date(),
              'Method':policy,'N':len(q),'Positives':int(q.y.sum()),**s})
    for policy in policies:
        q[policy+'_brier']=(q[policy]-q.y)**2
    contrasts=[]
    for a,b in [('Expanding','Raw'),('Rolling60','Expanding'),('Approval60','Rolling60')]:
        d=float(np.mean(q[a+'_brier']-q[b+'_brier']))
        lo,hi=bootstrap_ci(q,a+'_brier',b+'_brier',start,end)
        contrasts.append({'Origin':key,'A':a,'B':b,'BrierDelta':d,'CI_low':lo,'CI_high':hi})
    p=outdir/key;p.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(pool_rows).to_csv(p/'update_pools.csv',index=False)
    if write_records:
        # PRIVATE record-level predictions: must remain in outputs/ excluded by Git.
        q[['order_id','t','y',*policies]].to_csv(p/'private_predictions.csv',index=False)
    return rows,contrasts


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    gr=ap.add_mutually_exclusive_group(required=True)
    gr.add_argument('--olist-zip',type=Path)
    gr.add_argument('--data-dir',type=Path)
    ap.add_argument('--out',type=Path,default=HERE/'outputs'/'olist_rebuild')
    ap.add_argument('--origins',nargs='+',choices=list(ORIGINS),default=list(ORIGINS))
    ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--save-private-predictions',action='store_true')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=args.threads):
        print('Loading Olist raw source...',flush=True)
        tables=base.load_tables(args.olist_zip or args.data_dir)
        cohort,flow=base.make_cohort(tables['orders'],'deadline')
        flow.to_csv(args.out/'cohort_flow.csv',index=False)
        assert len(cohort)==96394, f'Unexpected cohort size {len(cohort)}'
        print('Building historical seller features...',flush=True)
        d,nums,cats=base.make_features(tables,cohort)
        d['promise_end']=d.order_estimated_delivery_date.dt.normalize()+pd.Timedelta(days=1)
        d['month_sin']=np.sin(2*np.pi*(d.approval_month-1)/12)
        d['month_cos']=np.cos(2*np.pi*(d.approval_month-1)/12)
        nums=list(nums);nums.remove('approval_month');nums+=['month_sin','month_cos']
        allrows=[];allpairs=[]
        for key in args.origins:
            print('Fitting and replaying',key,flush=True)
            rows,contrasts=run_one(d,nums,cats,key,args.out,args.save_private_predictions)
            allrows+=rows;allpairs+=contrasts
            print(key,'done:',next(x for x in rows if x['Method']=='Expanding')['Brier'],flush=True)
        pd.DataFrame(allrows).to_csv(args.out/'summary.csv',index=False)
        pd.DataFrame(allpairs).to_csv(args.out/'paired_brier.csv',index=False)
        (args.out/'manifest.json').write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),
             'python':platform.python_version(),'origins':args.origins,'seed':SEED,'scope':'reconstructed Olist S1 main protocol',
             'private_order_records':args.save_private_predictions},indent=2),encoding='utf-8')
    print('DONE - private run outputs:',args.out.resolve())

if __name__=='__main__':main()
