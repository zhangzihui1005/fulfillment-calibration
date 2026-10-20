#!/usr/bin/env python3
"""Public aggregate-to-core-table entry. No private records, model training or data download.
Re-estimates paired score intervals from supplied daily sufficient statistics.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
PAIRS=[('Frozen','Raw'),('Expanding','Frozen'),('Expanding','Raw')]
def weights(n,l):
 rng=np.random.default_rng(20261009);ix=(rng.integers(0,n-l+1,size=(2000,(n+l-1)//l))[:,:,None]+np.arange(l)).reshape(2000,-1)[:,:n]
 return np.array([np.bincount(x,minlength=n) for x in ix])
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=ROOT/'build/core');a=ap.parse_args()
 if a.out.exists() and any(a.out.iterdir()):ap.error('Use a new or empty output directory.')
 a.out.mkdir(parents=True,exist_ok=True);threadpool_limits(1);E=ROOT/'evidence';D=E/'r2'
 d=pd.read_csv(D/'daily_model_sufficient_statistics.csv');reference=pd.read_csv(D/'core_triplets_all_backbones_blocks.csv');refx=pd.read_csv(D/'cross_backbone_effects.csv')
 rows=[];scores=[];effects={};cross=[];checks=[]
 for o in ['O1','O2','O3','O4']:
  ns=None
  for mode in ['Core','Restricted','CatBoost']:
   q=d[(d.Origin==o)&(d.Backbone==mode)];n=q[q.Method=='Raw'].sort_values('DayIndex').N.to_numpy()
   assert len(n)==60 and n.sum()>0
   if ns is not None:assert np.array_equal(n,ns)
   ns=n;ds={}
   for method in ['Raw','Frozen','Expanding']:
    g=q[q.Method==method].sort_values('DayIndex');assert np.array_equal(g.N,n)
    ds[method]={m:g[m+'Sum'].to_numpy() for m in ['Brier','LogLoss']}
    scores.append({'Backbone':mode,'Origin':o,'Method':method,'N':int(n.sum()),'Positives':int(g.Positives.sum()),'Brier':g.BrierSum.sum()/n.sum(),'LogLoss':g.LogLossSum.sum()/n.sum(),'MeanPred':g.PredictionSum.sum()/n.sum(),'ActualRate':g.Positives.sum()/n.sum()})
   effects[mode,o]={}
   for aa,bb in PAIRS:
    for m in ['Brier','LogLoss']:
     diff=ds[aa][m]-ds[bb][m];effects[mode,o][aa,bb,m]=diff
     for l in [3,7,14]:
      w=weights(60,l);lo,hi=np.quantile(w@diff/(w@n),[.025,.975]);point=diff.sum()/n.sum();r=reference.query('Backbone==@mode and Origin==@o and A==@aa and B==@bb and Metric==@m and BlockDays==@l').iloc[0]
      err=max(abs(point-r.Delta),abs(lo-r.CI_low),abs(hi-r.CI_high));checks.append(err);assert err<1e-11
      rows.append({'Backbone':mode,'Origin':o,'A':aa,'B':bb,'Metric':m,'BlockDays':l,'Delta':point,'CI_low':lo,'CI_high':hi})
  for mode in ['CatBoost','Restricted']:
   for aa,bb in [('Expanding','Frozen'),('Expanding','Raw')]:
    for m in ['Brier','LogLoss']:
     diff=effects[mode,o][aa,bb,m]-effects['Core',o][aa,bb,m]
     for l in [3,7,14]:
      w=weights(60,l);lo,hi=np.quantile(w@diff/(w@ns),[.025,.975]);p=diff.sum()/ns.sum();r=refx.query('Origin==@o and Alternative==@mode and A==@aa and B==@bb and Metric==@m and BlockDays==@l').iloc[0];err=max(abs(p-r.Delta),abs(lo-r.CI_low),abs(hi-r.CI_high));checks.append(err);assert err<1e-11
      cross.append({'Origin':o,'Alternative':mode,'Reference':'Core','A':aa,'B':bb,'Metric':m,'BlockDays':l,'Delta':p,'CI_low':lo,'CI_high':hi})
 bounds=pd.read_csv(D/'unknown_outcome_triplet_bounds.csv');low=(bounds.KnownLossDifferenceSum+bounds.UnknownLowerSum)/bounds.N;hi=(bounds.KnownLossDifferenceSum+bounds.UnknownUpperSum)/bounds.N
 checks.extend(np.abs(low-bounds.BrierLower));checks.extend(np.abs(hi-bounds.BrierUpper));assert max(checks)<1e-11
 for name,frame in [('core_triplets.csv',pd.DataFrame(rows)),('absolute_backbone_scores.csv',pd.DataFrame(scores)),('cross_backbone_effects.csv',pd.DataFrame(cross)),('unknown_outcome_triplet_bounds.csv',bounds)]:frame.to_csv(a.out/name,index=False,float_format='%.17g')
 # One compact main table with complete paired score differences.
 m=pd.read_csv(E/'e17/manuscript_S1_complete_main_table.csv');p=pd.DataFrame(rows);main=[]
 for o in ['O1','O2','O3','O4']:
  item={'Origin':o}
  for method in ['Raw','Frozen','Expanding','PoolRate','Platt','BrierOpt','PeriodicLR']:
   r=m.query('Origin==@o and Method==@method').iloc[0]
   for met in ['Brier','LogLoss']:item[method+'_'+met]=r[met]
  for aa,bb in PAIRS:
   for met in ['Brier','LogLoss']:
    r=p.query('Backbone=="Core" and Origin==@o and A==@aa and B==@bb and Metric==@met and BlockDays==7').iloc[0]
    for x in ['Delta','CI_low','CI_high']:item[f'{aa}_{bb}_{met}_{x}']=r[x]
  main.append(item)
 pd.DataFrame(main).to_csv(a.out/'Table_core_comparisons.csv',index=False,float_format='%.17g')
 result={'passed':True,'checks':len(checks),'max_abs_difference':max(checks),'paired_intervals_rebuilt':len(rows),'cross_backbone_intervals_rebuilt':len(cross),'unknown_ranges_checked':len(bounds),'input_scope':'Public daily sufficient statistics and aggregated unknown-loss bounds; no record-level or model reproduction','no_training':True}
 (a.out/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
