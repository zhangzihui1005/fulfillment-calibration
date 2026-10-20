"""Align Frozen/Raw/Expanding evidence using archived E17 records, without model fitting.
Private input: the extracted 03_E17 experiment directory. Public output: aggregate statistics.
Run: python scripts/align_core_comparisons.py --e17-dir PATH --out evidence/r2
"""
from pathlib import Path
import argparse, json, hashlib
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits
ORIGINS={'O1':'2017-07-01','O2':'2017-10-01','O3':'2018-01-01','O4':'2018-04-01'}
PAIRS=[('Frozen','Raw'),('Expanding','Frozen'),('Expanding','Raw')]
SEED=20261009

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def weights(n,l):
 rng=np.random.default_rng(SEED)
 ix=(rng.integers(0,n-l+1,size=(2000,(n+l-1)//l))[:,:,None]+np.arange(l)).reshape(2000,-1)[:,:n]
 return np.array([np.bincount(x,minlength=n) for x in ix])
def loss(p,y,metric):
 if metric=='Brier':return (p-y)**2
 p=np.clip(p,1e-7,1-1e-7);return -y*np.log(p)-(1-y)*np.log1p(-p)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--e17-dir',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
 root=args.e17_dir;out=args.out;out.mkdir(parents=True,exist_ok=True)
 threadpool_limits(1)
 def save(name,rows):pd.DataFrame(rows).to_csv(out/name,index=False,float_format='%.17g')
 inputs={};checks=[];daily=[];cross=[];weekly=[];dailytraj=[];unknown=[];cohorts=[];reconstruct=[];absolute=[]
 original_metrics=pd.read_csv(root/'metrics.csv');original_pairs=pd.read_csv(root/'paired_intervals.csv')
 tri=original_pairs[(original_pairs.Metric.isin(['Brier','LogLoss'])) & original_pairs.apply(lambda r:(r.A,r.B) in PAIRS,axis=1)].copy()
 tri.to_csv(out/'core_triplets_all_backbones_blocks.csv',index=False,float_format='%.17g')
 tri[(tri.Backbone=='Core')&(tri.BlockDays==7)].to_csv(out/'core_triplets_7day.csv',index=False,float_format='%.17g')
 W={l:weights(60,l) for l in [3,7,14]}
 for key,fit in ORIGINS.items():
  start=pd.Timestamp(fit)+pd.Timedelta(days=60);qs={};deltas={}
  for mode in ['Core','Restricted','CatBoost']:
   path=root/'predictions'/f'{mode}_{key}.csv.gz';inputs[path.name]=digest(path)
   q=pd.read_csv(path,parse_dates=['t']).sort_values('order_id').reset_index(drop=True);qs[mode]=q
   if mode!='Core':
    assert q.order_id.equals(qs['Core'].order_id);assert q.t.equals(qs['Core'].t);assert q.y.equals(qs['Core'].y)
   assert not q.order_id.duplicated().any()
   di=(q.t.dt.normalize()-start).dt.days.to_numpy();assert ((di>=0)&(di<60)).all()
   n=np.bincount(di,minlength=60);pos=np.bincount(di,weights=q.y,minlength=60);y=q.y.to_numpy(float)
   ds={}
   for method in ['Raw','Frozen','Expanding']:
    p=q[method].to_numpy(float);ds[method]={}
    for met in ['Brier','LogLoss']:
     ls=loss(p,y,met);s=np.bincount(di,weights=ls,minlength=60);ds[method][met]=s
     v=original_metrics.query('Backbone==@mode and Origin==@key and Method==@method')[met].iloc[0]
     checks.append({'check':'metric','origin':key,'mode':mode,'method':method,'metric':met,'max_error':abs(ls.mean()-v),'pass':bool(abs(ls.mean()-v)<1e-12)})
    predsum=np.bincount(di,weights=p,minlength=60)
    for d in range(60):daily.append({'Backbone':mode,'Origin':key,'Date':str((start+pd.Timedelta(days=d)).date()),'DayIndex':d,'Method':method,'N':int(n[d]),'Positives':int(pos[d]),'PredictionSum':predsum[d],'BrierSum':ds[method]['Brier'][d],'LogLossSum':ds[method]['LogLoss'][d]})
    absolute.append(original_metrics.query('Backbone==@mode and Origin==@key and Method==@method').iloc[0].to_dict())
   deltas[mode]={}
   for a,b in PAIRS:
    for met in ['Brier','LogLoss']:
     d=ds[a][met]-ds[b][met];deltas[mode][a,b,met]=d
     for l,w in W.items():
      lo,hi=np.quantile((w@d)/(w@n),[.025,.975]);row=tri.query('Backbone==@mode and Origin==@key and A==@a and B==@b and Metric==@met and BlockDays==@l').iloc[0]
      err=max(abs(d.sum()/n.sum()-row.Delta),abs(lo-row.CI_low),abs(hi-row.CI_high));checks.append({'check':'archived_interval','origin':key,'mode':mode,'method':a+'-'+b,'metric':met+f'_L{l}','max_error':err,'pass':bool(err<1e-12)})
  for mode in ['CatBoost','Restricted']:
   for a,b in [('Expanding','Frozen'),('Expanding','Raw')]:
    for met in ['Brier','LogLoss']:
     vec=deltas[mode][a,b,met]-deltas['Core'][a,b,met]
     for l,w in W.items():
      lo,hi=np.quantile((w@vec)/(w@n),[.025,.975]);cross.append({'Origin':key,'Alternative':mode,'Reference':'Core','A':a,'B':b,'Metric':met,'BlockDays':l,'Delta':vec.sum()/n.sum(),'CI_low':lo,'CI_high':hi,'N':int(n.sum()),'Interpretation':'Exploratory paired difference of archived prediction-loss contrasts; not a refitting effect'})
  q=qs['Core'].copy();di=(q.t.dt.normalize()-start).dt.days.to_numpy();y=q.y.to_numpy(float)
  poolpath=root/f'Core_{key}_pools.csv';inputs[poolpath.name]=digest(poolpath);pool=pd.read_csv(poolpath,parse_dates=['Date','FitDate'])
  fpool=pool[pool.Method.eq('Frozen')];assert fpool.Intercept.nunique()==1
  bf=float(fpool.Intercept.iloc[0]);ep=pool[pool.Method.eq('Expanding')].set_index('Date')
  for d in range(60):
   mask=di==d;s=q[mask];nq=len(s);date=start+pd.Timedelta(days=d);pl=ep.loc[date]
   row={'Origin':key,'Date':str(date.date()),'DayIndex':d,'N':nq,'Positives':int(s.y.sum()),'ActualRate':s.y.mean(),'RawMean':s.Raw.mean(),'FrozenMean':s.Frozen.mean(),'ExpandingMean':s.Expanding.mean(),'FrozenIntercept':bf,'ExpandingIntercept':pl.Intercept,'PoolN':pl.N,'PoolPositives':pl.Positives,'PoolRate':pl.PoolRate,'MeanOrderAge':pl.MeanOrderAge}
   for a,b in PAIRS:
    for met in ['Brier','LogLoss']:
     ls=loss(s[a].to_numpy(),s.y.to_numpy(),met)-loss(s[b].to_numpy(),s.y.to_numpy(),met);row[f'{a}_{b}_{met}Sum']=ls.sum()
    ls=loss(s[a].to_numpy(),s.y.to_numpy(),'Brier')-loss(s[b].to_numpy(),s.y.to_numpy(),'Brier')
    row[f'{a}_{b}_OnTimeSum']=ls[s.y.to_numpy()==0].sum();row[f'{a}_{b}_LateSum']=ls[s.y.to_numpy()==1].sum()
   dailytraj.append(row)
  # Frozen scores for additional records: apply stored intercept, never refit.
  upath=root/'predictions'/f'Unknown_{key}.csv.gz';inputs[upath.name]=digest(upath)
  u=pd.read_csv(upath,parse_dates=['order_approved_at','order_delivered_customer_date']);p=np.clip(u.Raw.to_numpy(),1e-7,1-1e-7);u['Frozen']=expit(np.log(p)-np.log1p(-p)+bf)
  v=u[u.OriginalCohort].set_index('order_id').loc[q.order_id]
  err=float(np.max(np.abs(v.Frozen.to_numpy()-q.Frozen.to_numpy())));assert err<1e-12
  reconstruct.append({'Origin':key,'FrozenIntercept':bf,'OriginalN':len(q),'MaxFrozenPredictionDifference':err,'GeneratedAdditionalScores':len(u)-len(q),'ModelRefitted':False})
  horizon=pd.Timestamp('2018-10-17 17:30:18')
  for scope in ['Noncancelled','CancellationHypothetical']:
   sub=u if scope=='CancellationHypothetical' else u[~u.order_status.isin(['canceled','unavailable'])]
   prev=None
   for h in [0,30,60,90]:
    s=sub[(horizon-sub.order_approved_at).dt.total_seconds()/86400>=h].copy()
    known=s.y.notna() & s.order_delivered_customer_date.le(horizon) & ~s.order_status.isin(['canceled','unavailable']);k=known.to_numpy();ids='\n'.join(sorted(s.order_id.astype(str)));sid=hashlib.sha256(ids.encode()).hexdigest()
    cohorts.append({'Origin':key,'Scope':scope,'FollowupDays':h,'N':len(s),'Known':int(k.sum()),'Unknown':int((~k).sum()),'SetHash':sid,'IdenticalToPreviousThreshold':bool(prev==sid),'HorizonProxy':str(horizon),'Formula':'(HorizonProxy - approval_time) in elapsed days >= FollowupDays'})
    prev=sid
    for a,b in PAIRS:
     p=s[a].to_numpy();z=s[b].to_numpy();yy=s.y.to_numpy();actual=((p[k]-yy[k])**2-(z[k]-yy[k])**2).sum();d0=p[~k]**2-z[~k]**2;d1=(1-p[~k])**2-(1-z[~k])**2
     low=actual+np.minimum(d0,d1).sum();high=actual+np.maximum(d0,d1).sum()
     row={'Origin':key,'Scope':scope,'FollowupDays':h,'A':a,'B':b,'N':len(s),'Known':int(k.sum()),'Unknown':int((~k).sum()),'KnownLossDifferenceSum':actual,'UnknownLowerSum':np.minimum(d0,d1).sum(),'UnknownUpperSum':np.maximum(d0,d1).sum(),'BrierLower':low/len(s),'BrierUpper':high/len(s),'SetHash':sid,'BoundsAreCI':False}
     unknown.append(row)
     if a=='Expanding' and b=='Raw':
      old=pd.read_csv(root/'unknown_outcome_bounds.csv').query('Origin==@key and Scope==@scope and FollowupDays==@h').iloc[0];error=max(abs(row['BrierLower']-old.BrierLower),abs(row['BrierUpper']-old.BrierUpper));checks.append({'check':'unknown_existing','origin':key,'mode':scope,'method':str(h),'metric':'bounds','max_error':error,'pass':bool(error<1e-12)})
 d=pd.DataFrame(dailytraj)
 for key in ORIGINS:
  s=d[d.Origin.eq(key)].copy();s['Segment']=s.DayIndex//7+1
  for seg,g in s.groupby('Segment',sort=True):
   nn=g.N.sum();row={'Origin':key,'Segment':seg,'Start':g.Date.iloc[0],'EndInclusive':g.Date.iloc[-1],'Days':len(g),'N':nn,'Positives':g.Positives.sum(),'ActualRate':g.Positives.sum()/nn}
   for col in ['RawMean','FrozenMean','ExpandingMean','FrozenIntercept','ExpandingIntercept','PoolN','PoolRate','MeanOrderAge']:
    assert not g.loc[g.N.gt(0),col].isna().any()
    row[col]=(g[col].fillna(0)*g.N).sum()/nn
   for a,b in PAIRS:
    for met in ['Brier','LogLoss','OnTime','Late']:row[f'{a}_{b}_{met}']=g[f'{a}_{b}_{met}Sum'].sum()/nn
   weekly.append(row)
 save('backbone_absolute_scores.csv',absolute);save('daily_model_sufficient_statistics.csv',daily);save('cross_backbone_effects.csv',cross);save('daily_trajectory_with_frozen.csv',dailytraj);save('weekly_trajectory_with_frozen.csv',weekly);save('unknown_outcome_triplet_bounds.csv',unknown);save('unknown_followup_sets.csv',cohorts);save('reconstruction_checks.csv',checks)
 assert all(c['pass'] for c in checks)
 (out/'frozen_extended_reconstruction.json').write_text(json.dumps(reconstruct,indent=2),encoding='utf8')
 manifest={'seed':SEED,'replicates':2000,'block_days':[3,7,14],'scope':'Supplementary paired re-analysis of fixed archived predictions; no model fitting or retuning','new_analyses':['cross-backbone effect differences','Frozen extended-cohort bounds','Frozen time trajectories'],'existing_intervals_rechecked':sum(c['check']=='archived_interval' for c in checks),'checks_passed':len(checks),'max_existing_error':max(c['max_error'] for c in checks),'input_sha256':inputs}
 (out/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
 print(json.dumps({k:v for k,v in manifest.items() if k!='input_sha256'},indent=2))
if __name__=='__main__':main()
