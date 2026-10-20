import os
for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]:os.environ[k]="1"
from pathlib import Path
import json,hashlib,time,argparse
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from threadpoolctl import threadpool_limits
SRC=None
REF=None
def initialize_worker(predictions,reference):
    global SRC,REF
    SRC=Path(predictions);REF=Path(reference)
ORIGINS={"O1":"2017-07-01","O2":"2017-10-01","O3":"2018-01-01","O4":"2018-04-01"}

def weights(n,l):
    rng=np.random.default_rng(20261009)
    ix=(rng.integers(0,n-l+1,size=(2000,(n+l-1)//l))[:,:,None]+np.arange(l)).reshape(2000,-1)[:,:n]
    return np.array([np.bincount(x,minlength=n) for x in ix])

def one(job):
    threadpool_limits(1)
    backbone,origin=job
    q=pd.read_csv(SRC/f"{backbone}_{origin}.csv.gz",parse_dates=["t"])
    assert q.order_id.is_unique and q[["Raw","Frozen","Expanding","y"]].notna().all().all()
    start=pd.Timestamp(ORIGINS[origin])+pd.Timedelta(days=60)
    day=((q.t.dt.normalize()-start).dt.days).to_numpy()
    assert ((day>=0)&(day<60)).all()
    y=q.y.to_numpy();n=len(y)
    ref=pd.read_csv(REF/"core_triplets_all_backbones_blocks.csv")
    sums=[];weekly=[];groups=[];intervals=[];checks=[];daily=[]
    for a,b in [("Expanding","Frozen"),("Expanding","Raw")]:
        e=q[a].to_numpy();f=q[b].to_numpy();delta=e-f
        v={"P":delta**2,"A":2*delta*(y-f),"Delta":(e-y)**2-(f-y)**2,
           "BrierReference":(f-y)**2,"BrierUpdated":(e-y)**2}
        ec=np.clip(e,1e-7,1-1e-7);fc=np.clip(f,1e-7,1-1e-7)
        v["LogLossDelta"]=-(y*np.log(ec)+(1-y)*np.log1p(-ec))+(y*np.log(fc)+(1-y)*np.log1p(-fc))
        err=float(np.max(np.abs(v["P"]-v["A"]-v["Delta"])))
        assert err<1e-12
        meta={"Backbone":backbone,"Origin":origin,"Updated":a,"Reference":b}
        row={**meta,"N":n,**{k:float(x.mean()) for k,x in v.items()},"IdentityError":err}
        sums.append(row)
        counts=np.bincount(day,minlength=60)
        ds={k:np.bincount(day,weights=x,minlength=60) for k,x in v.items()}
        for d in range(60):
            daily.append({**meta,"DayIndex":d,"Date":str((start+pd.Timedelta(days=d)).date()),"N":int(counts[d]),
                          **{k+"Sum":float(x[d]) for k,x in ds.items()}})
        for week in range(9):
            mask=day//7==week
            if not mask.any():continue
            weekly.append({**meta,"Week":week+1,"Start":str((start+pd.Timedelta(days=week*7)).date()),
                           "CalendarDays":min(7,60-week*7),"N":int(mask.sum()),
                           **{k:float(x[mask].mean()) for k,x in v.items()}})
        for label in [0,1]:
            mask=y==label
            groups.append({**meta,"Outcome":label,"GroupN":int(mask.sum()),"TotalN":n,
                           **{k:float(x[mask].sum()/n) for k,x in v.items()}})
        for block in [3,7,14]:
            w=weights(60,block);den=w@counts
            r={**meta,"BlockDays":block,"N":n}
            for k,x in ds.items():
                vals=w@x/den;lo,hi=np.quantile(vals,[.025,.975])
                r.update({k:row[k],k+"_low":lo,k+"_high":hi})
            for met,key in [("Brier","Delta"),("LogLoss","LogLossDelta")]:
                rr=ref.query("Backbone==@backbone and Origin==@origin and A==@a and B==@b and Metric==@met and BlockDays==@block").iloc[0]
                er=max(abs(r[key]-rr.Delta),abs(r[key+"_low"]-rr.CI_low),abs(r[key+"_high"]-rr.CI_high))
                assert er<1e-11,(meta,block,key,er)
                checks.append({**meta,"BlockDays":block,"Metric":met,"MaxHistoricalError":er,"IdentityError":err})
            assert np.max(np.abs(w@(ds["P"]-ds["A"]-ds["Delta"])/den))<1e-12
            intervals.append(r)
    return sums,weekly,groups,intervals,checks,daily

if __name__=="__main__":
    start=time.perf_counter()
    ap=argparse.ArgumentParser(description="Saved-score loss diagnosis; requires separately held private predictions.")
    ap.add_argument("--predictions-dir",type=Path,required=True)
    ap.add_argument("--reference-dir",type=Path,required=True,help="Directory containing core_triplets_all_backbones_blocks.csv")
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--workers",type=int,default=min(12,os.cpu_count() or 1))
    args=ap.parse_args();OUT=args.out
    if OUT.exists() and any(OUT.iterdir()):ap.error("Use a new or empty output directory.")
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"code_lock.json").write_text(json.dumps({"sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}))
    jobs=[(b,o) for b in ["Core","Restricted","CatBoost"] for o in ORIGINS]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize_worker,initargs=(str(args.predictions_dir.resolve()),str(args.reference_dir.resolve()))) as ex:res=list(ex.map(one,jobs))
    names=["decomposition","weekly","outcome_contributions","paired_intervals","validation_checks","daily_sufficient_statistics"]
    for i,name in enumerate(names):
        pd.DataFrame([r for job in res for r in job[i]]).to_csv(OUT/f"{name}.csv",index=False,float_format="%.17g")
    check=pd.read_csv(OUT/"validation_checks.csv")
    result={"passed":True,"prediction_files":12,"paired_analyses":24,"archived_score_interval_checks":len(check),
            "max_archived_error":float(check.MaxHistoricalError.max()),"max_identity_error":float(check.IdentityError.max()),
            "workers":args.workers,"library_threads":1,"elapsed_seconds":time.perf_counter()-start,
            "scope":"Saved predictions only; no fits; realized-loss diagnostics"}
    (OUT/"validation.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
