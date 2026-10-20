"""Independent aggregate/array verification; no private Olist records required."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from scipy.stats import t
from scipy.special import expit
from numpy.polynomial.hermite import hermgauss

def verify(out):
    checks=0;maxerr=0.
    def close(a,b,tol=1e-11):
        nonlocal checks,maxerr
        err=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
        assert err<tol,err
        checks+=1;maxerr=max(maxerr,err)
    ep=pd.read_csv(out/"E18/paired_intervals.csv")
    ed=pd.read_csv(out/"E18/daily_sufficient_statistics.csv")
    groups=pd.read_csv(out/"E18/outcome_contributions.csv")
    total=pd.read_csv(out/"E18/decomposition.csv")
    keys=["Backbone","Origin","Updated","Reference"]
    for key,g in ed.groupby(keys):
        g=g.sort_values("DayIndex");sel=np.ones(len(ep),bool)
        for k,v in zip(keys,key):sel&=ep[k].eq(v)
        sub=ep[sel]
        close(g.PSum-g.ASum,g.DeltaSum)
        for block in [3,7,14]:
            rng=np.random.default_rng(20261009)
            starts=rng.integers(0,60-block+1,size=(2000,(60+block-1)//block))
            idx=(starts[:,:,None]+np.arange(block)).reshape(2000,-1)[:,:60]
            # Direct indexing, independently of the matrix-weight implementation.
            den=g.N.to_numpy()[idx].sum(axis=1);row=sub[sub.BlockDays.eq(block)].iloc[0]
            for term in ["P","A","Delta","BrierReference","BrierUpdated","LogLossDelta"]:
                vals=g[term+"Sum"].to_numpy()
                ci=np.quantile(vals[idx].sum(axis=1)/den,[.025,.975])
                close([vals.sum()/g.N.sum(),*ci],[row[term],row[term+"_low"],row[term+"_high"]])
    for key,g in groups.groupby(keys):
        sel=np.ones(len(total),bool)
        for k,v in zip(keys,key):sel&=total[k].eq(v)
        row=total[sel].iloc[0]
        close(g[["P","A","Delta"]].sum().to_numpy(),row[["P","A","Delta"]].to_numpy(float))
    ef=pd.read_csv(out/"E19/paired_effects.csv")
    scores=["Brier","LogLoss","ExpectedBrier","ExpectedLogLoss","ProbabilityMSE","MeanPred","ActualRate"]
    methods=["Raw","Frozen","Expanding","Rolling60","PoolRate"]
    policies=["Immediate","Observed","Closed"]
    periods={"Full":slice(None),"Phase1":slice(0,60),"Phase2":slice(60,120)}
    for scenario in ["Correct","FixedBias","Reversal","Shape"]:
        ms=[];di=[]
        for i in range(100):
            with np.load(out/"E19/sequences"/f"{scenario}_{i:03d}.npz") as d:
                m=d["metrics"];a=d["diagnostics"];p=d["pools"]
                meta=json.loads(str(d["metadata"]))
                assert meta["event_and_boundary_checks"] and meta["closed_invariance"]
                close(m[0,:,0],m[1,:,0]);close(m[1,:,0],m[2,:,0])
                close(m[:,0,1],m[:,0,2])
                close(a[:,:,:,0]-a[:,:,:,1],a[:,:,:,3])
                close(a[:,:,:,0]-a[:,:,:,2],a[:,:,:,4])
                for j,ref in enumerate([1,0]):
                    close(a[:,:,j,3],m[:,:,2,0]-m[:,:,ref,0])
                    close(a[:,:,j,4],m[:,:,2,2]-m[:,:,ref,2])
                # ExpectedBrier - ProbabilityMSE is the same irreducible noise across methods.
                noise=m[:,:,:,2]-m[:,:,:,4]
                close(noise,np.broadcast_to(noise[:,:,0,None],noise.shape))
                assert (p[:,:,:,5]==1).all()
                if scenario=="Correct":
                    close(m[:,:,0,4],0.)
                    assert (m[:,:,:,2]-m[:,:,0,None,2]>=-1e-14).all()
                if scenario=="Shape":
                    nodes,w=hermgauss(200);z=-2.5+np.sqrt(2)*nodes;w=w/np.sqrt(np.pi)
                    rates=[w@expit(b*z+c) for b,c in zip(meta["beta"],meta["intercept"])]
                    close(rates,np.repeat(w@expit(z),3),tol=1e-10)
                ms.append(m)
        ms=np.stack(ms)
        for _,row in ef[ef.Scenario.eq(scenario)].iterrows():
            p=policies.index(row.Policy);sl=periods[row.Period]
            a=methods.index(row.Updated);b=methods.index(row.Reference);j=scores.index(row.Metric)
            x=(ms[:,p,sl,a,j]-ms[:,p,sl,b,j]).mean(axis=1)
            se=x.std(ddof=1)/10;delta=t.ppf(.975,99)*se
            close([x.mean(),se,x.mean()-delta,x.mean()+delta],
                  [row.Mean,row.MCSE,row.CI_low,row.CI_high])
    result={"passed":True,"checks":checks,"max_numeric_difference":maxerr,
            "E18_scope":"All decomposition terms, date-block intervals, and outcome additivity from public daily sums",
            "E19_scope":"All 400 saved sequence arrays; 900 paired-effect rows; identities, irreducible risk, invariant policies, constant shape prevalence",
            "independent_implementation":True,"independent_person":False}
    (out/"public_verification.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True);verify(ap.parse_args().out)
