"""Independent aggregate/array verification; no private Olist records required."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from scipy.stats import t
from scipy.special import expit
from numpy.polynomial.hermite import hermgauss

def verify_e18(out):
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
    return {"checks":checks,"max_numeric_difference":maxerr}
