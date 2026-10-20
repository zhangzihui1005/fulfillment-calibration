"""Portable extraction of the archived E17 calibration functions.
Function bodies are copied without numerical changes; only I/O and orchestration
are separate. No pickle, model deserialization, training or network access occurs.
See docs/E17_PORTABILITY.md for provenance and validation scope.
"""
import hashlib
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import brentq, minimize, minimize_scalar
ORIGINS={'O1':'2017-07-01','O2':'2017-10-01','O3':'2018-01-01','O4':'2018-04-01'}
def logit(p):p=np.clip(np.asarray(p,float),1e-7,1-1e-7);return np.log(p/(1-p))

def intercept(z,y):
    c=np.log(y.mean()/(1-y.mean()));return brentq(lambda b:expit(z+b).mean()-y.mean(),c-z.max()-1,c-z.min()+1,xtol=1e-12,rtol=1e-12)

def solve_platt(z,y,b):
    def fg(x):
        v=x[0]+x[1]*z;q=expit(v);r=q-y
        return np.mean(np.logaddexp(0,v)-y*v),np.array([r.mean(),np.mean(r*z)])
    fit=minimize(fg,[b,1.],jac=True,method='L-BFGS-B',bounds=[(-30,30),(0,10)],options={'ftol':1e-13,'gtol':1e-9,'maxiter':500,'maxls':40})
    return fit

def solve_brier(z,y,b):
    def loss(b):return np.mean((expit(z+b)-y)**2)
    def grad(b):q=expit(z+b);return np.mean(2*(q-y)*q*(1-q))
    grid=np.linspace(-30,30,241);vals=np.array([loss(x) for x in grid]);gr=np.array([grad(x) for x in grid]);candidates=[-30.,30.,float(np.clip(b,-30,30))]
    for i in range(len(grid)-1):
        if gr[i]*gr[i+1]<0:candidates.append(brentq(grad,grid[i],grid[i+1],xtol=1e-11))
    for i in np.flatnonzero((vals[1:-1]<vals[:-2])&(vals[1:-1]<vals[2:]))+1:
        opt=minimize_scalar(loss,bounds=(grid[i-1],grid[i+1]),method='bounded',options={'xatol':1e-10});candidates.append(opt.x)
    best=min(candidates,key=loss);assert loss(best)<=loss(b)+1e-12 or abs(b)>30
    return best,grad(best),len(candidates),loss(best),loss(b)

def replay(h,key,methods):
    fit=pd.Timestamp(ORIGINS[key]);start=fit+pd.Timedelta(days=60);end=start+pd.Timedelta(days=60)
    h=h.sort_values(['t','order_id']).reset_index(drop=True);assert h.t.min()>=fit
    q=h[h.t.ge(start)&h.t.lt(end)].copy().reset_index(drop=True);q['Raw']=q.raw;z=logit(h.raw);zq=logit(q.raw);logs=[];cached={}
    for m in methods:q[m]=np.nan
    for t in pd.date_range(start,end,inclusive='left'):
        target=(q.t>=t)&(q.t<t+pd.Timedelta(days=1))
        for m in methods:
            cutoff=start if m=='Frozen' else t;mask=(h.t<cutoff)&(h.feedback<cutoff)&(h.eligibility<cutoff)
            if m=='Rolling60':mask&=h.eligibility>=cutoff-pd.Timedelta(days=60)
            if m=='Approval60':mask&=h.t>=cutoff-pd.Timedelta(days=60)
            pool=h[mask];y=pool.y.to_numpy(float);zz=z[mask];n=len(y);pos=int(y.sum());active=n>=1000 and pos>=50 and n-pos>=50
            poolhash=hashlib.sha256('\n'.join(pool.order_id.astype(str)).encode()).hexdigest();cachekey=(poolhash,m)
            if cachekey in cached:a,beta,solver,brierextra=cached[cachekey]
            else:
                a=intercept(zz,y) if active else 0.;beta=1.;solver='OK' if active else 'GuardRaw';brierextra={}
                if active and m=='PoolRate':a=np.log(y.mean()/(1-y.mean()));beta=0.
                elif active and m=='Platt':
                    opt=solve_platt(zz,y,a)
                    if opt.success and np.isfinite(opt.x).all():a,beta=opt.x
                    else:solver='SolverFallbackToLLIntercept'
                    brierextra={'SolverMessage':str(opt.message),'Iterations':int(opt.nit)}
                elif active and m=='BrierOpt':
                    a,g,count,best,ll=solve_brier(zz,y,a);brierextra={'Gradient':g,'CandidateCount':count,'HistoricalBrier':best,'HistoricalLLInterceptBrier':ll}
                cached[cachekey]=(a,beta,solver,brierextra)
            q.loc[target,m]=expit(a+beta*zq[target]) if active else q.loc[target,'Raw']
            logs.append({'OriginKey':key,'Date':t,'Method':m,'FitDate':cutoff,'N':n,'Positives':pos,'PoolRate':y.mean() if n else np.nan,'NQuery':int(target.sum()),'Intercept':a,'Slope':beta,'Active':active,'Solver':solver,'AtBoundary':bool(abs(a)>=29.999 or (m=='Platt' and (beta<=1e-7 or beta>=9.999999))),'MeanOrderAge':(t-pool.t).dt.total_seconds().mean()/86400,'PoolHash':poolhash,**brierextra})
    assert np.isfinite(q[['Raw',*methods]]).all().all();return q,pd.DataFrame(logs)
