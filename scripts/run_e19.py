"""Protocol-fixed simulation. No parameter search or outcome-dependent scenario selection."""
import os
for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]:os.environ[k]="1"
from pathlib import Path
import argparse,json,time,hashlib,platform
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.special import expit
from scipy.optimize import brentq
from numpy.polynomial.hermite import hermgauss
from threadpoolctl import threadpool_limits

METHODS=["Raw","Frozen","Expanding","Rolling60","PoolRate"]
POLICIES=["Immediate","Observed","Closed"]
SCORES=["Brier","LogLoss","ExpectedBrier","ExpectedLogLoss","ProbabilityMSE","MeanPred","ActualRate"]
TERMS=["P","A_realized","A_probability","Delta_realized","Delta_expected"]
POOL_FIELDS=["N","Positives","PositiveRate","MeanOrderAge","Intercept","Active","MeanHistoricalPi","MeanCurrentPi","NewtonIterations","BrentFallback"]
SCENARIOS=["Correct","FixedBias","Reversal","Shape"]

def solve(z,y,b0=0.):
    n=len(y);pos=int(y.sum())
    if n<1000 or min(pos,n-pos)<50:return 0.,False,0,False,0.
    rate=pos/n;b=float(b0)
    for it in range(1,31):
        p=expit(z+b);res=float(p.mean()-rate)
        if abs(res)<=1e-12:return b,True,it,False,abs(res)
        grad=float(np.mean(p*(1-p)))
        if grad<1e-14:break
        b-=np.clip(res/grad,-5,5)
    c=np.log(rate/(1-rate))
    b=brentq(lambda x:expit(z+x).mean()-rate,c-z.max()-1,c-z.min()+1,xtol=1e-12,rtol=1e-12)
    return b,True,it,True,abs(float(expit(z+b).mean()-rate))

def population_parameters(cfg):
    nodes,w=hermgauss(100);zz=cfg["base_scores"]["mean"]+np.sqrt(2)*cfg["base_scores"]["sd"]*nodes
    w=w/np.sqrt(np.pi);base=float(w@expit(zz));params=[]
    for s in cfg["scenarios"]:
        beta=np.array(s["beta"],float)
        cs=np.array(s["c"],float) if isinstance(s["c"],list) else np.array([
            brentq(lambda c:float(w@expit(b*zz+c))-base,-10,10,xtol=1e-14) for b in beta])
        means=np.array([w@expit(b*zz+c) for b,c in zip(beta,cs)])
        if s["name"]=="Shape":assert np.max(np.abs(means-base))<1e-12
        params.append((beta,cs,means))
    return params

def run_sequence(job):
    cfg,si,rep,out_dir,technical=job
    threadpool_limits(1);tick=time.perf_counter()
    rng=np.random.default_rng(np.random.SeedSequence([cfg["seed_entropy"],si,rep]))
    days=cfg["days"];per=cfg["orders_per_day"];n=days*per
    a=(np.arange(days)[:,None]+np.sort(rng.random((days,per)),axis=1)).ravel()
    z=rng.normal(cfg["base_scores"]["mean"],cfg["base_scores"]["sd"],n)
    promise=rng.choice(cfg["promise_days"],n,p=cfg["promise_probabilities"])
    d=np.floor(a)+promise+1
    beta,cs,pop=population_parameters(cfg)[si]
    phase=np.minimum((a//60).astype(int),2)
    pi=expit(beta[phase]*z+cs[phase])
    y=(rng.random(n)<pi).astype(np.int8)
    uniform=rng.uniform(.05,.95,n);late=rng.uniform(.1,7,n)
    v=np.where(y==0,a+uniform*(d-a),d+late)
    r=np.minimum(v,d);closed=np.maximum(r,d)
    assert np.array_equal((v>d).astype(np.int8),y)
    assert np.array_equal(closed,d) and np.all(v>a) and np.all(r>a)
    earlier=np.where(y==0,a+0.5*(v-a),r)
    alternate_closed=np.maximum(earlier,d)
    assert np.array_equal(alternate_closed,closed)
    raw=expit(z);metrics=np.zeros((3,120,5,7));diag=np.zeros((3,120,2,5))
    pools=np.zeros((3,120,3,10))
    total_brent=0;max_res=0.;max_identity=0.;max_brent_difference=0.;checked_roots=0
    eligibility=[a,r,closed]
    input_hash=hashlib.sha256()
    for arr in [a,z,d,pi,y,v]:input_hash.update(arr.tobytes())
    for policy,elig in enumerate(eligibility):
        ix=np.argsort(elig,kind="stable");es=elig[ix];zs=z[ix];ys=y[ix];aps=a[ix];pis=pi[ix]
        f_end=int(np.searchsorted(es,60,side="left"))
        f_res=solve(zs[:f_end],ys[:f_end]);last=[f_res[0],f_res[0]]
        for day in range(60,180):
            k=day-60;target=slice(day*per,(day+1)*per);zq=z[target];yq=y[target];pq=pi[target]
            end=int(np.searchsorted(es,day,side="left"))
            begin=int(np.searchsorted(es,day-60,side="left"))
            bounds=[(0,f_end),(0,end),(begin,end)]
            fits=[f_res,solve(zs[:end],ys[:end],last[0]),solve(zs[begin:end],ys[begin:end],last[1])]
            last=[fits[1][0],fits[2][0]]
            q=np.stack([raw[target],expit(zq+fits[0][0]) if fits[0][1] else raw[target],
                        expit(zq+fits[1][0]) if fits[1][1] else raw[target],
                        expit(zq+fits[2][0]) if fits[2][1] else raw[target],
                        np.full(per,float(ys[:end].mean())) if fits[1][1] else raw[target]],axis=0)
            assert np.isfinite(q).all()
            if day==60:assert np.max(np.abs(q[1]-q[2]))<1e-12
            for m,(lo,hi) in enumerate(bounds):
                zz=zs[lo:hi];yy=ys[lo:hi];b,active,it,br,res=fits[m]
                fitday=60 if m==0 else day
                if hi>lo:assert es[hi-1]<fitday and np.max(aps[lo:hi])<fitday
                assert hi-lo==len(yy)
                if m>0 or day==60:total_brent+=int(br);max_res=max(max_res,res)
                ph=day//60
                pools[policy,k,m]=[len(yy),yy.sum(),yy.mean() if len(yy) else np.nan,
                    day-aps[lo:hi].mean() if len(yy) else np.nan,b,int(active),
                    pis[lo:hi].mean() if len(yy) else np.nan,
                    expit(beta[ph]*zz+cs[ph]).mean() if len(yy) else np.nan,it,int(br)]
                # Technical runs compare independent solver at prespecified boundary days.
                if technical and day in [60,119,120,179] and active:
                    rate=yy.mean();c=np.log(rate/(1-rate))
                    oracle_b=brentq(lambda x:expit(zz+x).mean()-rate,c-zz.max()-1,c-zz.min()+1,xtol=1e-13)
                    max_brent_difference=max(max_brent_difference,abs(oracle_b-b));checked_roots+=1
                    assert abs(oracle_b-b)<1e-9
            qq=np.clip(q,1e-7,1-1e-7)
            real=(q-yq)**2;mse=(q-pq)**2;expected=mse+pq*(1-pq)
            values=[real.mean(axis=1),-(yq*np.log(qq)+(1-yq)*np.log1p(-qq)).mean(axis=1),
                    expected.mean(axis=1),-(pq*np.log(qq)+(1-pq)*np.log1p(-qq)).mean(axis=1),
                    mse.mean(axis=1),q.mean(axis=1),np.repeat(yq.mean(),5)]
            metrics[policy,k]=np.stack(values,axis=1)
            for pair,ref in enumerate([1,0]):
                delta=q[2]-q[ref];P=float(np.mean(delta**2))
                ar=float(np.mean(2*delta*(yq-q[ref])));ap=float(np.mean(2*delta*(pq-q[ref])))
                dr=float(np.mean(real[2]-real[ref]));de=float(np.mean(expected[2]-expected[ref]))
                err=max(abs(P-ar-dr),abs(P-ap-de));max_identity=max(max_identity,err)
                assert err<1e-12
                diag[policy,k,pair]=[P,ar,ap,dr,de]
            if policy==2 and technical:
                # Equality of the actual closed pool membership under earlier on-time observations.
                alternate_mask=(a<day)&(alternate_closed<day)
                primary_mask=(a<day)&(closed<day)
                assert np.array_equal(alternate_mask,primary_mask)
                if day in [60,120,179]:
                    zz=z[alternate_mask];yy=y[alternate_mask]
                    alt=solve(zz,yy,fits[1][0])
                    assert np.max(np.abs(expit(zq+alt[0])-q[2]))<1e-11
    assert np.max(np.abs(metrics[0,:,0]-metrics[1,:,0]))==0
    assert np.max(np.abs(metrics[0,:,0]-metrics[2,:,0]))==0
    result={"scenario":SCENARIOS[si],"replicate":rep,"seconds":time.perf_counter()-tick,
            "input_sha256":input_hash.hexdigest(),"orders":n,"brent_fallbacks":total_brent,
            "max_solver_residual":max_res,"max_identity_error":max_identity,
            "technical_root_checks":checked_roots,"max_independent_root_difference":max_brent_difference,
            "event_and_boundary_checks":True,"closed_invariance":True,"raw_policy_invariance":True,
            "population_event_rates":pop.tolist(),"beta":beta.tolist(),"intercept":cs.tolist()}
    dest=Path(out_dir);dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/f"{SCENARIOS[si]}_{rep:03d}.npz",metrics=metrics,diagnostics=diag,pools=pools,
                        metadata=np.array(json.dumps(result)))
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True);ap.add_argument("--technical",action="store_true")
    ap.add_argument("--workers",type=int,default=min(20,os.cpu_count() or 1))
    ap.add_argument("--technical-attempt",type=int,default=2)
    ap.add_argument("--check-only",action="store_true",help="Validate the formal execution plan without generating sequences or creating output files.")
    args=ap.parse_args()
    out=args.out.resolve();protocol=json.loads((out/"protocol.json").read_text(encoding="utf-8"));cfg=protocol["E19"]
    ph=hashlib.sha256((out/"protocol.json").read_bytes()).hexdigest()
    assert ph==json.loads((out/"protocol_lock.json").read_text())["sha256"]
    destination=out/"E19"/(f"technical_{args.technical_attempt:02d}" if args.technical else "sequences")
    assert not destination.exists(),"Fresh execution directory required"
    jobs=[(cfg,s,r,str(destination),args.technical) for s in range(4)
          for r in ([10000] if args.technical else range(cfg["replicates"]))]
    if args.workers<1:ap.error("--workers must be positive")
    if args.check_only:
        population_parameters(cfg)
        print(json.dumps({"passed":True,"mode":"technical" if args.technical else "formal",
                          "planned_sequences":len(jobs),"workers":args.workers,"protocol_sha256":ph,
                          "no_sequences_generated":True,"destination_exists":destination.exists()},indent=2))
        return
    destination.mkdir(parents=True,exist_ok=False)
    lock={"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "protocol_sha256":ph,"workers":args.workers,"library_threads":1,
          "python":platform.python_version(),"technical":args.technical}
    (destination/"run_lock.json").write_text(json.dumps(lock,indent=2))
    start=time.perf_counter();results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        fs=[ex.submit(run_sequence,j) for j in jobs]
        for f in as_completed(fs):
            results.append(f.result())
            if len(results)%20==0 or args.technical or len(results)==len(jobs):
                print(json.dumps({"completed":len(results),"total":len(jobs),
                                  "elapsed_seconds":round(time.perf_counter()-start,2)}),flush=True)
    result={"passed":True,"sequences":len(results),"elapsed_seconds":time.perf_counter()-start,
            "max_solver_residual":max(r["max_solver_residual"] for r in results),
            "max_identity_error":max(r["max_identity_error"] for r in results),
            "brent_fallbacks":sum(r["brent_fallbacks"] for r in results),
            "technical_root_checks":sum(r["technical_root_checks"] for r in results),
            "max_independent_root_difference":max(r["max_independent_root_difference"] for r in results),
            **lock}
    (destination/"validation.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
if __name__=="__main__":main()
