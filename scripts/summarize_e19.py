from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.stats import t
from run_e19 import METHODS,POLICIES,SCORES,TERMS,POOL_FIELDS,SCENARIOS

PERIODS={"Full":slice(0,120),"Phase1":slice(0,60),"Phase2":slice(60,120)}
PAIRS=[("Expanding","Frozen"),("Frozen","Raw"),("Expanding","Raw"),("Rolling60","Expanding"),("PoolRate","Expanding")]
def stat(x):
    n=len(x);se=float(np.std(x,ddof=1)/np.sqrt(n));mu=float(np.mean(x));h=float(t.ppf(.975,n-1)*se)
    return {"Mean":mu,"SD":float(np.std(x,ddof=1)),"MCSE":se,"CI_low":mu-h,"CI_high":mu+h,"Sequences":n}

def main(out):
    dest=out/"E19";validation=json.loads((dest/"sequences/validation.json").read_text())
    assert validation["passed"] and validation["sequences"]==400
    absolute=[];paired=[];diagnostics=[];rep_scores=[];rep_diagnostics=[];rep_pairs=[];pool_summary=[];daily=[];daily_diag=[];metadata=[]
    all_pool_fallbacks=0
    for scenario in SCENARIOS:
        blobs=[np.load(dest/"sequences"/f"{scenario}_{r:03d}.npz") for r in range(100)]
        met=np.stack([b["metrics"] for b in blobs]);dia=np.stack([b["diagnostics"] for b in blobs]);pool=np.stack([b["pools"] for b in blobs])
        metadata.extend(json.loads(str(b["metadata"])) for b in blobs)
        all_pool_fallbacks+=int((pool[:,:,:,:,5]==0).sum())
        for pol,policy in enumerate(POLICIES):
            for period,sl in PERIODS.items():
                for m,method in enumerate(METHODS):
                    vals=met[:,pol,sl,m,:].mean(axis=1)
                    for r in range(100):
                        rep_scores.append({"Scenario":scenario,"Policy":policy,"Period":period,"Replicate":r,
                                           "Method":method,**dict(zip(SCORES,vals[r]))})
                    for j,metric in enumerate(SCORES):
                        absolute.append({"Scenario":scenario,"Policy":policy,"Period":period,"Method":method,
                                         "Metric":metric,**stat(vals[:,j])})
                for a,b in PAIRS:
                    vals=(met[:,pol,sl,METHODS.index(a),:]-met[:,pol,sl,METHODS.index(b),:]).mean(axis=1)
                    for r in range(100):
                        rep_pairs.append({"Scenario":scenario,"Policy":policy,"Period":period,"Replicate":r,
                                          "Updated":a,"Reference":b,**dict(zip(SCORES,vals[r]))})
                    for j,metric in enumerate(SCORES[:5]):
                        paired.append({"Scenario":scenario,"Policy":policy,"Period":period,"Updated":a,
                                       "Reference":b,"Metric":metric,**stat(vals[:,j])})
                for pair,ref in enumerate(["Frozen","Raw"]):
                    vals=dia[:,pol,sl,pair,:].mean(axis=1)
                    assert np.max(np.abs(vals[:,0]-vals[:,1]-vals[:,3]))<1e-12
                    assert np.max(np.abs(vals[:,0]-vals[:,2]-vals[:,4]))<1e-12
                    for r in range(100):
                        rep_diagnostics.append({"Scenario":scenario,"Policy":policy,"Period":period,"Replicate":r,
                                                "Updated":"Expanding","Reference":ref,**dict(zip(TERMS,vals[r]))})
                    for j,term in enumerate(TERMS):
                        diagnostics.append({"Scenario":scenario,"Policy":policy,"Period":period,
                                            "Updated":"Expanding","Reference":ref,"Term":term,**stat(vals[:,j])})
                for m,method in enumerate(["Frozen","Expanding","Rolling60"]):
                    vals=pool[:,pol,sl,m,:].mean(axis=1)
                    for j,field in enumerate(POOL_FIELDS):
                        pool_summary.append({"Scenario":scenario,"Policy":policy,"Period":period,
                                             "Method":method,"Field":field,**stat(vals[:,j])})
            for day in range(120):
                for m,method in enumerate(METHODS):
                    row={"Scenario":scenario,"Policy":policy,"Day":day+60,"Method":method}
                    for j,metric in enumerate(SCORES):row[metric]=float(met[:,pol,day,m,j].mean())
                    daily.append(row)
                row={"Scenario":scenario,"Policy":policy,"Day":day+60,"Reference":"Frozen"}
                for j,term in enumerate(TERMS):
                    s=stat(dia[:,pol,day,0,j])
                    row.update({term:s["Mean"],term+"_low":s["CI_low"],term+"_high":s["CI_high"]})
                row.update({"ExpandingPoolRate":float(pool[:,pol,day,1,2].mean()),
                            "ExpandingPoolAge":float(pool[:,pol,day,1,3].mean()),
                            "ExpandingIntercept":float(pool[:,pol,day,1,4].mean()),
                            "FrozenIntercept":float(pool[:,pol,day,0,4].mean())})
                daily_diag.append(row)
        for b in blobs:b.close()
    items={"absolute_scores":absolute,"paired_effects":paired,"decomposition":diagnostics,
           "replicate_scores":rep_scores,"replicate_decomposition":rep_diagnostics,"replicate_effects":rep_pairs,
           "pool_summary":pool_summary,"daily_scores":daily,"daily_decomposition":daily_diag,"sequence_metadata":metadata}
    for name,rows in items.items():pd.DataFrame(rows).to_csv(dest/f"{name}.csv",index=False,float_format="%.17g")
    summary={"passed":True,"independent_sequences":400,"scenario_policy_runs":1200,
             "generated_orders":400*180*500,"evaluation_orders_before_reusing_across_policies_methods":400*120*500,
             "support_fallback_pool_days":all_pool_fallbacks,
             "primary_endpoint":"Expanding-Frozen; observed and expected Brier",
             "unit":"independent sequence","interval":"t(99), 95% marginal, paired within sequence",
             "multiplicity":"No familywise significance or equivalence claims",
             "all_scenarios_reported":True,"parameters_tuned":False}
    (dest/"summary_validation.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True);main(ap.parse_args().out)
