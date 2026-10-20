#!/usr/bin/env python3
"""Reproducible exploratory Olist audit and approval-time risk baseline.

Reads the original ZIP without extracting or modifying it. Outputs are
aggregate statistics and a local prediction file. This is an exploratory
study on the eventually-delivered cohort, not a completed journal result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from threadpoolctl import threadpool_limits

SEED = 20261009
FIT_ASOF = pd.Timestamp("2018-01-01")
CAL_START = pd.Timestamp("2018-01-01")
CAL_END = pd.Timestamp("2018-02-01")
CAL_ASOF = pd.Timestamp("2018-03-01")
DEV_START = pd.Timestamp("2018-03-01")
DEV_END = pd.Timestamp("2018-04-01")
TIME_COLS = [
    "order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_delivered_customer_date",
    "order_estimated_delivery_date",
]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_tables(zip_path):
    names = {
        "orders": "olist_orders_dataset.csv",
        "items": "olist_order_items_dataset.csv",
        "products": "olist_products_dataset.csv",
        "customers": "olist_customers_dataset.csv",
        "sellers": "olist_sellers_dataset.csv",
    }
    source = Path(zip_path)
    if source.is_dir():
        tables = {name: pd.read_csv(source/filename) for name,filename in names.items()}
    else:
        with ZipFile(source) as z:
            tables = {name: pd.read_csv(z.open(filename)) for name, filename in names.items()}
    for c in TIME_COLS:
        tables["orders"][c] = pd.to_datetime(tables["orders"][c], errors="coerce")
    assert tables["orders"]["order_id"].is_unique
    assert tables["items"][["order_id", "order_item_id"]].duplicated().sum() == 0
    return tables


def make_cohort(orders, feedback_mode="delivered"):
    flow = [{"step": "all_orders", "remaining": len(orders), "removed_at_step": 0}]
    d = orders.copy()
    required = ["order_purchase_timestamp", "order_approved_at",
                "order_delivered_customer_date", "order_estimated_delivery_date"]
    conditions = [
        ("final_status_delivered", lambda x: x.order_status.eq("delivered")),
        ("required_dates_complete", lambda x: x[required].notna().all(axis=1)),
        ("approval_not_before_purchase", lambda x: x.order_approved_at.ge(x.order_purchase_timestamp)),
        ("delivery_not_before_approval", lambda x: x.order_delivered_customer_date.ge(x.order_approved_at)),
        ("promise_day_not_before_approval_day", lambda x: x.order_estimated_delivery_date.dt.normalize().ge(x.order_approved_at.dt.normalize())),
    ]
    for name, fn in conditions:
        before = len(d)
        d = d.loc[fn(d)].copy()
        flow.append({"step": name, "remaining": len(d), "removed_at_step": before-len(d)})
    d["y"] = (d.order_delivered_customer_date.dt.normalize() >
              d.order_estimated_delivery_date.dt.normalize()).astype("int8")
    d["y_timestamp"] = (d.order_delivered_customer_date > d.order_estimated_delivery_date).astype("int8")
    d["approved_month"] = d.order_approved_at.dt.to_period("M").astype(str)
    d["approval_day"] = d.order_approved_at.dt.normalize()
    d["delivery_days"] = (d.order_delivered_customer_date-d.order_approved_at).dt.total_seconds()/86400
    d["promise_days"] = (d.order_estimated_delivery_date.dt.normalize()-d.order_approved_at.dt.normalize()).dt.days
    d["label_observed_at"] = d.order_delivered_customer_date
    if feedback_mode == "deadline":
        deadline = d.order_estimated_delivery_date.dt.normalize()+pd.Timedelta(days=1)
        d["label_observed_at"] = pd.concat([d.order_delivered_customer_date,deadline],axis=1).min(axis=1)
    assert d.label_observed_at.ge(d.order_approved_at).all()
    return d, pd.DataFrame(flow)


def historical_seller_features(orders, items, cohort):
    """Use only prior approval events and delivery labels observed before t.

    Order-seller membership is deduplicated. Histories do not use a full-data
    target mean, and the current order can never supply its own outcome.
    Label availability follows the explicitly selected feedback scenario.
    """
    membership = items[["order_id", "seller_id"]].drop_duplicates()
    all_q = membership.merge(orders[["order_id", "order_approved_at"]], on="order_id", validate="many_to_one")
    all_q = all_q.dropna(subset=["order_approved_at"])
    query = membership.merge(cohort[["order_id", "order_approved_at"]], on="order_id", validate="many_to_one")
    events = membership.merge(cohort[["order_id", "label_observed_at", "y"]], on="order_id", validate="many_to_one")
    approvals_by_seller = {k: np.sort(g.order_approved_at.astype("int64").to_numpy()) for k, g in all_q.groupby("seller_id", sort=False)}
    events_by_seller = {k: g.sort_values("label_observed_at") for k, g in events.groupby("seller_id", sort=False)}
    parts = []
    for seller_id, q in query.groupby("seller_id", sort=False):
        qt = q.order_approved_at.astype("int64").to_numpy()
        ap = approvals_by_seller[seller_id]
        n_prior = np.searchsorted(ap, qt, side="left")
        ev = events_by_seller.get(seller_id)
        if ev is None:
            n_seen = np.zeros(len(q), dtype=int)
            n_late = np.zeros(len(q), dtype=int)
        else:
            et = ev.label_observed_at.astype("int64").to_numpy()
            n_seen = np.searchsorted(et, qt, side="left")
            sums = np.r_[0, ev.y.to_numpy().cumsum()]
            n_late = sums[n_seen]
            has = n_seen > 0
            if has.any():
                assert np.all(et[n_seen[has]-1] < qt[has])
        x = q[["order_id"]].copy()
        x["seller_prior_orders"] = n_prior
        x["seller_observed_outcomes"] = n_seen
        # Fixed, prespecified Beta(1,19) smoothing; no estimated future prior.
        x["seller_past_late_rate"] = (n_late+1)/(n_seen+20)
        parts.append(x)
    h = pd.concat(parts, ignore_index=True).groupby("order_id").agg(
        seller_prior_orders_min=("seller_prior_orders", "min"),
        seller_prior_orders_mean=("seller_prior_orders", "mean"),
        seller_observed_outcomes_min=("seller_observed_outcomes", "min"),
        seller_observed_outcomes_mean=("seller_observed_outcomes", "mean"),
        seller_past_late_rate_max=("seller_past_late_rate", "max"),
        seller_past_late_rate_mean=("seller_past_late_rate", "mean"),
    ).reset_index()
    return h


def make_features(tables, cohort):
    items = tables["items"].merge(tables["products"], on="product_id", how="left", validate="many_to_one")
    items = items.merge(tables["sellers"][["seller_id", "seller_state"]], on="seller_id", how="left", validate="many_to_one")
    items["volume_cm3"] = items.product_length_cm*items.product_height_cm*items.product_width_cm
    agg = items.groupby("order_id").agg(
        item_count=("order_item_id", "size"),
        product_count=("product_id", "nunique"),
        seller_count=("seller_id", "nunique"),
        item_value=("price", "sum"),
        freight_value=("freight_value", "sum"),
        weight_g_mean=("product_weight_g", "mean"),
        volume_cm3_mean=("volume_cm3", "mean"),
        product_photos_mean=("product_photos_qty", "mean"),
        product_description_length_mean=("product_description_lenght", "mean"),
    ).reset_index()
    # Representative category/state of largest-price item. All numeric seller
    # risk summaries above include every seller; no arbitrary first-row choice.
    main = items.sort_values(["order_id", "price", "order_item_id"], ascending=[True, False, True]).drop_duplicates("order_id")
    main = main[["order_id", "product_category_name", "seller_state"]].rename(columns={"seller_state":"largest_item_seller_state"})
    d = cohort.merge(agg, on="order_id", how="left", validate="one_to_one")
    d = d.merge(main, on="order_id", how="left", validate="one_to_one")
    d = d.merge(tables["customers"][["customer_id", "customer_state"]], on="customer_id", how="left", validate="many_to_one")
    h = historical_seller_features(tables["orders"], tables["items"], cohort)
    d = d.merge(h, on="order_id", how="left", validate="one_to_one")
    assert len(d) == len(cohort) and d.order_id.is_unique
    assert d.item_count.notna().all()
    d["order_value_proxy"] = d.item_value+d.freight_value
    d["approval_wait_hours"] = (d.order_approved_at-d.order_purchase_timestamp).dt.total_seconds()/3600
    d["approval_month"] = d.order_approved_at.dt.month
    d["approval_weekday"] = d.order_approved_at.dt.dayofweek
    d["approval_hour"] = d.order_approved_at.dt.hour
    d["same_state_largest_item"] = d.customer_state.eq(d.largest_item_seller_state).astype(int)
    d["seller_history_group"] = pd.cut(d.seller_prior_orders_min, [-1,0,20,np.inf], labels=["new_seller_in_order","1_to_20_prior_orders","over_20_prior_orders"]).astype(str)
    numeric = ["item_count", "product_count", "seller_count", "item_value", "freight_value", "weight_g_mean", "volume_cm3_mean", "product_photos_mean", "product_description_length_mean", "approval_wait_hours", "approval_month", "approval_weekday", "approval_hour", "same_state_largest_item", "promise_days", "seller_prior_orders_min", "seller_prior_orders_mean", "seller_observed_outcomes_min", "seller_observed_outcomes_mean", "seller_past_late_rate_max", "seller_past_late_rate_mean"]
    categorical = ["customer_state", "largest_item_seller_state", "product_category_name"]
    for c in categorical:
        d[c] = d[c].fillna("__missing__").astype(str)
    forbidden = set(TIME_COLS+["order_status", "y", "y_timestamp", "delivery_days", "review_score", "label_observed_at"])
    assert not forbidden.intersection(numeric+categorical)
    return d, numeric, categorical


def make_lr(numeric, categorical):
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), categorical),
    ])
    return Pipeline([("pre", pre), ("model", LogisticRegression(C=1.0, max_iter=1200, random_state=SEED))])


def make_hgb(numeric, categorical):
    pre = ColumnTransformer([
        ("num", "passthrough", numeric),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), categorical),
    ], sparse_threshold=0)
    cat_mask = [False]*len(numeric)+[True]*len(categorical)
    model = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
        min_samples_leaf=40, l2_regularization=10.0,
        categorical_features=cat_mask, early_stopping=False, random_state=SEED,
    )
    return Pipeline([("pre", pre), ("model", model)])


def metrics(y, p):
    p = np.clip(np.asarray(p), 1e-7, 1-1e-7)
    return {"n":len(y), "positives":int(np.sum(y)), "prevalence":float(np.mean(y)),
            "average_precision":float(average_precision_score(y,p)),
            "roc_auc":float(roc_auc_score(y,p)) if len(np.unique(y))>1 else np.nan,
            "brier":float(brier_score_loss(y,p)), "log_loss":float(log_loss(y,p,labels=[0,1]))}


def select_by_daily_budget(d, score, budget):
    a = pd.DataFrame({"i":np.arange(len(d)), "day":d.approval_day.to_numpy(),
                      "score":score, "order_id":d.order_id.to_numpy()})
    a = a.sort_values(["day","score","order_id"], ascending=[True,False,True], kind="mergesort")
    a["rank"] = a.groupby("day").cumcount()
    a["quota"] = np.floor(a.groupby("day")["i"].transform("size")*budget+1e-10).astype(int)
    selected = np.zeros(len(d), dtype=bool)
    selected[a.loc[a["rank"]<a["quota"],"i"]] = True
    return selected


def screening_result(d, chosen):
    y=d.y.to_numpy(); v=d.order_value_proxy.to_numpy()
    tp=int(y[chosen].sum()); n=int(chosen.sum())
    return {"selected":n, "actual_budget_share":n/len(d), "true_late_selected":tp,
            "precision_at_budget":tp/n if n else np.nan,
            "recall_at_budget":tp/y.sum(),
            "late_order_value_coverage":float((v*y*chosen).sum()/(v*y).sum())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    data_group=parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--zip", help="Path to the supplied original Olist ZIP")
    data_group.add_argument("--data-dir", help="Directory containing the original Olist CSV files, e.g. a Kaggle input directory")
    parser.add_argument("--out", default="results_reproduced")
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--feedback-mode",choices=["delivered","deadline"],default="delivered",
                        help="Assumed feedback mechanism; neither is an observed Olist label-ingestion log")
    args=parser.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    source=Path(args.zip or args.data_dir)
    tables=load_tables(source)
    cohort,flow=make_cohort(tables["orders"],args.feedback_mode)
    flow.to_csv(out/"cohort_flow.csv",index=False)
    cohort.groupby("approved_month").agg(orders=("y","size"), late_orders=("y","sum"), late_rate=("y","mean"), promise_days_median=("promise_days","median"), delivery_days_median=("delivery_days","median")).to_csv(out/"monthly_cohort_audit.csv")
    d,numeric,categorical=make_features(tables,cohort)
    train_mask=(d.order_approved_at<FIT_ASOF)&(d.label_observed_at<FIT_ASOF)
    cal_candidate=(d.order_approved_at>=CAL_START)&(d.order_approved_at<CAL_END)
    cal_mask=cal_candidate&(d.label_observed_at<CAL_ASOF)
    dev_mask=(d.order_approved_at>=DEV_START)&(d.order_approved_at<DEV_END)
    tr=d.loc[train_mask].copy(); ca=d.loc[cal_mask].copy(); dv=d.loc[dev_mask].copy().reset_index(drop=True)
    assert tr.label_observed_at.max()<FIT_ASOF
    assert ca.label_observed_at.max()<CAL_ASOF
    assert set(tr.order_id).isdisjoint(ca.order_id)
    assert set(tr.order_id).isdisjoint(dv.order_id)
    assert set(ca.order_id).isdisjoint(dv.order_id)
    split_rows=[]
    for name,df in [("train",tr),("calibration",ca),("development_march",dv)]:
        split_rows.append({"split":name,"n":len(df),"positives":int(df.y.sum()),"prevalence":float(df.y.mean()),"approval_min":str(df.order_approved_at.min()),"approval_max":str(df.order_approved_at.max()),"delivery_max":str(df.order_delivered_customer_date.max()),"feedback_max":str(df.label_observed_at.max())})
    pd.DataFrame(split_rows).to_csv(out/"split_summary.csv",index=False)
    manifest={
        "purpose":"Exploratory feasibility study; not final test evidence or a novel method.",
        "source_name":source.name,"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None,
        "source_file_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.glob("*.csv"))} if source.is_dir() else None,
        "versions":{"python":platform.python_version(),"pandas":pd.__version__,"numpy":np.__version__,"scipy":scipy.__version__,"scikit_learn":sklearn.__version__},
        "seed":SEED,"fit_asof":str(FIT_ASOF),"calibration_asof":str(CAL_ASOF),
        "feedback_mode":args.feedback_mode,
        "feedback_definition": "actual delivery" if args.feedback_mode=="delivered" else "min(actual delivery, promised date + one calendar day); assumes immediate tracking and unchanged promise",
        "development_start":str(DEV_START),"development_end_exclusive":str(DEV_END),
        "cohort_n":len(d),"late_n_day_definition":int(d.y.sum()),"late_n_timestamp_definition":int(d.y_timestamp.sum()),
        "same_day_label_disagreements":int((d.y!=d.y_timestamp).sum()),
        "training_pending_outcomes_excluded":int(((d.order_approved_at<FIT_ASOF)&~train_mask).sum()),
        "calibration_pending_outcomes_excluded":int((cal_candidate&~cal_mask).sum()),
        "features_numeric":numeric,"features_categorical":categorical,
        "seller_history_prior":{"alpha":1,"beta":19,"availability":"assumed feedback time strictly before approval time"},
        "validation_type":"One month, no hyperparameter search; all eventually-delivered validation outcomes are scored retrospectively.",
        "unscored_future":"No model scores or model selection results are computed for approvals on/after 2018-04-01.",
        "known_limitations":["Conditioned on eventual delivered status and complete consistent timestamps.","Static promise and product attributes assumed available at approval; historical revisions are unavailable.","Feedback availability filtering and eventual-delivery selection can create bias; maturity-window sensitivity is still required.","No interventions, profit, carrier identity, raw product text, or product images observed.","Daily top-budget selection is a retrospective batch screening simulation.","No causal economic benefits or statistical significance claimed."],
    }
    write_json(out/"run_manifest.json",manifest)
    print("Cohort and splits:",json.dumps(split_rows),flush=True)
    all_metrics=[]; pred={}; training_seconds={}
    numeric_no_promise=[c for c in numeric if c!="promise_days"]
    specifications=[("LogisticRegression",make_lr(numeric,categorical)),("HistGradientBoosting",make_hgb(numeric,categorical)),("HGB_without_promise",make_hgb(numeric_no_promise,categorical))]
    with threadpool_limits(limits=args.threads):
        for name,model in specifications:
            t0=time.perf_counter(); model.fit(tr[numeric+categorical],tr.y)
            training_seconds[name]=time.perf_counter()-t0
            p=model.predict_proba(dv[numeric+categorical])[:,1]
            pred[name]=p
            all_metrics.append({"model":name,**metrics(dv.y,p),"training_seconds":training_seconds[name]})
            print(name,all_metrics[-1],flush=True)
            if name=="HistGradientBoosting":
                pc=model.predict_proba(ca[numeric+categorical])[:,1]
                zc=np.log(np.clip(pc,1e-7,1-1e-7)/(1-np.clip(pc,1e-7,1-1e-7))).reshape(-1,1)
                calibrator=LogisticRegression(C=1000.0,max_iter=1000,random_state=SEED)
                calibrator.fit(zc,ca.y)
                zd=np.log(np.clip(p,1e-7,1-1e-7)/(1-np.clip(p,1e-7,1-1e-7))).reshape(-1,1)
                cp=calibrator.predict_proba(zd)[:,1]
                pred["HGB_static_Platt"]=cp
                all_metrics.append({"model":"HGB_static_Platt",**metrics(dv.y,cp),"training_seconds":np.nan})
                write_json(out/"static_calibration.json",{"slope":float(calibrator.coef_[0,0]),"intercept":float(calibrator.intercept_[0]),"calibration_n":len(ca),"source":"January scores from the frozen January-1 backbone; labels available by March-1 only."})
    prior=float(tr.y.mean()); pred["Constant_train_prevalence"]=np.full(len(dv),prior)
    all_metrics.append({"model":"Constant_train_prevalence",**metrics(dv.y,pred["Constant_train_prevalence"]),"training_seconds":0.0})
    pd.DataFrame(all_metrics).to_csv(out/"baseline_metrics.csv",index=False)
    rng=np.random.default_rng(SEED)
    scores={"Random_seeded":rng.random(len(dv)),"Value_only":dv.order_value_proxy.to_numpy(),"Seller_history_only":dv.seller_past_late_rate_mean.to_numpy(),"LR_risk":pred["LogisticRegression"],"HGB_risk":pred["HistGradientBoosting"],"HGB_calibrated_risk":pred["HGB_static_Platt"],"HGB_risk_times_value":pred["HistGradientBoosting"]*dv.order_value_proxy.to_numpy(),"HGB_calibrated_risk_times_value":pred["HGB_static_Platt"]*dv.order_value_proxy.to_numpy()}
    screening=[]
    for budget in [0.01,0.05,0.10]:
        for name,s in scores.items():
            chosen=select_by_daily_budget(dv,s,budget)
            screening.append({"strategy":name,"nominal_daily_budget":budget,**screening_result(dv,chosen)})
    pd.DataFrame(screening).to_csv(out/"screening_metrics.csv",index=False)
    group_rows=[]
    for group,g in dv.groupby("seller_history_group",sort=False):
        for name,p in pred.items():
            group_rows.append({"group":group,"model":name,**metrics(g.y,p[g.index])})
    pd.DataFrame(group_rows).to_csv(out/"seller_group_metrics.csv",index=False)
    reliability=[]
    for name,p in pred.items():
        for b in range(10):
            mask=(p>=b/10)&(p<(b+1)/10 if b<9 else p<=1)
            if mask.any():
                reliability.append({"model":name,"bin_left":b/10,"bin_right":(b+1)/10,"n":int(mask.sum()),"mean_prediction":float(p[mask].mean()),"observed_rate":float(dv.y.to_numpy()[mask].mean())})
    pd.DataFrame(reliability).to_csv(out/"reliability_bins.csv",index=False)
    # This file is for local reproducibility; it contains anonymized IDs and
    # should not be redistributed as if the original data had an unrestricted license.
    output_pred=dv[["order_id","approval_day","y","order_value_proxy","seller_history_group"]].copy()
    for name,p in pred.items(): output_pred[name]=p
    output_pred.to_csv(out/"development_predictions_local.csv",index=False)
    dictionary=[]
    for c in numeric+categorical:
        is_history=c.startswith(("seller_prior_","seller_observed_","seller_past_"))
        dictionary.append({"feature":c,"type":"numeric" if c in numeric else "categorical","availability":"strictly prior events only" if is_history else "approval-time assumption"})
    pd.DataFrame(dictionary).to_csv(out/"feature_dictionary.csv",index=False)
    print("Complete. Aggregate results:",out,flush=True)


if __name__=="__main__":
    main()
