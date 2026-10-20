"""Additional checks of retained simulation summaries and sequence scores."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t


def verify_details(source: Path, out: Path):
    checks = 0
    maxerr = 0.0

    def close(a, b):
        nonlocal checks, maxerr
        err = float(np.max(np.abs(np.asarray(a, float)-np.asarray(b, float))))
        if err >= 1e-11:
            raise ValueError(f"Simulation summary mismatch: {err}")
        checks += 1
        maxerr = max(maxerr, err)

    scores = pd.read_csv(source/"replicate_scores.csv")
    effects = pd.read_csv(source/"replicate_effects.csv")
    key = ["Scenario", "Policy", "Period", "Replicate"]
    metrics = ["Brier", "LogLoss", "ExpectedBrier", "ExpectedLogLoss",
               "ProbabilityMSE", "MeanPred", "ActualRate"]
    for (updated, reference), g in effects.groupby(["Updated", "Reference"]):
        x = g.set_index(key).sort_index()
        a = scores[scores.Method.eq(updated)].set_index(key).sort_index()
        b = scores[scores.Method.eq(reference)].set_index(key).sort_index()
        if not x.index.equals(a.index) or not a.index.equals(b.index):
            raise ValueError("Unpaired archived effects")
        close(x[metrics], a[metrics]-b[metrics])

    rep = pd.read_csv(source/"replicate_decomposition.csv")
    ref = pd.read_csv(source/"decomposition.csv")
    terms = ["P", "A_realized", "A_probability", "Delta_realized", "Delta_expected"]
    close(rep.P-rep.A_realized, rep.Delta_realized)
    close(rep.P-rep.A_probability, rep.Delta_expected)
    keys = ["Scenario", "Policy", "Period", "Updated", "Reference"]
    rows = []
    for k, g in rep.groupby(keys):
        if sorted(g.Replicate.tolist()) != list(range(100)):
            raise ValueError("Incorrect decomposition sequence IDs")
        for term in terms:
            x = g[term].to_numpy()
            mu, sd = float(x.mean()), float(x.std(ddof=1))
            se = sd / 10
            h = t.ppf(.975, 99)*se
            rows.append({**dict(zip(keys, k)), "Term": term, "Mean": mu, "SD": sd,
                         "MCSE": se, "CI_low": mu-h, "CI_high": mu+h, "Sequences": 100})
    frame = pd.DataFrame(rows)
    a, b = frame.set_index(keys+["Term"]).sort_index(), ref.set_index(keys+["Term"]).sort_index()
    if not a.index.equals(b.index):
        raise ValueError("Decomposition reference keys differ")
    close(a, b[a.columns])
    frame.to_csv(out/"decomposition.csv", index=False, float_format="%.17g")

    # Daily cross-method invariant and consistency of mean daily scores with
    # period-level absolute scores; these do not verify individual event times.
    daily = pd.read_csv(source/"daily_scores.csv")
    for _, g in daily.groupby(["Scenario", "Policy", "Day"]):
        noise = g.ExpectedBrier.to_numpy()-g.ProbabilityMSE.to_numpy()
        close(noise, np.repeat(noise[0], len(noise)))
    absolute = pd.read_csv(source/"absolute_scores.csv")
    for (scenario, policy, method), g in daily.groupby(["Scenario", "Policy", "Method"]):
        for period, sub in [("Full", g), ("Phase1", g[g.Day.lt(120)]), ("Phase2", g[g.Day.ge(120)])]:
            ref = absolute[(absolute.Scenario==scenario)&(absolute.Policy==policy)&
                           (absolute.Method==method)&(absolute.Period==period)].set_index("Metric")
            close(sub[metrics].mean().to_numpy(), ref.loc[metrics, "Mean"].to_numpy())
    daydiag = pd.read_csv(source/"daily_decomposition.csv")
    close(daydiag.P-daydiag.A_realized, daydiag.Delta_realized)
    close(daydiag.P-daydiag.A_probability, daydiag.Delta_expected)
    meta = pd.read_csv(source/"sequence_metadata.csv")
    if len(meta) != 400 or meta.duplicated(["scenario", "replicate"]).any():
        raise ValueError("Sequence metadata incomplete")
    if not meta[["event_and_boundary_checks", "closed_invariance", "raw_policy_invariance"]].all().all():
        raise ValueError("Archived generator validation failed")
    return {"checks": checks, "max_abs_difference": maxerr,
            "decomposition_rows_rebuilt": len(frame),
            "sequence_metadata_rows": len(meta),
            "scope": "Saved summary identities and paired period scores; archived event checks read, not independently rerun"}
