"""Reconstruct core evidence; no model training or new simulation by default."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
from scipy.stats import t
from verify_e18_daily import verify_e18

ROOT = Path(__file__).resolve().parents[1]
SCORES = ["Brier", "LogLoss", "ExpectedBrier", "ExpectedLogLoss", "ProbabilityMSE"]
PAIRS = [("Expanding", "Frozen"), ("Frozen", "Raw"), ("Expanding", "Raw"),
         ("Rolling60", "Expanding"), ("PoolRate", "Expanding")]


def simulation_from_scores(out):
    source = ROOT / "evidence/E19"
    d = pd.read_csv(source / "replicate_scores.csv")
    idcols = ["Scenario", "Policy", "Period", "Method", "Replicate"]
    if d.duplicated(idcols).any():
        raise ValueError("Duplicate simulation observation")
    allmetrics = SCORES + ["MeanPred", "ActualRate"]
    absolute, effects, quantiles = [], [], []

    def stats(x):
        n = len(x)
        if n != 100:
            raise ValueError("Expected 100 sequences per group")
        mu = float(np.mean(x))
        sd = float(np.std(x, ddof=1))
        se = sd / np.sqrt(n)
        h = float(t.ppf(.975, n-1)) * se
        return dict(Mean=mu, SD=sd, MCSE=se, CI_low=mu-h, CI_high=mu+h, Sequences=n)

    for key, group in d.groupby(["Scenario", "Policy", "Period", "Method"]):
        group = group.sort_values("Replicate")
        if group.Replicate.tolist() != list(range(100)):
            raise ValueError("Missing sequence IDs")
        base = dict(zip(["Scenario", "Policy", "Period", "Method"], key))
        for metric in allmetrics:
            absolute.append({**base, "Metric": metric, **stats(group[metric].to_numpy())})

    for key, group in d.groupby(["Scenario", "Policy", "Period"]):
        base = dict(zip(["Scenario", "Policy", "Period"], key))
        for updated, reference in PAIRS:
            a = group[group.Method.eq(updated)].set_index("Replicate").sort_index()
            b = group[group.Method.eq(reference)].set_index("Replicate").sort_index()
            if not a.index.equals(b.index) or len(a) != 100:
                raise ValueError("Unpaired sequence scores")
            for metric in SCORES:
                x = a[metric].to_numpy() - b[metric].to_numpy()
                label = {**base, "Updated": updated, "Reference": reference, "Metric": metric}
                effects.append({**label, **stats(x)})
                q = np.quantile(x, [.1, .5, .9])
                quantiles.append({**label, "Sequences": len(x), "Q10": q[0],
                                  "Median": q[1], "Q90": q[2]})

    check_results = {}
    for name, rows, keys, values in [
        ("absolute_scores", absolute, ["Scenario", "Policy", "Period", "Method", "Metric"],
         ["Mean", "SD", "MCSE", "CI_low", "CI_high", "Sequences"]),
        ("paired_effects", effects, ["Scenario", "Policy", "Period", "Updated", "Reference", "Metric"],
         ["Mean", "SD", "MCSE", "CI_low", "CI_high", "Sequences"]),
        ("sequence_effect_quantiles", quantiles, ["Scenario", "Policy", "Period", "Updated", "Reference", "Metric"],
         ["Q10", "Median", "Q90", "Sequences"]),
    ]:
        frame = pd.DataFrame(rows)
        reference = pd.read_csv(source / (name + ".csv"))
        a = frame.set_index(keys).sort_index()
        b = reference.set_index(keys).sort_index()
        if not a.index.equals(b.index):
            raise ValueError("Reference keys differ: " + name)
        error = float(np.max(np.abs(a[values].to_numpy() - b[values].to_numpy())))
        if error >= 1e-11:
            raise ValueError(f"{name} differs from frozen reference: {error}")
        frame.to_csv(out / (name + ".csv"), index=False, float_format="%.17g")
        check_results[name] = dict(rows=len(frame), max_abs_difference=error)
    return check_results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        ap.error("Use a new or empty output directory")
    hashes = 0
    for line in (ROOT / "INPUTS.sha256").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Retained input changed: " + name)
        hashes += 1
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/rebuild_core_tables.py"),
                    "--out", str(out / "real_data")], check=True)
    e18 = verify_e18(ROOT / "evidence")
    (out / "simulation").mkdir()
    e19 = simulation_from_scores(out / "simulation")
    from verify_simulation_details import verify_details
    e19["additional_checks"] = verify_details(ROOT / "evidence/E19", out / "simulation")
    subprocess.run([sys.executable, str(ROOT / "scripts/make_revision_tables.py")],
                   check=True)
    shutil.copytree(ROOT / "tables", out / "tables")
    shutil.copy2(ROOT / "evidence/protocols/S1_training_support.csv",
                 out / "tables/Table1_training_support.csv")
    subprocess.run([sys.executable, str(ROOT / "scripts/make_manuscript_extended.py"),
                    "--out", str(out)], check=True)
    report = {
        "passed": True, "scientific_version": "0.4.1", "distribution": "paper",
        "retained_input_hashes": hashes,
        "real_data": json.loads((out / "real_data/validation.json").read_text()),
        "E18_daily_reconstruction": e18, "E19_period_score_reconstruction": e19,
        "training_rerun": False, "simulation_generation_rerun": False,
        "independent_person_replication": False,
        "scope": "Daily-sum real-data checks and period-score simulation aggregation; see docs/REPRODUCTION_COVERAGE.md"
    }
    (out / "VALIDATION.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
