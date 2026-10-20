# Manuscript evidence and execution coverage

| Manuscript evidence | Retained inputs and entry | What is checked |
|---|---|---|
| Table 1: origin/training support | `evidence/protocols/S1_training_support.csv`, fixed protocol and feature dictionary | Archived support table; cohort selection is not re-established by aggregate checks |
| Tables 2–3: scores and initial/update/total contrasts; Figure 2 | `evidence/e17/manuscript_S1_complete_main_table.csv`, daily model sums, `rebuild_core_tables.py` | Re-estimated Raw/Frozen/Expanding Brier and log-loss intervals at 3/7/14 days, all origins and three backbones; other comparator scores retained as archived values |
| Table 4: historical-window contrasts | `evidence/e17/manuscript_primary_pairs_both_scores.csv`, `make_revision_tables.py` | Archived comparison table regenerated; these additional intervals are not re-estimated by the core checker |
| Figure 3: calendar sensitivity | Full four-origin comparison CSV and O2 monthly CSVs | Archived results and plotting source; no S0/S2/S3 refitting by the core checker |
| Figures 4–5: trajectories and calibration | Weekly trajectories and reliability-bin CSV, `make_revision_figures.py` | Aggregate plots reproduced; intervals supplied rather than recomputed |
| Table 5: realized loss terms | E18 daily sums, decomposition and outcome contributions | Every retained term and 3/7/14-day interval re-estimated; outcome additivity checked |
| Table 6 / Figure 6: controlled simulation | Plain CSV replicate scores, absolute and paired references, protocol, `run_e19.py` | 1,260 absolute scores, 900 paired effects and 900 quantile rows independently reconstructed from period-level scores; full array generation available separately |
| Table 7: screening | Decision metrics and paired intervals | Archived decision table reproduced; no private-order ranking reconstruction |
| Backbone and unknown-outcome robustness | Daily sums, aligned effect references and aggregate loss bounds | Cross-backbone intervals and unknown-outcome arithmetic checked |
| Figure 1 | Original frozen PDF | Preserved unchanged; this is a conceptual diagram |

Historical plotting filenames predate final manuscript layout: `figure2_primary`
maps to Figure 2; `figure3_calendar` to Figure 3; `figure5_O2_trajectory` and
`figure5_O3_trajectory` are inputs to Figure 4; `figure6_O2_reliability_zoom`
and `figure6_O3_reliability_zoom` are inputs to Figure 5. `figure4_windows`
is an additional plot of the Table 4 contrasts, not final manuscript Figure 4.
The final multi-panel layout is part of the manuscript source; plot components
are not claimed to be byte-identical to final composed pages.

Paper-relevant DataCo and LaDe summaries and LaDe support/pool information are included. Earlier exploratory archives are not duplicated. Prior author-side validation
reports do not imply that all training or the full 400-sequence simulation was
rerun while producing this manuscript distribution.

The period-score simulation check starts after the simulated trajectories have
been generated. It cannot check event-boundary invariants or daily pool dynamics.
The unchanged generator and locked protocol permit reviewers to inspect and
rerun that upstream layer if needed. Loss contrasts are paired by sequence;
policies reuse the same generated sequence and are not independent replicates.


## Supplementary coverage and retained execution boundaries

| Supplement | Included evidence | Execution boundary |
|---|---|---|
| S1–S3: cohort, time and implementation | Cohort flow, pending counts, feature dictionary, availability audit, training-boundary checks, solver/support summary, E17 protocol | Core/S1 raw-data entry available; original record-level audit logs not redistributed |
| S4–S5: full comparisons and diagnostics | Complete E17 metrics/intervals, calibration intercept/slope, reliability bins, trajectories | Core three-contrast intervals re-estimated; supplied other intervals retained |
| S6: robustness and unknown outcomes | All R2 backbone effects, bounds, follow-up sets and reconstruction summaries | Rebuild from sufficient statistics; does not retrain alternative backbones |
| S7: screening and auxiliary tasks | Decision metrics/list changes, DataCo summary, LaDe scores/pairs and pool summary | Frozen aggregate evidence; no auxiliary-task raw-data training entry claimed |
| S9–S10: aligned effects and loss diagnosis | All R2 aggregate files, E18 daily/weekly sums, outcome contributions, intervals and checks | Public daily reconstruction and separate private-record alignment/replay available |
| S11: controlled simulation | All final E19 CSV outputs, metadata, protocol and generator | Period-score, daily-summary and decomposition checks; full daily arrays generated separately |
| S12: reproduction scope | README commands, dependency locks, input and distribution manifests, local validation report | Author-side execution, not independent-person replication |

Calibration-policy functions for PoolRate, Platt and BrierOpt are present in
`src/fulfillment_recalibration/e17_calibration.py`. The saved-score replay entry
uses them conditional on out-of-training scores. CatBoost, Restricted LR and
PeriodicLR settings are specified in `evidence/protocols/E17_protocol_public.json`,
but the manuscript package does not claim a portable all-model raw-data entry.
This is an existing study reproducibility limit, not a successful full-model
rerun inferred from aggregate checks.

The validation command also verifies per-sequence effects against score differences,
reconstructs simulation decomposition intervals, checks daily expected-risk identities,
and compares daily means with period summaries. Generator metadata are archived
statements; reading them does not independently revalidate the underlying events.
