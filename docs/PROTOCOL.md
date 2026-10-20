# Study protocol and limitations

## Main Olist outcome

At order-approval time estimate delivery after the promised **calendar day**. Only finally delivered, time-consistent orders are in the primary analytical cohort (96,394 orders). That future-status selection is a limitation; this repository does not claim a prospective all-order deployment evaluation.

In the primary deadline-closed scenario, binary outcome observability is reconstructed as `min(actual delivery time, deadline end)`, while calibration eligibility is `max(observation, deadline end)` and thus **equals deadline end** for all completed-delivery observations. Real platform ingestion logs were unavailable.

## Evaluation (UTC offset unspecified; original local timestamps)

| Origin | Fit cutoff | 60-day warm-up | 60-day evaluation [start, end) |
|---|---|---|---|
| O1 | 2017-07-01 | 2017-07-01–08-30 | 2017-08-30–10-29 |
| O2 | 2017-10-01 | 2017-10-01–11-30 | 2017-11-30–2018-01-29 |
| O3 | 2018-01-01 | 2018-01-01–03-02 | 2018-03-02–05-01 |
| O4 | 2018-04-01 | 2018-04-01–05-31 | 2018-05-31–07-30 |

The LR is independently trained at each origin using **only outcomes observable before fitting**; calibration uses outside-training predictions from subsequent orders. S1 is the main presentation (sin/cos month). S0 represents month as scalar; S2 one-hot; S3 drops month. Calendar encoding comparisons were devised after seeing early results: retrospective, not fully independent testing.

## Update policies

At 00:00 on each evaluation day, only eligibility strictly before 00:00 can enter. `Raw` does not recalibrate. `Frozen` fits once after 60-day warm-up. `Expanding` uses all eligible post-fit observations. `Rolling60` uses observations eligible within the preceding 60 days. `Approval60` uses orders approved within the preceding 60 days and eligible now. Calibration: unpenalized intercept in logit space, raw clipping at 1e-7, min pool 1000 with >=50 outcomes per class, else return Raw.

## Statistics

Brier is the primary **proper probability scoring loss**, not an isolated conditional-calibration metric. Bootstrap: 2000 paired replicates, contiguous noncircular seven-calendar-day blocks, identical resampled dates within a paired comparison; uncertainty conditional on frozen predictions. No adjustment for multiple comparisons or exploration/model choice. Overlapping training histories and evaluated periods are not independent prospective trials.

## Auxiliary datasets

DataCo is **dispatch/shipping delay** (not consumer delivery). LaDe-P is **pickup deadline exceedance** (not consumer delivery), with very low positive prevalence in Jilin. Report within-dataset comparisons; do not pool absolute Brier scores across different outcomes.


# Fixed-prediction analysis protocol

The R2 analysis preserves the prediction models, sample assignments and tuning parameters. It reports all three paired contrasts; reconstructs Frozen probabilities using stored offsets; and computes cross-backbone differences of effects from identical saved records. All new statistical selections followed inspection of the historical data and are exploratory. Original loss/interval files remain unchanged; additions live in evidence/r2/.
