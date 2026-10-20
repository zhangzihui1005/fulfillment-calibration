# E17 portability boundary

The functions `logit`, `intercept`, `solve_platt`, `solve_brier` and `replay`
are extracted with unchanged function bodies from the supplied frozen
`03_E17补充实验/code/run_e17.py`.

Original file SHA-256: `947b3d40dcbc697e6854657e44931e5f2882b2b2f10f474c79f114f839c701b2`.

The wrapper removes hard-coded original workstation paths and imports of private
training helpers. It reads CSV inside a **user-provided private** archive and does
not extract or execute that archive's code, deserialize models, or train a base
model. It reproduces the seven calibration policies conditional on the saved S1
scores and checks them against saved Core scores/pools. This does not independently
verify score-generation features or establish production label availability.
The original full private archive remains the source for E17 CatBoost, restricted
LR, periodic LR, auxiliary-task fitting and diagnostics. Those stages are not
advertised as a new portable end-to-end pipeline.

Public execution coverage is recorded in `REPRODUCTION_COVERAGE.md` and local checks in `VALIDATION.json`. The saved-score and raw-training scopes are separate.
