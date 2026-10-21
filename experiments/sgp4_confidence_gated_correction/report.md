# Correct only when the variance model supports a confident correction

The 80% model-estimated improvement threshold was fixed before scoring this experiment's test outcomes. Training dates are January 1-17, 2025; testing uses the same January 21-31 exploratory period as the earlier experiments. There is a January 18-20 gap.

We fitted separate mean errors and diagonal variances for each satellite and horizon. This uses 42-47 training forecasts per group, spread across 17 issue dates. Variance alone supplies no correction direction. The training mean supplies that direction.

We first shrink each mean correction component with w_j = mean_j^2 / (mean_j^2 + variance_j), and d_j = w_j mean_j. This is a simple heuristic: a large random spread relative to systematic bias makes that component small. It is not an optimality claim.

Under the fitted model e ~ N(mean, diagonal variance), a correction d reduces squared 3-D distance error exactly when e dot d > ||d||^2/2. Therefore its estimated success probability is Phi((mean dot d - ||d||^2/2) / sqrt(sum variance_j*d_j^2)). We apply d only if this estimate is at least 0.80. Otherwise the original position is preserved exactly.

**Selected 86 of 344 test forecasts. Left 258 unchanged.** Among the selected forecasts, 77 improved and 9 worsened.

| Selected group | Estimated chance of improvement | Actual improved / changed | Original selected-group RMSE | After correction |
|---|---:|---:|---:|---:|
| Swarm-A, 24 h | 86.6% | 26/29 | 656.14 m | 643.35 m |
| Swarm-B, 48 h | 85.7% | 25/29 | 1528.11 m | 1515.11 m |
| Swarm-C, 24 h | 89.2% | 26/28 | 755.84 m | 743.96 m |

All other satellite/horizon groups were left unchanged. The gate selects groups, so all forecasts for a given satellite at a given horizon receive the same confidence estimate. It does not identify which individual forecasts within that group will fail.

The following comparisons isolate the effect of the gate while keeping the satellite-specific fitting fixed:

| Policy | Overall 3-D RMSE | Improved | Worsened | Unchanged |
|---|---:|---:|---:|---:|
| Always apply full group mean | 1177.62 m | 199 | 145 | 0 |
| Always apply variance-weighted correction | 1285.22 m | 249 | 95 | 0 |
| Apply weighted correction only above 80% | 1289.85 m | 77 | 9 | 258 |

Original uncorrected overall RMSE: 1292.25 m. Gated overall RMSE: 1289.85 m (0.19% reduction). Original overall mean distance error: 897.22 m; gated: 891.26 m. Original median distance error: 566.89 m; gated: 566.89 m.

The first six changed test forecasts, selected by chronological issue time regardless of outcome:

| Satellite | Horizon | Issue time (UTC) | Original error | Gated error | Result |
|---|---:|---|---:|---:|---|
| Swarm-B | 48 h | 2025-01-21T14:24:42.160608Z | 993.88 m | 950.15 m | Improved |
| Swarm-C | 24 h | 2025-01-21T14:52:01.075296Z | 299.90 m | 271.99 m | Improved |
| Swarm-A | 24 h | 2025-01-21T14:52:06.808800Z | 454.99 m | 440.53 m | Improved |
| Swarm-C | 24 h | 2025-01-21T17:59:20.179968Z | 620.59 m | 619.64 m | Improved |
| Swarm-A | 24 h | 2025-01-21T17:59:25.903968Z | 544.46 m | 544.47 m | Worsened |
| Swarm-C | 24 h | 2025-01-21T21:06:39.269952Z | 277.58 m | 257.28 m | Improved |

Interpretation: a selective rule trades the size and frequency of corrections against the risk of worsening individual positions. Its fitted confidence is only a model estimate. An 80% probability of helping would still permit some failures even under a correct model. Here Gaussian tails, estimated parameters, time variation, and unknown maneuvers add further uncertainty.

Important controls and limits:

- The reported probability is an uncalibrated plug-in Gaussian model estimate, not a validated 80% guarantee.
- Only 42-47 training forecasts over 17 issue dates per group, with temporal and cross-satellite dependence.
- Confidence is constant within each satellite/horizon group, not a case-specific anomaly detector.
- The test period was used in previous exploratory experiments; a new independent period is needed for confirmation.
- Satellite-specific fitting and variance shrinkage are both changes relative to the earlier pooled mean correction.
- No verified maneuver exclusions, no rare-risk validation, and no guarantee of trajectory consistency.

All positions are in TEME, in metres. The script recomputed SGP4 positions, applied the rotated RTN corrections, and checked Cartesian error norms against RTN errors. It reloaded the first six changed reference states from SP3 files. A Monte Carlo check also confirmed the analytic success-probability formula under its assumed Gaussian model; this validates the calculation, not the real-data probability model.

Run from the Project directory using the existing environment:

```bash
.venv-sgp4-pilot/bin/python experiments/sgp4_confidence_gated_correction/run_experiment.py
```

`summary.json` stores all twelve fitted group models, their gates, scores, and input hash. `corrected_positions.json` stores all 344 test predictions, their confidence estimates, and corrected or preserved positions. Earlier experiment files and raw data remain unchanged.
