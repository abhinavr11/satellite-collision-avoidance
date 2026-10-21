# A small position-correction experiment

**Result: a simple training-only bias correction produces small average position-error improvements. It worsens many individual predictions.**

Variance describes error size, not its direction. We therefore fitted the mean RTN error from the same January 1-17 training residuals used for the uncertainty experiment. This adds an estimate of systematic error to the earlier uncertainty model. No test residual was used to fit or tune the correction.

For each horizon h, the correction is b(h) = average training reference-minus-SGP4 error in RTN coordinates. We apply r_corrected = r_SGP4 + RTN_basis_transpose times b(h). The RTN basis comes from the SGP4 predicted position and velocity, so it needs no future reference measurement. Reference positions are used only to score the test predictions.

The original uncertainty pilot estimated mean squared errors about the unchanged SGP4 position. This follow-up separates that second moment into bias squared plus variance. The centered uncertainty scales are saved, but these results measure position error, not the calibration of the new uncertainty regions.

Training issue dates: January 1-17, 2025. Gap: January 18-20. Test issue dates: January 21-31. There are 86 test TLE issue records, each evaluated at four horizons, giving 344 correlated test forecasts. This reuses the earlier pilot's test period and is exploratory.

| Horizon | Original 3-D RMSE | Corrected 3-D RMSE | RMSE reduction | Individual cases improved |
|---|---:|---:|---:|---:|
| 6 hours | 697.0 m | 693.6 m | 0.50% | 36/86 |
| 12 hours | 716.9 m | 661.7 m | 7.70% | 48/86 |
| 24 hours | 642.7 m | 632.9 m | 1.53% | 44/86 |
| 48 hours | 2294.9 m | 2118.1 m | 7.70% | 58/86 |

RMSE gives greater weight to large errors. Improvement in RMSE does not imply improvement in every other metric. At 6 hours, mean distance error increases even though RMSE improves slightly. At 12 hours, median distance error also increases. The JSON summary includes all these metrics.

The 24-hour correction learned from training is approximately: radial -75.2 m, along-track +41.0 m, cross-track +30.5 m. It is the same RTN offset for every 24-hour case in this simple model. It is not adjusted using the true position of the case being corrected.

The following examples are the **first six 24-hour test forecasts in chronological issue order**. They include failures; none were selected because the correction helped.

| Satellite | Prediction issued (UTC) | Reference comparison (UTC) | Original error | Corrected error | Result |
|---|---|---|---:|---:|---|
| Swarm-B | 2025-01-21T14:24:42.160608Z | 2025-01-22T14:24:42.160608Z | 288.0 m | 297.7 m | Worsened |
| Swarm-C | 2025-01-21T14:52:01.075296Z | 2025-01-22T14:52:01.075296Z | 299.9 m | 324.2 m | Worsened |
| Swarm-A | 2025-01-21T14:52:06.808800Z | 2025-01-22T14:52:06.808800Z | 455.0 m | 488.5 m | Worsened |
| Swarm-C | 2025-01-21T17:59:20.179968Z | 2025-01-22T17:59:20.179968Z | 620.6 m | 649.8 m | Worsened |
| Swarm-A | 2025-01-21T17:59:25.903968Z | 2025-01-22T17:59:25.903968Z | 544.5 m | 573.8 m | Worsened |
| Swarm-C | 2025-01-21T21:06:39.269952Z | 2025-01-22T21:06:39.269952Z | 277.6 m | 220.1 m | Improved |

Five of these first six forecasts worsen. Across all 86 test forecasts at 24 hours, 44 improve and 42 worsen. A small overall improvement is not a reliable correction for every individual encounter.

Actual corrected Cartesian positions are saved in `corrected_positions.json`, in TEME coordinates and metres. They were formed by rotating each training-derived RTN correction into TEME and adding it to the SGP4 position. All residual norms were checked against the RTN calculation. The six example reference positions were also independently reloaded from the SP3 files and transformed again; agreement was better than 0.00001 m.

Limits: only one month and three related satellites; no verified maneuver exclusions; time-varying bias; correlated samples; no demonstrated generalization to other satellites. Separate position corrections at different horizons do not constitute a validated new orbital propagator. More data and trajectory-consistent fitting would be needed before using corrected positions for collision avoidance.

Run from the Project directory using the existing pilot environment:

```bash
.venv-sgp4-pilot/bin/python experiments/sgp4_position_correction_pilot/run_experiment.py
```

Dependencies and the original reference-processing code are in the adjacent `sgp4_uncertainty_pilot` directory. The input residual file is hashed in this experiment's summary. Neither the raw data nor the earlier uncertainty experiment is modified.
