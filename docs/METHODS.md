# Methods and interpretation

## Reference-minus-prediction errors

SGP4/WGS72 propagates each historical TLE to its indexed future UTC comparison time. The precise Swarm products supply Earth-fixed reference positions and velocities every ten seconds. Cubic Hermite interpolation evaluates the reference at the requested time; local degree-seven position interpolation is an independent interpolation cross-check.

SP3 epochs are GPS time. GPS is 18 seconds ahead of UTC throughout the selected January/February 2025 interval. SP3 position units are kilometres; velocity units are decimetres per second. Conversion factors are 1000 and 0.1 respectively. File-name epochs are UTC, whereas the SP3 epoch records are GPS time.

Reference states in the IGS20/IGb20 terrestrial realizations are treated as ITRS and transformed to TEME with Astropy and the downloaded IERS table. Realization differences at centimetre scale are neglected for this pilot. The model uses no automatic IERS downloads while calculating errors.

The reference-minus-prediction position difference is projected onto the RTN basis calculated from the predicted position and velocity. This basis is available when making the prediction, without consulting the future reference. Checks include basis orthonormality, preservation of vector norms, coordinate round trips, and an explicit GMST82/polar-motion rotation calculation. These are implementation checks, not independent validation of the reference ephemerides.

## Chronological split

Forecast issue dates January 1–17 are training; January 18–20 are excluded; January 21–31 are testing. The split has 540/96/344 comparisons across four horizons. The latest training reference timestamp is earlier than the first test issue timestamp. Training parameters are never fitted from test residuals.

The correction studies reuse the test period from the uncertainty pilot. This is exploratory development, not three independent held-out studies. Nearby predictions share orbital conditions and reference data; Swarm A and C also have related trajectories. Simple independent-sample significance interpretations would be inappropriate. The uncertainty script includes an approximate day-block bootstrap, but eleven test days are too few to establish broad generalization.

## Uncertainty calibration

The fixed baseline has one fitted isotropic scale. The horizon-only control has four. The directional model has twelve: an RTN scale for each horizon. All use the same training dates.

The directional model fits mean squared errors about the unchanged SGP4 prediction. For a nonzero bias, this second moment is variance plus squared bias. The code deliberately does not subtract the mean in Experiment 1. Its Gaussian confidence regions remain centered at the SGP4 position.

For a test error and covariance, the squared Mahalanobis distance is compared with a three-degree-of-freedom chi-square quantile. Coverage is the fraction inside that region. Average coverage mismatch is averaged equally over 50%, 80%, 90%, 95%, and 99% confidence. Negative log-likelihood also penalizes overly broad or misaligned uncertainty. Positions remain unchanged, so a calibration improvement cannot be described as improved position accuracy.

## Mean position correction

Experiment 2 fits a mean RTN error for each horizon, pooling the three satellites. It rotates that mean into TEME and adds it to the predicted Cartesian position. The centered training variance is also recorded, but the experiment's reported improvement is in position error.

All later forecast instances are scored. The first six chronological 24-hour test forecasts are printed regardless of their result. The comparison uses 3-D RMSE, mean distance error, median distance error, and individual improved/worsened counts. These metrics answer different questions: RMSE emphasizes large errors, and an RMSE gain need not improve the median or most individual cases.

## Variance weighting and selective correction

Experiment 3 fits separate mean and diagonal centered variance for each satellite/horizon group. There are twelve groups, each with 42–47 training forecasts over seventeen issue dates. It uses sample variance with one degree-of-freedom correction.

The weight `mean² / (mean² + variance)` shrinks each candidate mean correction. This is a heuristic that suppresses components dominated by random error, not a proof of optimality. For a fixed candidate displacement, the Gaussian linear projection gives a closed-form probability that the squared 3-D error decreases. The script checks that probability calculation using Monte Carlo samples drawn from the assumed Gaussian model.

The 80% cutoff was fixed before scoring this experiment's test outcomes. Below the cutoff, the position is preserved bit-for-bit. Above it, the RTN correction is rotated into TEME and applied. Confidence is constant within each satellite/horizon group. It does not identify which individual forecasts within that group are unusual.

The estimated probability is a plug-in model probability. It ignores uncertainty in the fitted parameters and relies on simplified Gaussian and diagonal-covariance assumptions. It is not a calibrated guarantee of success. The experiment also changes the fitting groups and applies shrinkage, so same-group ungated controls are included to separate those effects from the gate.

## Data quality and scope

No verified maneuver history is available. All valid indexed comparisons were retained; no residuals were removed after looking at test outcomes. Some training errors reach approximately 58 km at 48 hours, and their cause is not established. This affects both scales and mean estimates.

The corrections are point-prediction adjustments at specific horizons. They do not define one dynamically consistent orbit. These experiments do not validate collision probabilities, rare collision-probability tails, or avoidance maneuvers. More independent time periods and better maneuver labels are needed before stronger claims.

## Method references

- [Astropy satellite-coordinate example](https://docs.astropy.org/en/stable/coordinates/satellites.html).
- [python-sgp4 documentation](https://pypi.org/project/sgp4/).
- [ESA/Delft precise reduced-dynamic Swarm orbit products](https://swarm-disc.github.io/product-catalogue-tools/SW_SP3xCOM_2_.html).

The root README states the exact formulas implemented by each pilot. Each experiment's summary stores its parameters, metrics, and input hashes; its report discusses the results and limitations.
