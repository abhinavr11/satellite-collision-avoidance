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

The first three experiments make point-prediction adjustments at specific horizons. They do not define one dynamically consistent orbit. More independent time periods and better maneuver labels are needed before stronger claims.

## Collision probability experiment

Experiment 4 follows Sections 3–6 of `Collision_Avoidance_Simple_Guide.md`. It recomputes Swarm A/C errors in GCRS using the reference RTN basis, then fits each satellite separately using January 1–17 issue dates. The original pilot's predicted-basis residuals are not reused. The January 18–20 gap and January 21–31 held-out period are preserved. Full K−1 sample covariances are saved for all four bins. Marginal Gaussian likelihood fits each mean quadratic and positive, nondecreasing standard-deviation quadratic in days; a correlation matrix of standardized training residuals supplies cross-direction terms.

The corrected mean adds predicted RTN basis times fitted bias; the uncorrected mean remains SGP4. Both methods use the same centred covariance function, so this is a controlled mean-correction ablation, not a refitted zero-mean baseline. Native TLE ages are preserved by the time shift. Continuous position splines supply mean derivatives for the relative velocity, including the correction's time dependence. All trajectories use the common synthetic UTC labels after native-epoch GCRS conversion.

The fixed encounter is assessed inside ±120 s of the stored reference crossing; each technique finds its own TCA within that event. The full two-hour inputs contain three different close approaches and are not treated as a single encounter. Independent-error relative covariance is projected onto the plane perpendicular to relative mean velocity. Positive-definite 2-D Gaussian probability is integrated over the combined-radius disk with Gauss–Legendre radial and angular rules; refinement must converge and seeded direct Monte Carlo checks sampling agreement. Local curvature, covariance variation and encounter duration are reported as short-encounter diagnostics.

The reference trajectories are treated as exact ground truth for the constructed case. Their continuous minimum separation determines the binary label y = 1 if the separation is at most R_A + R_C, otherwise y = 0. No covariance or collision probability is assigned to the reference trajectories. Only the two prediction techniques have probability forecasts. Their Brier scores (Pc−y)^2 and binary log losses −y log(Pc)−(1−y) log(1−Pc) are reported. Lower is better; a higher Pc scores better on a collision and worse on a non-collision. One selected constructed event cannot establish calibration or general Pc accuracy; validation needs collision and non-collision cases.

Default radii are configurable illustrative 5 m per object, giving a combined collision distance of 10 m. No verified maneuver log is available; optional supplied intervals are screened for overlap with the entire forecast interval. Otherwise all residuals are retained and the limitation is reported. Neither rare-event probability calibration nor avoidance maneuvers are validated by these experiments.

## Balanced forecast lead and confidence gating

Experiment 5 selects among 2-, 3-, 4-, 5- and 6-minute leads, with symmetric held-out windows. A common selection prefix ends six minutes before the saved crossing, no later than any candidate outer cutoff. Selection uses only reference states in that prefix. Rolling position forecasts must reduce RMSE by at least 75% for both objects and satisfy the existing 99% confidence/position gate. The longest eligible lead is used; insufficient-history or failed candidates remain in the record. Three to six nonoverlapping validation blocks are used as history permits. Test Pc and test errors never enter selection.

The selected current lead is three minutes, yielding 3421 training-prefix timestamps and 360 test timestamps per object. Final models are refitted on the selected recent window at the actual cutoff. This can still cause a gate to withhold correction; the outcome is retained rather than choosing another split. Original TLE epochs and forecast ages are preserved.

Recent fitting windows of 300, 600, 900 and 1800 seconds are evaluated by rolling training-only validation. Standardized forecast errors inflate the spreads by at least one and calibrate the correlation. The conditional improvement gate remains Phi((muᵀd−||d||²/2)/sqrt(dᵀPd)), with variance-shrunk d_j = mu_j²/(mu_j²+P_jj)*mu_j. Every validation block must improve RMSE and at least the threshold fraction of positions; minimum projected model confidence over the predefined future grid must pass 0.99. Gate decisions stay fixed across the holdout. This is model confidence, not a demonstrated 99% empirical guarantee.

## Centred covariance versus unremoved-bias error budget

P describes scatter after subtracting the fitted error mean mu. The old raw-mean baseline used this narrow centred residual covariance while leaving mu unapplied, which can generate extreme tail probabilities. The original control remains available explicitly as covariance-mode residual.

The default second_moment mode instead uses S = P + (mu−d)(mu−d)^T as the covariance parameter of a zero-mean Gaussian approximation about the displayed prediction. S equals the second moment of the remaining error, not its true centred covariance. Raw SGP4 uses d=0. A fully bias-corrected prediction with d=mu uses S=P; shrinkage or a withheld gate leaves its remaining bias in the error budget. Frozen uncertainty holds S at the cutoff; projected uncertainty evaluates it at forecast time. Object error budgets are added under the stated independent zero-mean approximation and rotated into GCRS using predicted RTN.

This changes the uncertainty model as well as the effect of correction and must not be presented as merely a new split or a covariance theorem. Identical-mean residual-covariance-only controls are reported alongside the primary forecasts. Neither mode is independently established as the true probability distribution, and plausible-looking probabilities do not prove calibration.

Tests check second moments against Gaussian samples, preserve trajectories across uncertainty modes, and verify exact equivalence between withheld corrected and projected raw predictions. Changing reference states after the common selection cutoff cannot change the selected lead. Other tests verify that test truth cannot alter fitted parameters or gates. Held-out position errors and collision labels are calculated only after predictions are fixed. The same case was previously inspected; independent encounters and both outcomes are needed for general claims.

## Method references

- [Astropy satellite-coordinate example](https://docs.astropy.org/en/stable/coordinates/satellites.html).
- [python-sgp4 documentation](https://pypi.org/project/sgp4/).
- [ESA/Delft precise reduced-dynamic Swarm orbit products](https://swarm-disc.github.io/product-catalogue-tools/SW_SP3xCOM_2_.html).

The root README states the exact formulas implemented by each pilot. Each experiment's summary stores its parameters, metrics, and input hashes; its report discusses the results and limitations.
