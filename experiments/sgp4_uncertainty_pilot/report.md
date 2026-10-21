# A small SGP4 uncertainty-calibration experiment

**Result: the direction-and-horizon uncertainty model matches the test errors better overall than either simple baseline. It is still too conservative at high confidence levels. The predicted positions do not improve or change.**

We used existing data for Swarm A, B, and C. Every comparison uses an older TLE propagated with SGP4 and an independently reconstructed precise reference position at the same future time. Horizons are 6, 12, 24, and 48 hours.

Training uses forecasts issued January 1-17, 2025: **540 comparisons**. Forecasts issued January 18-20 are excluded as a gap: **96 comparisons**. Testing uses forecasts issued January 21-31: **344 comparisons**, from 86 TLE issue records and four horizons. These are not 344 statistically independent samples. No training forecast target overlaps the testing issue period.

We compared three deliberately simple uncertainty models. SGP4 itself does not supply a covariance. Every model was fitted on the same training dates. No test errors were used to choose its fitted parameters. The horizon-only model is an additional control to check that the result is not solely caused by pooling horizons.

| Model | Average confidence mismatch | Mean negative log-likelihood |
|---|---:|---:|
| Fixed uncertainty in every direction and at every horizon | 16.7 percentage points | 26.47 |
| Different uncertainty at each horizon; same in every direction | 12.8 percentage points | 24.55 |
| Different uncertainty by horizon and RTN direction | 6.0 percentage points | 21.56 |

Lower is better for both metrics. Confidence mismatch is the average absolute difference between claimed and observed joint 3-D coverage at 50%, 80%, 90%, 95%, and 99%. Negative log-likelihood scores both uncertainty-region width and agreement with the observed errors; these values use position errors in metres.

| Claimed confidence | Fixed baseline: observed coverage | Horizon-only control | Direction + horizon: observed coverage |
|---|---:|---:|---:|
| 50% | 98.0% | 86.9% | 57.0% |
| 80% | 99.7% | 94.2% | 91.9% |
| 90% | 100.0% | 98.0% | 96.2% |
| 95% | 100.0% | 98.8% | 99.1% |
| 99% | 100.0% | 100.0% | 100.0% |

For example, a 50% uncertainty region should contain roughly half of the test reference positions. The fixed baseline contained 98.0%; the direction-and-horizon model contained 57.0%. That is substantially closer. However, its 95% regions contained 99.1%, so this model is not yet well calibrated at all confidence levels. The horizon-only control is slightly closer at 95% alone. Do not claim that the direction-and-horizon model wins every metric or horizon.

The SGP4 position errors are **identical for all three models**:

| Forecast horizon | Held-out 3-D position RMSE | Held-out median 3-D error |
|---|---:|---:|
| 6 hours | 697 m | 512 m |
| 12 hours | 717 m | 519 m |
| 24 hours | 643 m | 364 m |
| 48 hours | 2295 m | 1419 m |

Position RMSE need not increase monotonically over a finite sample. In this test it rises substantially at 48 hours.

The experiment learns uncertainty, not a position correction. The baseline has one fitted scale. The horizon-only control has four. The RTN model has twelve: three directions at each of four horizons. For each direction and horizon, the fitted squared scale is the training mean of the squared reference-minus-prediction error. We leave the mean at the SGP4 position, so those scales include any uncorrected systematic bias. These are second moments about the predicted position, not sample variances after subtracting a fitted bias.

Reference data are interpolated between 10-second epochs with cubic Hermite interpolation using position and velocity. Reference GPS timestamps are converted to UTC. Earth-fixed reference states are transformed to TEME using Astropy and the previously downloaded IERS table. Errors are expressed in the RTN basis of the SGP4 prediction. There are no online data downloads during the experiment.

Numerical checks passed: maximum difference between Hermite and independent local degree-7 interpolation is below 0.3 mm; reference-frame round trips and explicit GMST82/polar-motion position rotations agree below 0.00001 m. These checks guard against unit and frame errors; they are not independent validation of the source ephemerides. IGS20/IGb20 realization differences are ignored at this pilot error scale.

Important limits:

- All valid supplied cases were retained. No residual clipping or after-the-fact maneuver exclusions were used. There is no verified maneuver log.
- Some training 48-hour position errors reach about 58 km. Their cause is unverified. They strongly influence fitted scales and help explain the conservative test coverage.
- The data cover one month and three related satellites. The model pools their statistics. It has not been validated for other spacecraft.
- Nearby forecasts and Swarm A/C errors can be correlated. A day-block bootstrap favors the directional model in average log score, but only eleven test days are available, so this is exploratory evidence.
- Neither the fixed nor horizon-only baseline is an operational SGP4 covariance product. They are explicit reference models for this experiment.
- This result does not yet validate collision probabilities, rare collision-probability tails, or avoidance maneuvers.

**What we can say:** With this dataset and chronological split, learning the direction and horizon dependence gives a better overall uncertainty description than the specified simple baselines.

**What we cannot say:** SGP4 position accuracy improved, all uncertainty regions are calibrated, maneuvers were excluded, or real collision decisions became more accurate.

Files: `results.png` shows the experiment; `summary.json` contains fitted parameters and scores; `prediction_errors.json` contains every comparison and its model scores; `requirements.txt` records the five top-level package versions. The script prints the completed run output.

To reproduce from the repository root:

```bash
.venv/bin/python experiments/sgp4_uncertainty_pilot/run_experiment.py
```

For another environment, create a Python 3.12 environment and install `requirements.txt`. Preserve the downloaded `data` directory beside `experiments`.

Method references: [Astropy satellite-coordinate example](https://docs.astropy.org/en/stable/coordinates/satellites.html), [python-sgp4 documentation](https://pypi.org/project/sgp4/), and [ESA precise-orbit product description](https://swarm-disc.github.io/product-catalogue-tools/SW_SP3xCOM_2_.html). Exact source URLs and download hashes remain in `data/manifest.json`.
