# Final validation

Validated October 29, 2026. These checks establish implementation and reproducibility properties; they do not establish calibrated operational collision probabilities.

## Current implementation

- The complete five-experiment pipeline passed using the original downloaded source collection and the pinned Python 3.12 runtime dependencies.
- All 31 unit tests passed. They cover analytic Gaussian/chi-square integrals, rotation invariance, covariance/projection identities, continuous closest approach, stable log-probability underflow, Monte Carlo sampling checks, quadratic constraints, chronological splitting, coordinate-origin translation, confidence gates, exact preservation of withheld corrections, and reference-data leakage checks.
- All three saved-result validators passed: original variance/correction regressions, historical-calibration collision results, and the balanced within-encounter forecasts.
- An empty temporary data location reconstructed the Swarm A/C bundle using only copied core source files and the checked-in recipe. Reference and SGP4 trajectory tables matched the saved bundle byte for byte.
- An explicit two-minute fixed-window/residual-covariance run reproduced the earlier result. The original residual-only comparisons remain in the final balanced report.

## Final encounter setup

- The three-minute forecast lead is selected from candidate leads of 2–6 minutes using only reference data through the common earliest cutoff, six minutes before the crossing. The fixed rule requires at least 75% rolling position-RMSE reduction and the existing 99% conditional improvement gate for both objects. Neither candidate test errors nor Pc select the window.
- The selected case has 3421 available training-prefix timestamps and 360 test timestamps per object. The test covers three minutes before through three minutes after the crossing. Recent fitting-window choices and all gate decisions are made inside training and frozen before test.
- Test perturbation checks confirm that changing future reference states cannot change lead selection, fitting windows, model coefficients, calibration, gates, predictions or covariance.
- The reference paths determine the binary collision label. No reference covariance or reference collision probability is assigned.
- Primary forecasts explicitly use the second-moment error budget S = P + (mu−d)(mu−d)^T as a zero-mean Gaussian approximation. Saved controls use the original centred residual covariance P. Tests verify this identity against Gaussian samples and confirm that uncertainty-mode changes leave mean trajectories unchanged.
- Saved-result checks reconstruct applied corrections and both uncertainty matrices, recompute Pc, verify proper binary scores, and audit held-out position errors against the saved trajectories. No check requires the corrected method to win.

## Packaging and publication

Python sources parse, local Markdown links resolve, diffs pass whitespace checks, and no machine-specific source paths or downloaded data are included in the deliverable file set. The GitHub preflight checks the staged blobs, required implementation files, source notices, forbidden paths, symlinks, credential patterns and ordinary-file size limits. Downloaded data and environments remain ignored.

The finalize-for-push workflow also verifies the exact staged tree from a separate temporary checkout, running all unit tests and saved-result validators without access to downloaded inputs. The code and results are then committed locally; publishing is left to the user.

## Earlier clean-install evidence

The original standalone package was validated October 21, 2026 in a fresh Python 3.12 environment. Its downloader fetched all 108 core source files into an empty external location, recreated the 980-row comparison index, and verified 317 source/prepared core files. The original three experiments reproduced their saved numerical metrics and confidence-gated selection counts. Optional full-data mode was also exercised using verified local source caches; not every optional live endpoint was freshly downloaded.

Historical products are hash-pinned. Live Earth-orientation inputs can change upstream; their actual hash is recorded. Some saved historical experiment metadata describe an earlier IERS snapshot, while current collision results identify the snapshot used for those calculations.

## Scientific limits

This is one previously inspected, constructed encounter. Nearby one-second reference states are interpolated and strongly correlated. Gaussian tails, independent object errors, physical radius assumptions, maneuver screening and parameter uncertainty are not independently validated. Training-only backtests and more moderate-looking Pc values do not establish general probability calibration. No avoidance-maneuver optimization is implemented.
