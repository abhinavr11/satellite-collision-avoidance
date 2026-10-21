> This is the guide to the original full data collection from October 8, 2026. The GitHub version does not contain those datasets. Run `scripts/download_data.py` for core experiment inputs, or add `--all` for the full layout described below. Live snapshot counts can differ on a new download.

# Project data - start here

Downloaded on October 8, 2026 for the satellite collision-avoidance proposal.

**For the clean-versus-noisy comparison, start with the Swarm files.** The precise
reference trajectories are the best available approximation to truth. Historical
TLEs supply the less accurate orbit predictions after you propagate them with SGP4.
No artificial noise was added.

## What is included

| Folder or file | What it contains | Use |
|---|---|---|
| `orbit_predictions/historical/` | Complete downloaded TLE histories for Swarm A, B, and C; 29,279 records before removing duplicates | Historical prediction inputs |
| `orbit_predictions/calibration_window/` | Deduplicated TLE subsets around January 2025 | Small starting dataset |
| `reference_orbits/swarm_A/`, `swarm_B/`, `swarm_C/` | 102 ESA precise-orbit daily ZIPs, plus unpacked SP3 and HDR files | Accurate reference position and velocity |
| `reference_orbits/index.csv` | Time coverage, satellite ID, units, and reference frame of each SP3 | Find the right reference file |
| `calibration_pairs_index.csv` | 980 prediction/reference matching jobs at horizons of 6, 12, 24, and 48 hours | Know which TLE, future timestamp, and reference file to compare |
| `conjunctions/esa/` | Full ESA challenge release: training, testing, released test labels, and compressed original data | Historical collision-risk and covariance analysis |
| `conjunctions/socrates_snapshot.csv` | 189,612 close-approach screening records from the downloaded CelesTrak snapshot | Additional real encounter cases |
| `conjunctions/current_cases_with_orbits.csv` | 100 screening cases for which both objects have downloaded orbit elements | Starting cases for orbital propagation and maneuver experiments |
| `orbit_predictions/current_case_objects_gp.csv` | Orbit elements for the 191 objects in those 100 cases | Initialize both objects' nominal trajectories |
| `orbit_predictions/current_active_gp.csv` | Downloaded active-satellite orbit snapshot, 16,689 objects | Source catalog for the selected cases |
| `space_weather/` | GFZ Kp, ap, Ap, solar flux, and sunspot history | Atmospheric-drag inputs and optional uncertainty features |
| `earth_orientation/` | IERS Earth-orientation parameters and leap seconds | Convert coordinate frames and time scales |
| `documentation/`, `metadata/` | Provider descriptions, licenses, source listings, and metadata | Interpret and cite the data |

The reference files contain **881,280 position/velocity epochs**, sampled every
10 seconds. Their combined UTC coverage is December 31, 2024 at 23:59:42 through
February 3, 2025 at 23:59:32. The main calibration period is January 2025; the
extra days allow comparisons up to 48 hours later.

The ESA training CSV has **162,634 rows and 13,154 events**; the test CSV has
**24,484 rows and 2,167 events**. The released private-label file has one row per
test event. All three have 103 columns.

## How to use the matching index

Each row in `calibration_pairs_index.csv` tells you:

1. The satellite and the TLE file to open.
2. Which two-line TLE record to read (`tle_pair_index_zero_based`: record 0 is the first pair).
3. The future UTC time at which to evaluate the SGP4 prediction.
4. The precise reference file containing that time.

Interpolate the reference trajectory at the comparison time, transform both
positions into the same frame, and calculate reference minus prediction. The
index links files and times; it does **not** contain calculated position errors
or an already-fitted uncertainty model.

Satellite identities have been checked:

| Satellite | NORAD ID | SP3 ID |
|---|---|---|
| Swarm A | 39452 | L47 |
| Swarm B | 39451 | L48 |
| Swarm C | 39453 | L49 |

## Details that matter when comparing values

- SP3 timestamps are **GPS time**; the index uses UTC. Throughout this selected
  interval, GPS is 18 seconds ahead of UTC. Do not subtract 18 seconds twice.
- Read each SP3 header: the downloaded files use terrestrial frames **IGS20**
  or **IGb20**. SGP4 returns TEME. Transform before subtracting coordinates.
- SP3 positions are in **km** and velocities in **dm/s**. Multiply by 1000 and
  0.1 respectively to obtain metres and metres per second.
- The filename dates are UTC; the epoch records inside SP3 are GPS time.
- The ESA historical CDM table uses metres and metres per second for relative
  states. Its `risk` and `true_risk` values are base-10 logarithms of estimated
  collision probabilities. `true_risk` is a final estimate, not an observed collision.
- ESA covariance fields include standard deviations and correlations. A
  correlation must be multiplied by the two standard deviations to obtain a
  covariance entry. Preserve each object's RTN frame when combining them.
- ESA `t_span` and `c_span` describe object size used by the risk algorithm; the
  provider describes the chaser minimum as a diameter. Check size conventions
  before setting a combined hard-body radius.
- SOCRATES range, speed, and dilution values use **km**, **km/s**, and **km**.
  Its `MAX_PROB` is a maximum-probability screening value, not a calibrated
  collision probability to treat as ground truth.
- The 100 current cases exclude relative speeds below 1 km/s and require both
  orbit-element epochs to precede the listed TCA. They are a convenience subset,
  not independently validated dangerous encounters. Recompute TCA with your model.

## What still needs modeling or screening

The downloaded data support the core experiments, but do not supply every modeling
decision. Set an acceptable collision-risk threshold, define the nominal covariance
baseline, and justify object-size and drag assumptions where measured values are
unavailable. TLEs do not include a full state covariance or an object's physical
drag coefficient and area-to-mass ratio. Estimate covariance from the matched
position/velocity errors when propagating a six-dimensional state uncertainty.

No authoritative Swarm maneuver history was obtained in this download. Exclude or
separately analyze known or detected maneuver intervals before fitting ordinary
prediction-error statistics. Check the reference product headers and data quality.
Use chronological calibration/test splits with a gap at least as long as the
largest prediction horizon; nearby error samples are correlated.

The anonymous ESA historical events are a separate dataset from the identified
Swarm calibration records. They cannot be joined by satellite ID or calendar date.
They also lack full absolute state vectors for exact historical nonlinear maneuver
replay. The current SOCRATES cases have both nominal orbits available, but their
covariances and physical sizes must be estimated or explicitly assumed. Do not
claim that a Swarm-calibrated uncertainty model is validated for unrelated objects
or for earlier years without additional evidence.

## Sources and attribution

- Precise reference trajectories: [ESA Swarm archive](https://swarm-diss.eo.esa.int/),
  `Level2daily/Latest_baselines/POD/RD/Sat_A`, `Sat_B`, and `Sat_C`; highest listed
  processing version per day, baseline 0203. Product description:
  [ESA/Delft precise reduced-dynamic orbits](https://swarm-disc.github.io/product-catalogue-tools/SW_SP3xCOM_2_.html),
  DOI [10.57780/esa-8579355](https://doi.org/10.57780/esa-8579355).
  Follow ESA's linked data policy and terms. These are retrospective reference
  products; they were not available to the original historical prediction.
- Historical TLEs: the research authors' [orbit-prediction dataset repository](https://github.com/Jeong-Seung-Won/-IEEE-Access-Toward-Data-Driven-Satellite-Orbit-Prediction),
  pinned to commit `b1473e2cddbe660f0fde66c3abb8315beadb7653`. The authors identify
  Space-Track as the original source. The repository's MIT license and README
  are preserved. This is a research redistribution, not a fresh authenticated
  download directly from Space-Track.
- Historical conjunctions: [ESA Collision Avoidance Challenge dataset](https://zenodo.org/records/4463683),
  DOI [10.5281/zenodo.4463683](https://doi.org/10.5281/zenodo.4463683), CC BY 4.0.
  Provider column definitions are saved in `documentation/esa_columns.html` and
  the raw dataset's accompanying TXT. Retain attribution to the dataset authors.
- Current conjunctions and orbital elements: [CelesTrak SOCRATES](https://celestrak.org/SOCRATES/)
  and [GP elements](https://celestrak.org/NORAD/elements/). Formats and usage policy
  are saved locally. These are frozen download snapshots; do not treat them as a
  live catalog or mix them with the January 2025 historical calibration inputs.
- Space weather: [GFZ Kp data](https://kp.gfz.de/en/data), DOI
  [10.5880/Kp.0001](https://doi.org/10.5880/Kp.0001). The file's license header
  specifies CC BY 4.0, except sunspot numbers, which use CC BY-NC 4.0.
- Earth orientation: [IERS finals2000A](https://datacenter.iers.org/data/9/finals2000A.all)
  and [IERS leap-second table](https://hpiers.obspm.fr/iers/bul/bulc/Leap_Second.dat).

## Reproduce and verify

From the repository root, using Python 3.9 or later:

```bash
python3 scripts/download_data.py --all
python3 scripts/prepare_data.py
```

`manifest.json` records each original download's URL, timestamp, size, and SHA-256.
The ESA archive also passed its provider's published MD5 check. The preparation
script verified all downloaded hashes, TLE checksums and Git blob identities,
SP3 satellite IDs, epoch counts, sample spacing, physically plausible states,
and CSV row structure. Results are in `validation_summary.json`.
`prepared_files_manifest.json` hashes unpacked data and generated indexes.
`download_failures.json` is empty after successful completion.

These checks establish file integrity and basic data consistency. Frame
transformations, propagated errors, covariance calibration, and the scientific
experiments still need to be implemented and validated.
