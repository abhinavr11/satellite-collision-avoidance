# Swarm A/C: reusable shifted trajectory dataset

This standalone bundle contains **only Swarm A and Swarm C**, on a common
constructed timeline. It covers **one hour before through one hour after**
the selected reference crossing, sampled every **one second**: 7,201 states
per satellite in each trajectory table. The original source SP3 files contain
full days at 10-second spacing; the exported reference states are interpolated,
not independent one-second observations.

## Files

| File | Contents |
|---|---|
| `aligned_trajectories.csv` | **Recommended for comparison:** one row per common timestamp, with reference and SGP4 positions/velocities for both satellites; 7,201 rows. |
| `reference_trajectories.csv` | Reference positions and velocities for both objects; 14,402 rows. |
| `sgp4_trajectories.csv` | Uncorrected SGP4 positions and velocities from the two selected TLEs; identical schema and times. |
| `encounter.json` | Time mapping, identity, units, source paths/hashes, original TLE epochs, checks, and center geometry. |
| `sources/` | The two original daily SP3 files, the two selected native TLE records, and the downloaded Earth-orientation table. |
| `load_trajectories.py` | NumPy loader; no dependence on the original project layout. |
| `rebuild.py` | Offline rebuild and numerical checks using the bundled source files. |
| `export_aligned.py` | Recreate the combined four-trajectory table from the two separate tables, checking exact timestamp alignment. |
| `plot_trajectories.py` | Recreate the preview from the saved tables. |
| `trajectory_preview.png` | Full trajectory view and separation near the reference crossing. |
| `requirements.txt` | Versions used to generate the bundle. |

The source collection and other satellites are outside this folder. Every input
needed for rebuilding these two tables is copied into `sources/`; no symlink or
temporary search file is needed after copying this whole bundle elsewhere.

## Time mapping and coordinate frame

The common center is **2025-01-30 20:54:01.692525 UTC**. The exported window is
19:54:01.692525–21:54:01.692525 UTC on that synthetic date.

| Object | Original center (UTC) | Add to original time to obtain synthetic time |
|---|---|---:|
| Swarm A, NORAD 39452 | 2025-01-30 20:54:01.692525 | 0 s |
| Swarm C, NORAD 39453 | 2025-02-01 02:33:11.651308 | −106,749.958783 s |

All exported positions are **GCRS Cartesian coordinates in metres**. Velocities
are in **metres per second**, including the velocity change from transforming
the rotating terrestrial frame. Each state is transformed at its **original
source time**, then relabelled with the synthetic time. There is no artificial
position displacement. UTC, GPS conversion, interpolation and frame assumptions
are recorded in the metadata.

The original TLE text remains unchanged. Propagate it at `source_time_utc`, then
transform TEME to native-time GCRS and apply the time labels above. **Do not edit
the TLE epoch or propagate the native C TLE directly at the synthetic UTC.**
The synthetic epoch field in the metadata is only for forecast-age bookkeeping.
The two TLE forecast ages at the center are approximately 23.8003 and 30.0432 h;
both remain within 6–48 h throughout the exported window.

At `elapsed_seconds = 0`, the reference separation is approximately **0.286 m**
and relative speed is **248.10 m/s**. The uncorrected SGP4 separation at that same
time is approximately **443.51 m**. The center was selected using reference
trajectories; it is not necessarily the SGP4 time of closest approach.

## CSV schema

### Combined table: all four trajectories

Use `aligned_trajectories.csv` to compare the objects at the same simulated time.
Its single `time_utc` column applies to **all four trajectories** in each row:

- `swarm_a_reference_x_m`, `swarm_a_reference_y_m`, `swarm_a_reference_z_m`
- `swarm_a_sgp4_x_m`, `swarm_a_sgp4_y_m`, `swarm_a_sgp4_z_m`
- `swarm_c_reference_x_m`, `swarm_c_reference_y_m`, `swarm_c_reference_z_m`
- `swarm_c_sgp4_x_m`, `swarm_c_sgp4_y_m`, `swarm_c_sgp4_z_m`

Each prefix also has `vx_m_s`, `vy_m_s`, and `vz_m_s`. The two original timestamps
are stored as `swarm_a_source_time_utc` and `swarm_c_source_time_utc`; the forecast
ages are `swarm_a_tle_age_hours` and `swarm_c_tle_age_hours`. Values are copied
exactly from the separate trajectory tables, with no further interpolation.

Here `reference` means the original **GPS-based processed precise orbit** from
ESA SP3 files, transformed to GCRS and interpolated to the shared sample grid.
Raw GPS receiver observations are not part of the downloaded data. The original
unshifted SP3 files for both satellites are also stored in `sources/`.

### Separate reference and prediction tables

These two CSVs are sorted by `elapsed_seconds`, then satellite. There is one row per
satellite per time. Filter by `satellite` before interpolation or plotting.

| Columns | Meaning |
|---|---|
| `satellite`, `norad_id` | Object identity. |
| `elapsed_seconds` | Integer seconds from the shared reference crossing; zero is included. |
| `time_utc` | Common synthetic timestamp, ISO 8601 with `Z`. |
| `source_time_utc` | Original native timestamp before the shift; ISO 8601 with `Z`. |
| `tle_age_hours` | Source time minus original TLE epoch, in hours. |
| `x_m`, `y_m`, `z_m` | Geocentric GCRS position in metres. |
| `vx_m_s`, `vy_m_s`, `vz_m_s` | GCRS velocity in metres per second. |

## Load and use

From inside this folder:

```python
import numpy as np
from load_trajectories import load_trajectories, load_metadata

reference = load_trajectories()        # defaults to reference
prediction = load_trajectories('sgp4')
metadata = load_metadata()

a, c = reference['Swarm-A'], reference['Swarm-C']
t = a['elapsed_seconds']             # shape (7201,)
r_a = a['position_m']                 # shape (7201, 3)
v_a = a['velocity_m_s']               # shape (7201, 3)
distance_m = np.linalg.norm(r_a - c['position_m'], axis=1)
print(distance_m[t == 0])
```

From the project root, add the dataset folder to Python's import path first:

```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path('data/encounters/swarm_ac_shifted_v1').resolve()))
from load_trajectories import load_trajectories
```

The loader verifies each trajectory file's SHA-256 before reading it. Plain CSV
also works with pandas, MATLAB, Julia, or any other CSV reader.

## Rebuild and verify

Use Python 3.12 with `requirements.txt`. Run inside this folder:

```sh
python rebuild.py
python load_trajectories.py
python plot_trajectories.py
```

Rebuilding requires no network connection. Checks verify source hashes, satellite
IDs, UTC/GPS handling, interpolation against a local degree-seven polynomial,
frame round trips, velocities against position derivatives, and agreement with
the previously selected reference crossing. Output checksums and numerical
results are written to `encounter.json`.

Reference velocities are derivatives of the Hermite reference trajectory before
frame conversion. SGP4 velocities are the values supplied by the propagator,
then transformed with the positions. SGP4's supplied velocity can differ slightly
from a numerical derivative of its positions; both native and transformed
derivative residuals are recorded in the checks. They are not silently replaced
by a different velocity definition.

## Interpretation and source provenance

This is a **constructed encounter from real trajectory segments**, not a recorded
historical conjunction or collision. Precise reference orbits are approximate
truth. Sub-metre interpolated separation does not establish sub-metre absolute
reference accuracy. One-second sampling does not increase source measurement
accuracy.

Reference states come from the downloaded ESA Swarm reduced-dynamic SP3 orbit
products (baseline 0203). Their original headers, object IDs, terrestrial frames,
and GPS times are preserved in the copied SP3 files. IGS20/IGb20 are treated as
ITRS for these transformations, matching the existing project's convention.
Historical TLE provenance remains recorded by the original relative data path,
record index and history hash in `encounter.json`. Data-provider terms remain
applicable; this bundle assigns no new license to source data.

This bundle contains trajectories only. It has no fitted bias, covariance,
collision probability, object-radius assumption, or avoidance maneuver.
