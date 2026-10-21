#!/usr/bin/env python3
"""Small, chronological SGP4 uncertainty experiment using downloaded Swarm data.

No positions are corrected. No residuals are clipped or maneuver flags inferred.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import csv
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import numpy as np
import scipy
from scipy.interpolate import BarycentricInterpolator
from scipy.stats import chi2
import astropy
from astropy import units as u
from astropy.coordinates import CartesianDifferential, CartesianRepresentation, ITRS, TEME
from astropy.time import Time
from astropy.utils import iers
import erfa
from sgp4.api import Satrec, WGS72, SGP4_ERRORS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DATA = Path(os.environ.get("SGP4_DATA_DIR", PROJECT / "data")).resolve()
HORIZONS = (6, 12, 24, 48)
LEVELS = (0.50, 0.80, 0.90, 0.95, 0.99)
TRAIN_END = datetime(2025, 1, 18, tzinfo=timezone.utc)
TEST_START = datetime(2025, 1, 21, tzinfo=timezone.utc)
REFERENCE_MAP = {"Swarm-A": "L47", "Swarm-B": "L48", "Swarm-C": "L49"}
iers.conf.auto_download = False
iers.conf.auto_max_age = None


def parse_utc(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=110)
def read_sp3(relative_path):
    path = DATA / relative_path
    positions, velocities, epochs = [], [], []
    frames = []
    with path.open() as f:
        header = f.readline()
        assert header[46:51].strip() in {"IGS20", "IGb20"}
        frame = header[46:51].strip()
        for line in f:
            if line.startswith("%c M"):
                assert line.split()[3] == "GPS"
            if line.startswith("* "):
                values = line[1:].split()
                gps = datetime(*map(int, values[:5]), tzinfo=timezone.utc) + timedelta(seconds=float(values[5]))
                # Selected reference period is entirely in Jan/Feb 2025.
                assert datetime(2025, 1, 1, tzinfo=timezone.utc) <= gps < datetime(2025, 2, 5, tzinfo=timezone.utc)
                epochs.append((gps - timedelta(seconds=18)).timestamp())
            elif line.startswith("P"):
                positions.append([float(line[start:start+14])*1000 for start in (4, 18, 32)])
                frames.append(line[1:4])
            elif line.startswith("V"):
                velocities.append([float(line[start:start+14])*0.1 for start in (4, 18, 32)])
    t, r, v = np.array(epochs), np.array(positions), np.array(velocities)
    assert t.size == len(r) == len(v) == 8640
    assert np.allclose(np.diff(t), 10)
    return t, r, v, frames[0], frame


def reference_at(relative_path, unix_time, expected_id):
    times, positions, velocities, sp3_id, frame = read_sp3(relative_path)
    assert sp3_id == expected_id
    k = int(np.searchsorted(times, unix_time, side="right") - 1)
    k = min(k, len(times) - 2)
    assert 0 <= k < len(times)-1 and times[k] <= unix_time <= times[k+1]
    dt = times[k+1] - times[k]
    s = (unix_time - times[k])/dt
    r0, r1, v0, v1 = positions[k], positions[k+1], velocities[k], velocities[k+1]
    r = (2*s**3-3*s**2+1)*r0 + (s**3-2*s**2+s)*dt*v0 + (-2*s**3+3*s**2)*r1 + (s**3-s**2)*dt*v1
    v = ((6*s*s-6*s)*r0 + (3*s*s-4*s+1)*dt*v0 + (-6*s*s+6*s)*r1 + (3*s*s-2*s)*dt*v1)/dt
    # Independent local degree-7 position interpolation diagnoses unit/time bugs.
    lo = max(0, min(k-3, len(times)-8))
    xx = times[lo:lo+8] - times[k]
    poly = BarycentricInterpolator(xx, positions[lo:lo+8], axis=0)
    interpolation_difference_m = float(np.linalg.norm(poly(unix_time-times[k])-r))
    assert interpolation_difference_m < 0.01, interpolation_difference_m
    return r, v, interpolation_difference_m, frame


@lru_cache(maxsize=10)
def read_tles(relative_path):
    lines = (DATA / relative_path).read_text().splitlines()
    return [Satrec.twoline2rv(lines[k], lines[k+1], WGS72) for k in range(0, len(lines), 2)]


def calculate_errors():
    rows = list(csv.DictReader((DATA / "calibration_pairs_index.csv").open()))
    count = len(rows)
    predicted_r = np.empty((count, 3))
    predicted_v = np.empty((count, 3))
    reference_r = np.empty((count, 3))
    reference_v = np.empty((count, 3))
    t = Time([row["comparison_time_utc"] for row in rows], scale="utc")
    interpolation_checks = []
    for i, row in enumerate(rows):
        sat = read_tles(row["tle_file"])[int(row["tle_pair_index_zero_based"])]
        assert sat.satnum == int(row["norad_cat_id"])
        code, r, v = sat.sgp4(t[i].jd1, t[i].jd2)
        if code:
            raise RuntimeError(SGP4_ERRORS[code])
        predicted_r[i], predicted_v[i] = np.asarray(r)*1000, np.asarray(v)*1000
        reference_r[i], reference_v[i], check, frame = reference_at(
            row["reference_file"], float(t[i].unix), REFERENCE_MAP[row["satellite"]])
        interpolation_checks.append(check)
        row["reference_frame"] = frame
    print(f"Loaded reference states for {count} forecasts; transforming coordinate frames.", flush=True)
    table = iers.IERS_A.open(str(DATA / "earth_orientation/finals2000A.all"))
    with iers.earth_orientation_table.set(table):
        itrs_repr = CartesianRepresentation(reference_r.T*u.m, differentials=CartesianDifferential(reference_v.T*u.m/u.s))
        reference_teme = ITRS(itrs_repr, obstime=t).transform_to(TEME(obstime=t))
        ref_r_teme = reference_teme.cartesian.xyz.to_value(u.m).T
        ref_v_teme = reference_teme.cartesian.differentials["s"].d_xyz.to_value(u.m/u.s).T
        roundtrip = reference_teme.transform_to(ITRS(obstime=t)).cartesian.xyz.to_value(u.m).T
        roundtrip_error = np.max(np.linalg.norm(roundtrip-reference_r, axis=1))
        assert roundtrip_error < 1e-5
        # Cross-check TEME->ITRS positions with an explicit Vallado GMST82 rotation.
        gmst = erfa.gmst82(t.ut1.jd1, t.ut1.jd2)
        xp, yp = table.pm_xy(t)
        polar = erfa.pom00(xp.to_value(u.rad), yp.to_value(u.rad), np.zeros(count))
        c, s = np.cos(gmst), np.sin(gmst)
        r = ref_r_teme
        pef = np.column_stack((c*r[:, 0]+s*r[:, 1], -s*r[:, 0]+c*r[:, 1], r[:, 2]))
        explicit = np.einsum("nij,nj->ni", polar, pef)
        rotation_error = np.max(np.linalg.norm(explicit-reference_r, axis=1))
        assert rotation_error < 1e-5
    # Use the predicted RTN basis, which is available without the reference orbit.
    radial = predicted_r / np.linalg.norm(predicted_r, axis=1)[:, None]
    normal = np.cross(predicted_r, predicted_v)
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    along = np.cross(normal, radial)
    basis = np.stack((radial, along, normal), axis=1)
    np.testing.assert_allclose(basis @ np.transpose(basis, (0, 2, 1)), np.broadcast_to(np.eye(3), basis.shape), atol=1e-12)
    errors = np.einsum("nij,nj->ni", basis, ref_r_teme-predicted_r)
    np.testing.assert_allclose(np.linalg.norm(errors, axis=1), np.linalg.norm(ref_r_teme-predicted_r, axis=1), atol=1e-8)
    for i, row in enumerate(rows):
        issue = parse_utc(row["tle_epoch_utc"])
        row["split"] = "train" if issue < TRAIN_END else "test" if issue >= TEST_START else "gap"
        row["issue_day"] = issue.date().isoformat()
        row["error_rtn_m"] = errors[i].tolist()
        row["error_3d_m"] = float(np.linalg.norm(errors[i]))
    max_train_target = max(parse_utc(row["comparison_time_utc"]) for row in rows if row["split"] == "train")
    min_test_issue = min(parse_utc(row["tle_epoch_utc"]) for row in rows if row["split"] == "test")
    assert max_train_target < min_test_issue
    sanity = {"max_interpolation_crosscheck_m": max(interpolation_checks),
              "max_frame_roundtrip_error_m": float(roundtrip_error),
              "max_explicit_rotation_crosscheck_m": float(rotation_error),
              "latest_training_target_utc": max_train_target.isoformat(),
              "earliest_test_issue_utc": min_test_issue.isoformat(),
              "prediction_vectors_unchanged_by_calibration": True,
              "gps_utc_offset_seconds": 18, "sgp4_gravity_model": "WGS72"}
    print(f"Computed {count} reference-minus-SGP4 errors; conversions and interpolation checked.", flush=True)
    return rows, sanity


def fit_and_score(rows):
    errors = np.array([row["error_rtn_m"] for row in rows])
    horizons = np.array([int(row["horizon_hours"]) for row in rows])
    train = np.array([row["split"] == "train" for row in rows])
    test = np.array([row["split"] == "test" for row in rows])
    # Both models keep the predictive mean at the original SGP4 prediction.
    # MLE second moments about zero therefore include any uncorrected bias.
    # Baseline: one isotropic scale across every direction and forecast horizon.
    nominal_sigma = float(np.sqrt(np.mean(errors[train]**2)))
    baseline = np.full_like(errors, nominal_sigma)
    horizon_isotropic = np.zeros_like(errors)
    calibrated = np.zeros_like(errors)
    parameters = {}
    for horizon in HORIZONS:
        mask = train & (horizons == horizon)
        sigma = np.sqrt(np.mean(errors[mask]**2, axis=0))
        calibrated[horizons == horizon] = sigma
        horizon_isotropic[horizons == horizon] = np.sqrt(np.mean(errors[mask]**2))
        parameters[str(horizon)] = {"train_rows": int(mask.sum()), "sigma_rtn_m": sigma.tolist(),
                                    "training_mean_error_rtn_m": errors[mask].mean(axis=0).tolist()}
    model_params = {"baseline": {"isotropic_sigma_m": nominal_sigma, "fitted_parameters": 1},
                    "horizon_isotropic": {"fitted_parameters": 4, "sigma_by_horizon_m": {
                        str(h): float(horizon_isotropic[horizons == h][0, 0]) for h in HORIZONS}},
                    "calibrated": {"by_horizon_hours": parameters, "fitted_parameters": 12}}
    scores = {}
    for name, sigmas in (("baseline", baseline), ("horizon_isotropic", horizon_isotropic), ("calibrated", calibrated)):
        d2 = np.sum((errors/sigmas)**2, axis=1)
        nll = 0.5 * (3*np.log(2*np.pi)+2*np.log(sigmas).sum(axis=1)+d2)
        for i, row in enumerate(rows):
            row[f"{name}_mahalanobis_squared"] = float(d2[i])
            row[f"{name}_nll"] = float(nll[i])
        group_scores = {}
        for key, mask in [("all", test)] + [(str(h), test & (horizons == h)) for h in HORIZONS]:
            observed = [float(np.mean(d2[mask] <= chi2.ppf(q, 3))) for q in LEVELS]
            group_scores[key] = {"test_rows": int(mask.sum()),
                                 "joint_coverage": dict(zip(map(str, LEVELS), observed)),
                                 "mean_absolute_coverage_error": float(np.mean(np.abs(np.array(observed)-LEVELS))),
                                 "mean_negative_log_likelihood": float(nll[mask].mean()),
                                 "mean_95pct_ellipsoid_volume_m3": float(np.mean(4*np.pi/3*chi2.ppf(.95, 3)**1.5*np.prod(sigmas[mask], axis=1)))}
        scores[name] = group_scores
    # Day-block uncertainty estimate: preserve same-day satellite/horizon dependence.
    test_days = sorted({row["issue_day"] for row in rows if row["split"] == "test"})
    rng = np.random.default_rng(20261020)
    block_indices = {day: np.array([i for i, row in enumerate(rows)
                                   if row["split"] == "test" and row["issue_day"] == day]) for day in test_days}
    boot = []
    for _ in range(2000):
        indices = np.concatenate([block_indices[day] for day in rng.choice(test_days, len(test_days), replace=True)])
        nll_gain = np.mean([rows[i]["baseline_nll"] - rows[i]["calibrated_nll"] for i in indices])
        boot.append(nll_gain)
    errors_by_horizon = {}
    for horizon in HORIZONS:
        mask = test & (horizons == horizon)
        errors_by_horizon[str(horizon)] = {"rows": int(mask.sum()),
            "rms_position_error_m": float(np.sqrt(np.mean(np.sum(errors[mask]**2, axis=1)))),
            "median_position_error_m": float(np.median(np.linalg.norm(errors[mask], axis=1))),
            "rms_error_rtn_m": np.sqrt(np.mean(errors[mask]**2, axis=0)).tolist()}
    per_satellite_24 = {}
    for satellite in REFERENCE_MAP:
        selected = [row for row in rows if row["satellite"] == satellite and row["split"] == "test" and int(row["horizon_hours"]) == 24]
        per_satellite_24[satellite] = {"rows": len(selected), **{
            f"{model}_95pct_coverage": float(np.mean([row[f"{model}_mahalanobis_squared"] <= chi2.ppf(.95, 3) for row in selected]))
            for model in ("baseline", "calibrated")}}
    return {"model_parameters": model_params, "test_scores": scores,
            "position_errors_unchanged": errors_by_horizon,
            "per_satellite_24h_coverage": per_satellite_24,
            "day_block_bootstrap": {"blocks": len(test_days), "resamples": 2000,
                "metric": "baseline NLL minus calibrated NLL; positive favors calibrated",
                "approximate_95pct_interval": np.quantile(boot, [.025, .975]).tolist(),
                "caution": "Only eleven test days; temporal dependence can extend beyond one day."}}


def make_plot(summary):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = {"baseline": "#c26b39", "horizon_isotropic": "#4b936d", "calibrated": "#2369a0"}
    labels = {"baseline": "Fixed isotropic", "horizon_isotropic": "Horizon only", "calibrated": "Direction + horizon"}
    for model in colors:
        scores = summary["test_scores"][model]
        axes[0, 0].plot(np.array(LEVELS)*100, [scores["all"]["joint_coverage"][str(q)]*100 for q in LEVELS],
                        "o-", color=colors[model], label=labels[model])
    axes[0, 0].plot([45, 100], [45, 100], "--", color="#777777", label="Ideal coverage")
    axes[0, 0].set(xlabel="Claimed confidence (%)", ylabel="Observed test coverage (%)",
                   title="Does the uncertainty match the errors?", xlim=(45, 101), ylim=(35, 102))
    axes[0, 0].legend(fontsize=9)
    x = np.arange(4)
    for shift, model in ((-.26, "baseline"), (0, "horizon_isotropic"), (.26, "calibrated")):
        y = [summary["test_scores"][model][str(h)]["joint_coverage"]["0.95"]*100 for h in HORIZONS]
        bars = axes[0, 1].bar(x+shift, y, width=.24, color=colors[model], label=labels[model])
    axes[0, 1].axhline(95, ls="--", color="#777777")
    axes[0, 1].set(xticks=x, xticklabels=[f"{h} h" for h in HORIZONS], ylabel="Observed test coverage (%)",
                   title="95% regions: closer to 95% is better", ylim=(0, 110))
    axes[0, 1].legend(fontsize=9, loc="lower left")
    sigmas = summary["model_parameters"]["calibrated"]["by_horizon_hours"]
    for i, (name, color) in enumerate(zip(("Radial", "Along-track", "Cross-track"), ("#4b936d", "#2369a0", "#a0649a"))):
        axes[1, 0].plot(HORIZONS, [sigmas[str(h)]["sigma_rtn_m"][i] for h in HORIZONS], "o-", label=name, color=color)
    axes[1, 0].set(xlabel="Prediction horizon (hours)", ylabel="Fitted 1-sigma scale (m)",
                   title="Uncertainty fitted on training dates")
    axes[1, 0].legend(fontsize=9)
    rms = [summary["position_errors_unchanged"][str(h)]["rms_position_error_m"] for h in HORIZONS]
    axes[1, 1].plot(HORIZONS, rms, "o-", color="#2369a0")
    axes[1, 1].set(xlabel="Prediction horizon (hours)", ylabel="3-D position RMSE (m)",
                   title="Position errors are unchanged by calibration")
    for h, value in zip(HORIZONS, rms):
        axes[1, 1].annotate(f"{value:,.0f} m", (h, value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9)
    fig.suptitle("SGP4 + empirical uncertainty: a small Swarm experiment\nTrain Jan 1-17; gap Jan 18-20; test Jan 21-31, 2025", fontsize=14)
    fig.savefig(HERE / "results.png", dpi=180)
    plt.close(fig)


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    rows, sanity = calculate_errors()
    results = fit_and_score(rows)
    summary = {"description": "SGP4 positions unchanged; compare fitted fixed isotropic, horizon-only isotropic, and direction/horizon RTN uncertainty models.",
               "splits": {name: sum(row["split"] == name for row in rows) for name in ("train", "gap", "test")},
               "training_issue_dates": "2025-01-01 through 2025-01-17",
               "excluded_gap_issue_dates": "2025-01-18 through 2025-01-20",
               "test_issue_dates": "2025-01-21 through 2025-01-31",
               "sanity_checks": sanity, **results,
               "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "astropy", "sgp4", "matplotlib")},
               "source_hashes": {str(path.relative_to(DATA)): fingerprint(path) for path in
                   [DATA / "calibration_pairs_index.csv", DATA / "earth_orientation/finals2000A.all"]},
               "limitations": ["No verified maneuver labels; all supplied valid cases retained, including possible maneuver contamination.",
                   "One month and three related satellites; same-day/nearby samples are correlated.",
                   "Baseline is an explicit fitted model; SGP4 itself supplies no covariance.",
                   "Zero-mean Gaussian assumption is a simplification; fitted scales include uncorrected bias.",
                   "Reference IGS20/IGb20 realizations are treated as ITRS; centimetre-level realization offsets are negligible at these error scales.",
                   "This experiment does not validate collision probabilities or avoidance maneuvers."]}
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (HERE / "prediction_errors.json").write_text(json.dumps(rows, indent=2) + "\n")
    (HERE / "requirements.txt").write_text("\n".join(f"{name}=={version}" for name, version in summary["versions"].items()) + "\n")
    make_plot(summary)
    print(json.dumps({"splits": summary["splits"], "checks": sanity, "scores": summary["test_scores"],
                      "position_error": summary["position_errors_unchanged"], "block_bootstrap": summary["day_block_bootstrap"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
