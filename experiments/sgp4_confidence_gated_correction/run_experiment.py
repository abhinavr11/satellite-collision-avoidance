#!/usr/bin/env python3
"""Training-only, variance-weighted position correction with a fixed 80% gate.

The gate estimates P(||error-correction|| < ||error||) under a fitted Gaussian.
It is not a guarantee, or confidence in the estimated model parameters.
"""
from pathlib import Path
import importlib.util
import hashlib
import json
import numpy as np
from scipy.stats import norm
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import CartesianRepresentation, ITRS, TEME
from astropy.utils import iers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
UNCERTAINTY = HERE.parent / "sgp4_uncertainty_pilot"
PREVIOUS = HERE.parent / "sgp4_position_correction_pilot"
spec = importlib.util.spec_from_file_location("uncertainty_pilot", UNCERTAINTY / "run_experiment.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
THRESHOLD = 0.80


def fit_models(training):
    models = {}
    for satellite in pilot.REFERENCE_MAP:
        for horizon in pilot.HORIZONS:
            group = [r for r in training if r["satellite"] == satellite and int(r["horizon_hours"]) == horizon]
            error = np.array([r["error_rtn_m"] for r in group])
            mean = error.mean(axis=0)
            variance = error.var(axis=0, ddof=1)
            weights = mean**2 / (mean**2 + variance)
            candidate = weights*mean
            projected_sigma = float(np.sqrt(np.dot(candidate**2, variance)))
            p = float(norm.cdf((mean @ candidate - .5*(candidate @ candidate))/projected_sigma)) if projected_sigma else .5
            assert .5 <= p <= 1 and np.all((0 <= weights) & (weights <= 1))
            models[f"{satellite}/{horizon}"] = {
                "satellite": satellite, "horizon_hours": horizon, "training_forecasts": len(group),
                "training_issue_days": len({r["issue_day"] for r in group}),
                "mean_error_rtn_m": mean.tolist(), "variance_rtn_m2": variance.tolist(),
                "correction_weights": weights.tolist(), "candidate_correction_rtn_m": candidate.tolist(),
                "estimated_improvement_probability": p, "apply_correction": p >= THRESHOLD,
            }
    return models


def metrics(rows, kind):
    if not rows:
        return {"forecasts": 0}
    before = np.array([r["original_error_m"] for r in rows])
    after = np.array([r[f"{kind}_error_m"] for r in rows])
    rmse_before, rmse_after = np.sqrt(np.mean(before**2)), np.sqrt(np.mean(after**2))
    return {"forecasts": len(rows), "original_rmse_m": float(rmse_before), "corrected_rmse_m": float(rmse_after),
            "rmse_reduction_percent": float(100*(1-rmse_after/rmse_before)),
            "original_mean_error_m": float(before.mean()), "corrected_mean_error_m": float(after.mean()),
            "original_median_error_m": float(np.median(before)), "corrected_median_error_m": float(np.median(after)),
            "improved": int(np.sum(after < before-1e-7)), "worsened": int(np.sum(after > before+1e-7)),
            "unchanged": int(np.sum(np.abs(after-before) <= 1e-7))}


def check_analytic_probability(models):
    rng = np.random.default_rng(20261020)
    result = {}
    for key, model in models.items():
        if not model["apply_correction"]:
            continue
        mean = np.array(model["mean_error_rtn_m"])
        sigma = np.sqrt(model["variance_rtn_m2"])
        correction = np.array(model["candidate_correction_rtn_m"])
        simulated = rng.normal(mean, sigma, size=(100000, 3))
        empirical = float(np.mean(np.sum((simulated-correction)**2, axis=1) < np.sum(simulated**2, axis=1)))
        assert abs(empirical-model["estimated_improvement_probability"]) < .01
        result[key] = {"analytic": model["estimated_improvement_probability"], "monte_carlo": empirical}
    return result


def main():
    source = UNCERTAINTY / "prediction_errors.json"
    rows = json.loads(source.read_text())
    training = [r for r in rows if r["split"] == "train"]
    testing = sorted([r for r in rows if r["split"] == "test"], key=lambda r: (r["tle_epoch_utc"], r["satellite"], int(r["horizon_hours"])))
    assert max(pilot.parse_utc(r["comparison_time_utc"]) for r in training) < min(pilot.parse_utc(r["tle_epoch_utc"]) for r in testing)
    # Fit decisions before consulting any test residuals.
    models = fit_models(training)
    monte_carlo_checks = check_analytic_probability(models)
    old = json.loads((PREVIOUS / "corrected_positions.json").read_text())
    old_lookup = {(r["satellite"], r["tle_epoch_utc"], r["horizon_hours"]): r for r in old}
    output = []
    for row in testing:
        key = f"{row['satellite']}/{row['horizon_hours']}"
        model = models[key]
        error = np.array(row["error_rtn_m"])
        candidate = np.array(model["candidate_correction_rtn_m"])
        applied = candidate if model["apply_correction"] else np.zeros(3)
        time = Time(row["comparison_time_utc"], scale="utc")
        sat = pilot.read_tles(row["tle_file"])[int(row["tle_pair_index_zero_based"])]
        code, r, v = sat.sgp4(time.jd1, time.jd2)
        assert code == 0
        r, v = np.array(r)*1000, np.array(v)*1000
        radial = r/np.linalg.norm(r)
        normal = np.cross(r, v)
        normal /= np.linalg.norm(normal)
        basis = np.stack((radial, np.cross(normal, radial), normal))
        cached = old_lookup[row["satellite"], row["tle_epoch_utc"], int(row["horizon_hours"])]
        reference = np.array(cached["reference_position_teme_m"])
        np.testing.assert_allclose(r, cached["sgp4_position_teme_m"], atol=1e-8, rtol=0)
        new_position = r + basis.T @ applied
        actual_error = float(np.linalg.norm(reference-new_position))
        np.testing.assert_allclose(actual_error, np.linalg.norm(error-applied), atol=1e-7)
        if not model["apply_correction"]:
            assert np.array_equal(new_position, r)
        output.append({"satellite": row["satellite"], "horizon_hours": int(row["horizon_hours"]),
                       "tle_epoch_utc": row["tle_epoch_utc"], "comparison_time_utc": row["comparison_time_utc"],
                       "reference_file": row["reference_file"], "model_key": key,
                       "estimated_improvement_probability": model["estimated_improvement_probability"],
                       "changed": model["apply_correction"], "original_error_m": row["error_3d_m"],
                       "gated_error_m": actual_error, "weighted_always_error_m": float(np.linalg.norm(error-candidate)),
                       "mean_always_error_m": float(np.linalg.norm(error-np.array(model["mean_error_rtn_m"]))),
                       "candidate_correction_rtn_m": candidate.tolist(), "applied_correction_rtn_m": applied.tolist(),
                       "original_position_teme_m": r.tolist(), "gated_position_teme_m": new_position.tolist(),
                       "reference_position_teme_m": reference.tolist()})
    changed = [r for r in output if r["changed"]]
    examples = changed[:6]
    eop = iers.IERS_A.open(str(pilot.DATA / "earth_orientation/finals2000A.all"))
    reference_checks = []
    with iers.earth_orientation_table.set(eop):
        for row in examples:
            time = Time(row["comparison_time_utc"], scale="utc")
            r_itrs, _, _, _ = pilot.reference_at(row["reference_file"], float(time.unix), pilot.REFERENCE_MAP[row["satellite"]])
            ref = ITRS(CartesianRepresentation(r_itrs*u.m), obstime=time).transform_to(TEME(obstime=time)).cartesian.xyz.to_value(u.m)
            diff = float(np.linalg.norm(ref-np.array(row["reference_position_teme_m"])))
            assert diff < 1e-5
            reference_checks.append(diff)
    comparisons = {kind: metrics(output, kind) for kind in ("mean_always", "weighted_always", "gated")}
    by_horizon = {str(h): {**metrics([r for r in output if r["horizon_hours"] == h], "gated"),
                            "changed": sum(r["changed"] for r in output if r["horizon_hours"] == h)} for h in pilot.HORIZONS}
    groups = {key: {**model, "test_results": metrics([r for r in output if r["model_key"] == key], "gated")} for key, model in models.items()}
    summary = {"threshold": THRESHOLD, "rule": "Variance-weighted mean correction; apply only if fitted Gaussian probability of reducing 3-D error is at least 0.80.",
               "formula_weights": "w_j = mean_j^2 / (mean_j^2 + variance_j); d_j = w_j * mean_j",
               "formula_probability": "p = Phi((mean dot d - ||d||^2/2) / sqrt(sum_j variance_j*d_j^2))",
               "training_dates": "2025-01-01 through 2025-01-17", "test_dates": "2025-01-21 through 2025-01-31",
               "changed_forecasts": len(changed), "unchanged_forecasts": len(output)-len(changed),
               "comparisons": comparisons, "selected_only": metrics(changed, "gated"),
               "by_horizon": by_horizon, "groups": groups, "first_six_changed_examples": examples,
               "checks": {"training_only_rule": True, "unchanged_positions_exactly_preserved": True,
                          "analytic_probability_monte_carlo": monte_carlo_checks,
                          "max_reloaded_example_reference_difference_m": max(reference_checks, default=0)},
               "source_prediction_errors_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "limitations": ["The reported probability is an uncalibrated plug-in Gaussian model estimate, not a validated 80% guarantee.",
                    "Only 42-47 training forecasts over 17 issue dates per group, with temporal and cross-satellite dependence.",
                    "Confidence is constant within each satellite/horizon group, not a case-specific anomaly detector.",
                    "The test period was used in previous exploratory experiments; a new independent period is needed for confirmation.",
                    "Satellite-specific fitting and variance shrinkage are both changes relative to the earlier pooled mean correction.",
                    "No verified maneuver exclusions, no rare-risk validation, and no guarantee of trajectory consistency."]}
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (HERE / "corrected_positions.json").write_text(json.dumps(output, indent=2)+"\n")
    write_report(summary)
    plot(summary, output)
    print(json.dumps({key: summary[key] for key in ["changed_forecasts", "unchanged_forecasts", "selected_only", "comparisons", "by_horizon", "checks"]}, indent=2))


def write_report(s):
    overall = s["comparisons"]["gated"]
    selected = s["selected_only"]
    lines = ["# Correct only when the variance model supports a confident correction", "",
             "The 80% model-estimated improvement threshold was fixed before scoring this experiment's test outcomes. Training dates are January 1-17, 2025; testing uses the same January 21-31 exploratory period as the earlier experiments. There is a January 18-20 gap.", "",
             "We fitted separate mean errors and diagonal variances for each satellite and horizon. This uses 42-47 training forecasts per group, spread across 17 issue dates. Variance alone supplies no correction direction. The training mean supplies that direction.", "",
             "We first shrink each mean correction component with w_j = mean_j^2 / (mean_j^2 + variance_j), and d_j = w_j mean_j. This is a simple heuristic: a large random spread relative to systematic bias makes that component small. It is not an optimality claim.", "",
             "Under the fitted model e ~ N(mean, diagonal variance), a correction d reduces squared 3-D distance error exactly when e dot d > ||d||^2/2. Therefore its estimated success probability is Phi((mean dot d - ||d||^2/2) / sqrt(sum variance_j*d_j^2)). We apply d only if this estimate is at least 0.80. Otherwise the original position is preserved exactly.", "",
             f"**Selected {s['changed_forecasts']} of {overall['forecasts']} test forecasts. Left {s['unchanged_forecasts']} unchanged.** Among the selected forecasts, {selected.get('improved',0)} improved and {selected.get('worsened',0)} worsened.", "",
             "| Selected group | Estimated chance of improvement | Actual improved / changed | Original selected-group RMSE | After correction |",
             "|---|---:|---:|---:|---:|"]
    for key, model in s["groups"].items():
        if model["apply_correction"]:
            m = model["test_results"]
            lines.append(f"| {model['satellite']}, {model['horizon_hours']} h | {100*model['estimated_improvement_probability']:.1f}% | {m['improved']}/{m['forecasts']} | {m['original_rmse_m']:.2f} m | {m['corrected_rmse_m']:.2f} m |")
    lines += ["", "All other satellite/horizon groups were left unchanged. The gate selects groups, so all forecasts for a given satellite at a given horizon receive the same confidence estimate. It does not identify which individual forecasts within that group will fail.", "",
              "The following comparisons isolate the effect of the gate while keeping the satellite-specific fitting fixed:", "",
              "| Policy | Overall 3-D RMSE | Improved | Worsened | Unchanged |", "|---|---:|---:|---:|---:|"]
    for kind, label in [("mean_always", "Always apply full group mean"), ("weighted_always", "Always apply variance-weighted correction"), ("gated", "Apply weighted correction only above 80%")]:
        m=s["comparisons"][kind]
        lines.append(f"| {label} | {m['corrected_rmse_m']:.2f} m | {m['improved']} | {m['worsened']} | {m['unchanged']} |")
    lines += ["", f"Original uncorrected overall RMSE: {overall['original_rmse_m']:.2f} m. Gated overall RMSE: {overall['corrected_rmse_m']:.2f} m ({overall['rmse_reduction_percent']:.2f}% reduction). Original overall mean distance error: {overall['original_mean_error_m']:.2f} m; gated: {overall['corrected_mean_error_m']:.2f} m. Original median distance error: {overall['original_median_error_m']:.2f} m; gated: {overall['corrected_median_error_m']:.2f} m.", "",
              "The first six changed test forecasts, selected by chronological issue time regardless of outcome:", "",
              "| Satellite | Horizon | Issue time (UTC) | Original error | Gated error | Result |", "|---|---:|---|---:|---:|---|"]
    for row in s["first_six_changed_examples"]:
        lines.append(f"| {row['satellite']} | {row['horizon_hours']} h | {row['tle_epoch_utc']} | {row['original_error_m']:.2f} m | {row['gated_error_m']:.2f} m | {'Improved' if row['gated_error_m']<row['original_error_m'] else 'Worsened'} |")
    lines += ["", "Interpretation: a selective rule trades the size and frequency of corrections against the risk of worsening individual positions. Its fitted confidence is only a model estimate. An 80% probability of helping would still permit some failures even under a correct model. Here Gaussian tails, estimated parameters, time variation, and unknown maneuvers add further uncertainty.", "",
              "Important controls and limits:", "", *[f"- {item}" for item in s["limitations"]], "",
              "All positions are in TEME, in metres. The script recomputed SGP4 positions, applied the rotated RTN corrections, and checked Cartesian error norms against RTN errors. It reloaded the first six changed reference states from SP3 files. A Monte Carlo check also confirmed the analytic success-probability formula under its assumed Gaussian model; this validates the calculation, not the real-data probability model.", "",
              "Run from the Project directory using the existing environment:", "", "```bash",
              ".venv-sgp4-pilot/bin/python experiments/sgp4_confidence_gated_correction/run_experiment.py", "```", "",
              "`summary.json` stores all twelve fitted group models, their gates, scores, and input hash. `corrected_positions.json` stores all 344 test predictions, their confidence estimates, and corrected or preserved positions. Earlier experiment files and raw data remain unchanged.", ""]
    (HERE / "report.md").write_text("\n".join(lines))


def plot(s, rows):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    x=np.arange(4)
    groups=[s["by_horizon"][str(h)] for h in pilot.HORIZONS]
    unchanged=np.array([m["unchanged"] for m in groups])
    improved=np.array([m["improved"] for m in groups])
    worse=np.array([m["worsened"] for m in groups])
    axes[0].bar(x, unchanged, color="#b7c2cc", label="Left unchanged")
    axes[0].bar(x, improved, bottom=unchanged, color="#388465", label="Changed: improved")
    axes[0].bar(x, worse, bottom=unchanged+improved, color="#bd6949", label="Changed: worsened")
    axes[0].set(xticks=x, xticklabels=[f"{h} h" for h in pilot.HORIZONS], ylabel="Test forecasts", title="Correct only when estimated chance is at least 80%", ylim=(0,120))
    axes[0].legend(fontsize=9, loc="upper center")
    selected=[r for r in rows if r["changed"]]
    gains=np.array([r["original_error_m"]-r["gated_error_m"] for r in selected])
    axes[1].bar(np.arange(1,len(gains)+1), gains, color=np.where(gains>=0,"#388465","#bd6949"), width=.85)
    axes[1].axhline(0,color="#555555",lw=1)
    axes[1].set(xlabel="Changed forecasts, in issue-date order", ylabel="Error reduction (m); positive is better", title=f"Selected cases: {s['selected_only'].get('improved',0)} improved, {s['selected_only'].get('worsened',0)} worsened")
    fig.suptitle("Variance-weighted SGP4 correction with a confidence gate\nTraining-only rule; later reference positions used for scoring", fontsize=14)
    fig.savefig(HERE / "results.png",dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
