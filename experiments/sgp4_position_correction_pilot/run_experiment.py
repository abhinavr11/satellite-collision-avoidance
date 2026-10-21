#!/usr/bin/env python3
"""Apply a training-only RTN mean-bias correction to the earlier SGP4 pilot.

This is a mean correction, not a displacement derived from variance alone.
No model variants are selected or tuned against the test outcomes.
"""
from pathlib import Path
import importlib.util
import hashlib
import json
import numpy as np
from astropy import units as u
from astropy.coordinates import CartesianRepresentation, ITRS, TEME
from astropy.time import Time
from astropy.utils import iers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "sgp4_uncertainty_pilot"
spec = importlib.util.spec_from_file_location("uncertainty_pilot", PREVIOUS / "run_experiment.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def metrics(selected):
    before = np.array([r["original_error_m"] for r in selected])
    after = np.array([r["corrected_error_m"] for r in selected])
    rms_before = float(np.sqrt(np.mean(before**2)))
    rms_after = float(np.sqrt(np.mean(after**2)))
    return {"test_forecasts": len(selected), "original_rmse_m": rms_before,
            "corrected_rmse_m": rms_after, "rmse_reduction_percent": 100*(1-rms_after/rms_before),
            "original_mean_distance_error_m": float(before.mean()),
            "corrected_mean_distance_error_m": float(after.mean()),
            "original_median_distance_error_m": float(np.median(before)),
            "corrected_median_distance_error_m": float(np.median(after)),
            "improved": int(np.sum(after < before)), "worsened": int(np.sum(after > before))}


def main():
    source = PREVIOUS / "prediction_errors.json"
    rows = json.loads(source.read_text())
    old_summary = json.loads((PREVIOUS / "summary.json").read_text())
    params = {}
    for horizon in pilot.HORIZONS:
        training = np.array([r["error_rtn_m"] for r in rows if r["split"] == "train" and int(r["horizon_hours"]) == horizon])
        bias = training.mean(axis=0)
        centered_variance = np.mean((training-bias)**2, axis=0)
        raw_second_moment = np.mean(training**2, axis=0)
        np.testing.assert_allclose(raw_second_moment, centered_variance+bias**2, atol=1e-8)
        np.testing.assert_allclose(bias, old_summary["model_parameters"]["calibrated"]["by_horizon_hours"][str(horizon)]["training_mean_error_rtn_m"], atol=1e-10)
        params[str(horizon)] = {"training_forecasts": len(training), "mean_error_rtn_m": bias.tolist(),
                               "sigma_after_centering_rtn_m": np.sqrt(centered_variance).tolist()}
    training_targets = [pilot.parse_utc(r["comparison_time_utc"]) for r in rows if r["split"] == "train"]
    test_rows = sorted([r for r in rows if r["split"] == "test"], key=lambda r: (r["tle_epoch_utc"], r["satellite"], int(r["horizon_hours"])))
    assert max(training_targets) < min(pilot.parse_utc(r["tle_epoch_utc"]) for r in test_rows)
    output = []
    for row in test_rows:
        horizon = str(row["horizon_hours"])
        bias = np.array(params[horizon]["mean_error_rtn_m"])
        error = np.array(row["error_rtn_m"])
        time = Time(row["comparison_time_utc"], scale="utc")
        satellite = pilot.read_tles(row["tle_file"])[int(row["tle_pair_index_zero_based"])]
        code, position, velocity = satellite.sgp4(time.jd1, time.jd2)
        assert code == 0
        position, velocity = np.array(position)*1000, np.array(velocity)*1000
        radial = position/np.linalg.norm(position)
        normal = np.cross(position, velocity)
        normal /= np.linalg.norm(normal)
        along = np.cross(normal, radial)
        basis = np.stack((radial, along, normal))
        displacement = basis.T @ bias
        corrected_position = position + displacement
        # Earlier pilot saved independently calculated reference-minus-SGP4 errors.
        reference_position = position + basis.T @ error
        before = float(np.linalg.norm(reference_position-position))
        after = float(np.linalg.norm(reference_position-corrected_position))
        np.testing.assert_allclose(before, row["error_3d_m"], atol=1e-7)
        np.testing.assert_allclose(after, np.linalg.norm(error-bias), atol=1e-7)
        output.append({"satellite": row["satellite"], "tle_epoch_utc": row["tle_epoch_utc"],
                       "comparison_time_utc": row["comparison_time_utc"], "horizon_hours": int(horizon),
                       "reference_file": row["reference_file"],
                       "sgp4_position_teme_m": position.tolist(), "position_correction_teme_m": displacement.tolist(),
                       "corrected_position_teme_m": corrected_position.tolist(),
                       "reference_position_teme_m": reference_position.tolist(),
                       "original_error_m": before, "corrected_error_m": after,
                       "error_reduction_percent": 100*(before-after)/before,
                       "improved": after < before})
    # First six chronological 24-hour test forecasts, with no selection by outcome.
    examples = [r for r in output if r["horizon_hours"] == 24][:6]
    table = iers.IERS_A.open(str(pilot.DATA / "earth_orientation/finals2000A.all"))
    max_reference_difference = 0.0
    with iers.earth_orientation_table.set(table):
        for row in examples:
            time = Time(row["comparison_time_utc"], scale="utc")
            r_itrs, _, _, _ = pilot.reference_at(row["reference_file"], float(time.unix), pilot.REFERENCE_MAP[row["satellite"]])
            fresh_reference = ITRS(CartesianRepresentation(r_itrs*u.m), obstime=time).transform_to(TEME(obstime=time)).cartesian.xyz.to_value(u.m)
            discrepancy = float(np.linalg.norm(fresh_reference-np.array(row["reference_position_teme_m"])))
            max_reference_difference = max(max_reference_difference, discrepancy)
            assert discrepancy < 1e-5
            np.testing.assert_allclose(np.linalg.norm(fresh_reference-np.array(row["corrected_position_teme_m"])), row["corrected_error_m"], atol=1e-5)
    grouped = {str(h): metrics([r for r in output if r["horizon_hours"] == h]) for h in pilot.HORIZONS}
    summary = {"method": "Add training mean reference-minus-SGP4 RTN error, pooled across three satellites and fitted separately at each horizon.",
               "formula": "b(h) = mean_train(e_RTN(h)); r_corrected_TEME = r_SGP4_TEME + RTN_basis.T @ b(h)",
               "training_issue_dates": "2025-01-01 through 2025-01-17",
               "test_issue_dates": "2025-01-21 through 2025-01-31",
               "parameters": params, "test_results_by_horizon_hours": grouped,
               "all_horizons": metrics(output), "first_six_24hour_examples": examples,
               "checks": {"all_corrections_fitted_without_test_data": True,
                          "latest_training_reference_before_first_test_issue": True,
                          "max_example_reference_recheck_m": max_reference_difference,
                          "example_selection": "First six chronological 24-hour test forecasts, irrespective of improvement"},
               "source_prediction_errors_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "limitations": ["Variance alone cannot determine a position correction; this fits the missing mean bias.",
                    "Same one-month data and chronological test period as the uncertainty pilot; this is an exploratory follow-up, not a new independent validation dataset.",
                    "Unknown maneuvers are not labeled or removed.", "Samples are correlated and the mean bias can vary over time.",
                    "Corrected positions at separate horizons are not guaranteed to form a dynamically consistent orbit."]}
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (HERE / "corrected_positions.json").write_text(json.dumps(output, indent=2) + "\n")
    report(summary)
    plot(summary)
    print(json.dumps({"by_horizon": grouped, "all_horizons": summary["all_horizons"], "checks": summary["checks"]}, indent=2))


def report(summary):
    lines = ["# A small position-correction experiment", "",
             "**Result: a simple training-only bias correction produces small average position-error improvements. It worsens many individual predictions.**", "",
             "Variance describes error size, not its direction. We therefore fitted the mean RTN error from the same January 1-17 training residuals used for the uncertainty experiment. This adds an estimate of systematic error to the earlier uncertainty model. No test residual was used to fit or tune the correction.", "",
             "For each horizon h, the correction is b(h) = average training reference-minus-SGP4 error in RTN coordinates. We apply r_corrected = r_SGP4 + RTN_basis_transpose times b(h). The RTN basis comes from the SGP4 predicted position and velocity, so it needs no future reference measurement. Reference positions are used only to score the test predictions.", "",
             "The original uncertainty pilot estimated mean squared errors about the unchanged SGP4 position. This follow-up separates that second moment into bias squared plus variance. The centered uncertainty scales are saved, but these results measure position error, not the calibration of the new uncertainty regions.", "",
             "Training issue dates: January 1-17, 2025. Gap: January 18-20. Test issue dates: January 21-31. There are 86 test TLE issue records, each evaluated at four horizons, giving 344 correlated test forecasts. This reuses the earlier pilot's test period and is exploratory.", "",
             "| Horizon | Original 3-D RMSE | Corrected 3-D RMSE | RMSE reduction | Individual cases improved |",
             "|---|---:|---:|---:|---:|"]
    for h, m in summary["test_results_by_horizon_hours"].items():
        lines.append(f"| {h} hours | {m['original_rmse_m']:.1f} m | {m['corrected_rmse_m']:.1f} m | {m['rmse_reduction_percent']:.2f}% | {m['improved']}/{m['test_forecasts']} |")
    lines += ["", "RMSE gives greater weight to large errors. Improvement in RMSE does not imply improvement in every other metric. At 6 hours, mean distance error increases even though RMSE improves slightly. At 12 hours, median distance error also increases. The JSON summary includes all these metrics.", "",
              "The 24-hour correction learned from training is approximately: radial -75.2 m, along-track +41.0 m, cross-track +30.5 m. It is the same RTN offset for every 24-hour case in this simple model. It is not adjusted using the true position of the case being corrected.", "",
              "The following examples are the **first six 24-hour test forecasts in chronological issue order**. They include failures; none were selected because the correction helped.", "",
              "| Satellite | Prediction issued (UTC) | Reference comparison (UTC) | Original error | Corrected error | Result |",
              "|---|---|---|---:|---:|---|"]
    for row in summary["first_six_24hour_examples"]:
        result = "Improved" if row["improved"] else "Worsened"
        lines.append(f"| {row['satellite']} | {row['tle_epoch_utc']} | {row['comparison_time_utc']} | {row['original_error_m']:.1f} m | {row['corrected_error_m']:.1f} m | {result} |")
    lines += ["", "Five of these first six forecasts worsen. Across all 86 test forecasts at 24 hours, 44 improve and 42 worsen. A small overall improvement is not a reliable correction for every individual encounter.", "",
              "Actual corrected Cartesian positions are saved in `corrected_positions.json`, in TEME coordinates and metres. They were formed by rotating each training-derived RTN correction into TEME and adding it to the SGP4 position. All residual norms were checked against the RTN calculation. The six example reference positions were also independently reloaded from the SP3 files and transformed again; agreement was better than 0.00001 m.", "",
              "Limits: only one month and three related satellites; no verified maneuver exclusions; time-varying bias; correlated samples; no demonstrated generalization to other satellites. Separate position corrections at different horizons do not constitute a validated new orbital propagator. More data and trajectory-consistent fitting would be needed before using corrected positions for collision avoidance.", "",
              "Run from the Project directory using the existing pilot environment:", "", "```bash",
              ".venv-sgp4-pilot/bin/python experiments/sgp4_position_correction_pilot/run_experiment.py", "```", "",
              "Dependencies and the original reference-processing code are in the adjacent `sgp4_uncertainty_pilot` directory. The input residual file is hashed in this experiment's summary. Neither the raw data nor the earlier uncertainty experiment is modified.", ""]
    (HERE / "report.md").write_text("\n".join(lines))


def plot(summary):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    x = np.arange(4)
    group = summary["test_results_by_horizon_hours"]
    for delta, field, label, color in [(-.18, "original_rmse_m", "Original SGP4", "#9b6b46"), (.18, "corrected_rmse_m", "SGP4 + mean correction", "#287caf")]:
        axes[0].bar(x+delta, [group[str(h)][field] for h in pilot.HORIZONS], width=.34, label=label, color=color)
    axes[0].set(xticks=x, xticklabels=[f"{h} h" for h in pilot.HORIZONS], ylabel="3-D position RMSE (m)", title="Average improvement is modest")
    axes[0].legend(fontsize=9)
    examples = summary["first_six_24hour_examples"]
    n = np.arange(1, len(examples)+1)
    axes[1].plot(n, [r["original_error_m"] for r in examples], "o-", label="Original SGP4", color="#9b6b46")
    axes[1].plot(n, [r["corrected_error_m"] for r in examples], "o-", label="After correction", color="#287caf")
    axes[1].set(xticks=n, xlabel="First six 24-hour test cases, in date order", ylabel="3-D position error (m)", title="Some corrections make the forecast worse")
    axes[1].legend(fontsize=9)
    fig.suptitle("SGP4 position correction using earlier mean errors\nNo test positions were used to fit the correction", fontsize=14)
    fig.savefig(HERE / "results.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
