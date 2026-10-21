#!/usr/bin/env python3
"""Check regression metrics, saved positions, and no-correction invariants."""
from pathlib import Path
import hashlib
import json
import math

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/"experiments"


def close(actual, expected, tolerance=.1):
    assert math.isfinite(actual) and abs(actual-expected)<=tolerance, (actual,expected)


def distance(a,b):
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))


def main():
    uncertainty=json.loads((EXP/"sgp4_uncertainty_pilot/summary.json").read_text())
    assert uncertainty["splits"]=={"train":540,"gap":96,"test":344}
    close(uncertainty["position_errors_unchanged"]["24"]["rms_position_error_m"],642.7117715651549)
    close(100*uncertainty["test_scores"]["calibrated"]["all"]["mean_absolute_coverage_error"],6.037209302325581)
    assert uncertainty["sanity_checks"]["prediction_vectors_unchanged_by_calibration"]
    residual=EXP/"sgp4_uncertainty_pilot/prediction_errors.json"
    source_hash=hashlib.sha256(residual.read_bytes()).hexdigest()
    mean=json.loads((EXP/"sgp4_position_correction_pilot/summary.json").read_text())
    gate=json.loads((EXP/"sgp4_confidence_gated_correction/summary.json").read_text())
    assert mean["source_prediction_errors_sha256"]==gate["source_prediction_errors_sha256"]==source_hash
    close(mean["all_horizons"]["corrected_rmse_m"],1204.7655935475505)
    assert gate["changed_forecasts"]==86 and gate["unchanged_forecasts"]==258
    close(gate["comparisons"]["gated"]["corrected_rmse_m"],1289.849325757335)
    assert gate["checks"]["training_only_rule"]
    positions=json.loads((EXP/"sgp4_position_correction_pilot/corrected_positions.json").read_text())
    gated=json.loads((EXP/"sgp4_confidence_gated_correction/corrected_positions.json").read_text())
    assert len(positions)==len(gated)==344
    for row in positions:
        close(distance(row["corrected_position_teme_m"],row["reference_position_teme_m"]),row["corrected_error_m"],1e-5)
    for row in gated:
        assert row["changed"]==(row["estimated_improvement_probability"]>=gate["threshold"])
        if not row["changed"]:
            assert row["gated_position_teme_m"]==row["original_position_teme_m"]
        close(distance(row["gated_position_teme_m"],row["reference_position_teme_m"]),row["gated_error_m"],1e-5)
    print("PASS: numerical regression checks, input hashes, corrected positions, and preserved forecasts.")


if __name__ == "__main__":
    main()
