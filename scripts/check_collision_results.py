"""Check saved collision outputs without assuming correction must improve Pc."""
from pathlib import Path
import argparse
import csv
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from collision_probability.model import evaluate_model
from collision_probability.evaluation import collision_label, binary_scores
from collision_probability.probability import collision_probability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'experiments/collision_probability')
    args = parser.parse_args()
    output = args.output_dir
    summary = json.loads((output/'summary.json').read_text())
    assert set(summary['encounter_results']) == {'uncorrected', 'corrected'}
    assert 'common_plane_comparison' not in summary
    assert 'reference_geometry' not in summary['encounter_results']
    truth = summary['ground_truth']
    assert truth['covariance_assigned'] is False and truth['reference_treated_as_exact'] is True
    assert not any('covariance' in k or 'pc' in k.lower() for k in truth if k != 'covariance_assigned')
    assert truth['collision_label'] == collision_label(truth['closest_approach']['miss_distance_m'], summary['combined_radius_m'])
    for name, result in summary['encounter_results'].items():
        expected = binary_scores(result['quadrature']['pc'], truth['collision_label'], result['quadrature']['log_pc'])
        assert summary['outcome_scores'][name] == expected
    assert summary['correction_covariance_shared_with_uncorrected'] is True
    checks = summary['calibration_checks']
    assert checks['latest_training_target_utc'] < checks['earliest_test_issue_utc']
    assert checks['reference_rtn_basis']
    for model in summary['models'].values():
        coefficients = np.array(model['sigma_coefficients_m'])
        assert np.all(coefficients[:, 0] > 0) and np.all(coefficients[:, 1:] >= 0)
        _, covariance = evaluate_model(model, np.linspace(.25, 2, 100))
        assert np.all(np.linalg.eigvalsh(covariance) > 0)
    for result in summary['encounter_results'].values():
        q = np.array(result['encounter_plane_basis'])
        c = np.array(result['plane_covariance_m2'])
        p = np.array(result['relative_covariance_gcrs_m2'])
        b = np.array(result['plane_mean_m'])
        v = np.array(result['relative_velocity_gcrs_m_s'])
        np.testing.assert_allclose(q.T@q, np.eye(2), atol=1e-12)
        np.testing.assert_allclose(q.T@v, 0, atol=1e-10)
        np.testing.assert_allclose(c, q.T@p@q, rtol=1e-12, atol=1e-8)
        np.testing.assert_allclose(p, sum(np.array(m) for m in result['object_covariances_gcrs_m2'].values()))
        assert abs(result['closest_approach']['elapsed_seconds']) < summary['encounter_half_window_seconds']
        assert not result['closest_approach']['boundary_minimum']
        assert abs(result['along_velocity_mean_m']) < .01
        assert 0 <= result['quadrature']['pc'] <= 1
        assert result['quadrature']['converged']
        regenerated = collision_probability(b, c, summary['combined_radius_m'])
        np.testing.assert_allclose(regenerated['pc'], result['quadrature']['pc'], rtol=1e-12)
    with (output/'encounter_trajectories.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 7201
    assert len({r['time_utc'] for r in rows}) == len(rows)
    for name in ('Swarm-A', 'Swarm-C'):
        prefix = name.lower().replace('-', '_')
        before = np.array([[float(r[prefix+'_uncorrected_'+a+'_m']) for a in ('x', 'y', 'z')] for r in rows])
        after = np.array([[float(r[prefix+'_corrected_'+a+'_m']) for a in ('x', 'y', 'z')] for r in rows])
        assert np.all(np.isfinite(before)) and np.all(np.isfinite(after))
        assert np.max(np.linalg.norm(after-before, axis=1)) > 0
    print('PASS: binary reference outcome without reference Pc/covariance, proper forecast scores, model constraints, chronology, projections, same-event TCA and saved trajectories.')


if __name__ == '__main__':
    main()
