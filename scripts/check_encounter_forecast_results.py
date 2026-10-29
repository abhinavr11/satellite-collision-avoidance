"""Validate the within-encounter split and three frozen-origin forecasts."""
from pathlib import Path
import argparse
import csv
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from collision_probability.encounter_forecast import METHODS, evaluate_encounter_model, shift_quadratic, error_budget_covariance
from collision_probability.confidence import candidate_correction
from collision_probability.evaluation import binary_scores, collision_label
from collision_probability.probability import collision_probability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'experiments/encounter_train_test')
    args = parser.parse_args()
    summary = json.loads((args.output_dir/'summary.json').read_text())
    assert set(summary['encounter_results']) == set(METHODS)
    assert summary['test_reference_used_for_fitting'] is False
    assert summary['historical_calibration_pairs_used'] is False
    assert summary['forecast_projection_is_extrapolation'] is True
    cutoff = summary['training_cutoff_elapsed_seconds']
    selection = summary.get('window_selection')
    if selection is not None:
        assert selection['selection_uses_pc'] is False and selection['selection_uses_test_truth'] is False
        assert selection['selection_reference_cutoff_seconds'] <= cutoff
        eligible = [c for c in selection['candidates'] if c['eligible']]
        assert selection['selected_lead_seconds'] == max(c['lead_seconds'] for c in eligible)
        for c in selection['candidates']:
            if 'latest_validation_truth_elapsed_seconds' in c:
                assert c['latest_validation_truth_elapsed_seconds'] <= selection['selection_reference_cutoff_seconds']
    assert cutoff < -summary['encounter_half_window_seconds']
    split = summary['split']
    assert split['training_last_utc'] == summary['forecast_reference_utc'] < split['test_first_utc']
    truth = summary['ground_truth']
    assert truth['covariance_assigned'] is False
    assert truth['collision_label'] == collision_label(truth['closest_approach']['miss_distance_m'], summary['combined_radius_m'])
    assert truth['closest_approach']['elapsed_seconds'] > cutoff
    for name, model in summary['models'].items():
        gates = summary.get('confidence_gates')
        assert model['training_rows'] <= split['training_samples_per_satellite']
        if gates is not None:
            assert model['training_prefix_rows_available'] == split['training_samples_per_satellite']
            gate = gates[name]
            assert gate['selection_uses_test_reference'] is False
            assert gate['decision_frozen_at_elapsed_seconds'] == cutoff
            assert gate['apply_correction'] == (gate['validation_position_check_passed'] and gate['model_confidence_check_passed'])
            assert model['covariance_calibration_uses_training_backtests_only'] is True
            for candidate in gate['candidate_windows']:
                for fold in candidate['rolling_validation_folds']:
                    assert fold['last_fit_elapsed_seconds'] == fold['origin_elapsed_seconds'] < fold['validation_first_elapsed_seconds']
                    assert fold['validation_last_elapsed_seconds'] <= cutoff
            if gate['apply_correction']:
                assert gate['minimum_test_horizon_model_confidence'] >= summary['confidence_threshold']
        assert model['training_last_elapsed_seconds'] == cutoff
        assert all(b['last_elapsed_seconds'] <= cutoff for b in model['training_bins'])
        for quantity in ('mean', 'sigma'):
            np.testing.assert_allclose(model[quantity+'_coefficients_about_cutoff'],
                                       shift_quadratic(model[quantity+'_coefficients_about_training_start'], model['fit_origin_shift_hours']))
        _, covariance = evaluate_encounter_model(model, np.array([0, -cutoff, -2*cutoff]))
        assert np.all(np.linalg.eigvalsh(covariance) > 0)
    for method, result in summary['encounter_results'].items():
        tca = result['closest_approach']['elapsed_seconds']
        np.testing.assert_allclose(result['forecast_lead_seconds'], tca-cutoff)
        assert tca > cutoff and abs(tca) < summary['encounter_half_window_seconds']
        q = np.array(result['encounter_plane_basis'])
        p, c = np.array(result['relative_covariance_gcrs_m2']), np.array(result['plane_covariance_m2'])
        np.testing.assert_allclose(q.T@q, np.eye(2), atol=1e-12)
        np.testing.assert_allclose(q.T@p@q, c)
        for name, parameters in result['parameters_at_tca'].items():
            lead = 0 if method == METHODS[0] else result['forecast_lead_seconds']
            mean, covariance = evaluate_encounter_model(summary['models'][name], lead)
            np.testing.assert_allclose(parameters['mean_error_rtn_m'], mean)
            np.testing.assert_allclose(parameters['covariance_rtn_m2'], covariance)
            allowed = method == METHODS[1] and (summary.get('confidence_gates') is None or summary['confidence_gates'][name]['apply_correction'])
            assert parameters['mean_correction_applied'] == allowed
            if not allowed:
                np.testing.assert_array_equal(parameters['applied_correction_rtn_m'], np.zeros(3))
            else:
                expected = mean if summary.get('confidence_gates') is None else candidate_correction(mean, covariance)[0]
                np.testing.assert_allclose(parameters['applied_correction_rtn_m'], expected)
            budget = covariance if summary.get('covariance_mode', 'residual') == 'residual' else error_budget_covariance(mean, covariance, parameters['applied_correction_rtn_m'])
            np.testing.assert_allclose(parameters['error_budget_rtn_m2'], budget)
        regenerated = collision_probability(result['plane_mean_m'], c, summary['combined_radius_m'])
        np.testing.assert_allclose(regenerated['pc'], result['quadrature']['pc'], rtol=1e-12)
        assert summary['outcome_scores'][method] == binary_scores(result['quadrature']['pc'], truth['collision_label'], result['quadrature']['log_pc'])
    assert summary['encounter_results'][METHODS[0]]['closest_approach'] == summary['encounter_results'][METHODS[2]]['closest_approach']
    for method, result in summary.get('residual_covariance_controls', {}).items():
        assert result['closest_approach'] == summary['encounter_results'][method]['closest_approach']
        regenerated = collision_probability(result['plane_mean_m'], result['plane_covariance_m2'], summary['combined_radius_m'])
        np.testing.assert_allclose(regenerated['pc'], result['quadrature']['pc'], rtol=1e-12)
    with (args.output_dir/'time_split.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    training, test = ([r for r in rows if r['split'] == split_name] for split_name in ('train', 'test'))
    assert len(training) == split['training_samples_per_satellite']
    assert len(test) == split['test_samples_per_satellite']
    assert all(float(r['elapsed_seconds']) <= cutoff for r in training)
    assert all(float(r['elapsed_seconds']) > cutoff for r in test)
    assert any(float(r['elapsed_seconds']) == 0 for r in test)
    assert test[-1]['time_utc'] == split['test_last_utc']
    with (args.output_dir/'forecast_trajectories.csv').open(newline='') as handle:
        trajectories = list(csv.DictReader(handle))
    assert len(trajectories) == len(test)
    for row in trajectories:
        for name in ('swarm_a', 'swarm_c'):
            for axis in ('x', 'y', 'z'):
                assert row[f'{name}_{METHODS[0]}_{axis}_m'] == row[f'{name}_{METHODS[2]}_{axis}_m']
    with (args.output_dir/'confidence_audit.csv').open(newline='') as handle:
        audits = list(csv.DictReader(handle))
    assert len(audits) == 2*len(test)
    by_time = {row['time_utc']: row for row in trajectories}
    for audit in audits:
        row = by_time[audit['time_utc']]
        name = audit['satellite'].lower().replace('-', '_')
        reference = np.array([float(row[f'{name}_reference_{axis}_m']) for axis in ('x', 'y', 'z')])
        raw = np.array([float(row[f'{name}_{METHODS[2]}_{axis}_m']) for axis in ('x', 'y', 'z')])
        corrected = np.array([float(row[f'{name}_{METHODS[1]}_{axis}_m']) for axis in ('x', 'y', 'z')])
        np.testing.assert_allclose(float(audit['raw_position_error_m']), np.linalg.norm(reference-raw), atol=1e-7)
        np.testing.assert_allclose(float(audit['corrected_position_error_m']), np.linalg.norm(reference-corrected), atol=1e-7)
        np.testing.assert_allclose(float(audit['applied_correction_norm_m']), np.linalg.norm(corrected-raw), atol=1e-7)
    for name in ('Swarm-A', 'Swarm-C'):
        rows = [r for r in audits if r['satellite'] == name]
        changed = [r for r in rows if float(r['applied_correction_norm_m']) > 1e-8]
        improved = sum(float(r['corrected_position_error_m']) < float(r['raw_position_error_m'])-1e-8 for r in changed)
        worsened = sum(float(r['corrected_position_error_m']) > float(r['raw_position_error_m'])+1e-8 for r in changed)
        recorded = summary['held_out_position_diagnostics'][name]['correction_audit']
        assert recorded['changed_samples'] == len(changed)
        assert recorded['changed_and_improved'] == improved
        assert recorded['changed_and_worsened'] == worsened
        # Check counts, without asserting that corrections MUST improve test.
        if summary.get('confidence_gates') is not None and not summary['confidence_gates'][name]['apply_correction']:
            assert not changed
    print('PASS: chronological split, held-out collision, training-only coefficients, cutoff rebasing, three isolated methods, projected covariance, Pc and binary scores.')


if __name__ == '__main__':
    main()
