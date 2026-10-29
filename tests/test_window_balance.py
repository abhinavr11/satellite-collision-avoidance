"""Training-only lead selection and explicit biased-error uncertainty budgets."""
import unittest
from unittest.mock import patch
import numpy as np

from collision_probability.data import SATELLITES
from collision_probability.encounter_forecast import (error_budget_covariance, ForecastTrajectories,
                                                     limit_encounter_end, METHODS)
from collision_probability.window_selection import select_balanced_window
from collision_probability.confidence import fit_confident_prefix
from test_encounter_forecast import example_states


class WindowBalanceTests(unittest.TestCase):
    def test_longest_eligible_lead_is_chosen_without_any_pc_input(self):
        states = {('sgp4', 'Swarm-A'): {'t': np.arange(-3600., 601.)}}
        seen = []

        def fake_fit(states, cutoff, end, threshold, **kwargs):
            horizon = kwargs['validation_horizon_seconds']
            seen.append((cutoff, horizon))
            passes = horizon <= 360
            entry = {'window_seconds': 300, 'validation_candidate_rmse_m': 10., 'validation_raw_rmse_m': 100.}
            gate = {'selected_window_seconds': 300, 'candidate_windows': [entry],
                    'minimum_test_horizon_model_confidence': .999 if passes else .9,
                    'apply_correction': passes}
            return {}, {}, None, None, {name: gate for name in SATELLITES}

        with patch('collision_probability.window_selection.fit_confident_prefix', fake_fit):
            selected = select_balanced_window(states, leads=(120, 180, 240))
        self.assertEqual(selected['selected_lead_seconds'], 180)
        self.assertEqual(selected['selection_reference_cutoff_seconds'], -240)
        self.assertFalse(selected['selection_uses_pc'])
        self.assertFalse(selected['selection_uses_test_truth'])
        self.assertTrue(all(cutoff == -240 for cutoff, _ in seen))

    def test_changing_reference_after_common_cutoff_cannot_change_selection(self):
        states = example_states()
        options = {'leads': (60, 120), 'threshold': .9, 'max_rmse_ratio': .9,
                   'min_folds': 2, 'max_folds': 2, 'windows': (180, 300)}
        original = select_balanced_window(states, **options)
        for name in SATELLITES:
            mask = states['reference', name]['t'] > original['selection_reference_cutoff_seconds']
            states['reference', name]['position'][mask] += 1e8
            states['reference', name]['velocity'][mask] *= -99
        changed = select_balanced_window(states, **options)
        self.assertEqual(original, changed)

    def test_second_moment_matches_gaussian_error_samples(self):
        mean = np.array([3., -4., 2.])
        covariance = np.array([[2., .3, 0], [.3, 4., -.2], [0, -.2, 1.]])
        correction = np.array([1., -2., .5])
        expected = error_budget_covariance(mean, covariance, correction)
        samples = np.random.default_rng(731).multivariate_normal(mean, covariance, 200_000)-correction
        second_moment = samples.T@samples/len(samples)
        np.testing.assert_allclose(expected, second_moment, rtol=.015, atol=.025)
        np.testing.assert_array_equal(error_budget_covariance(mean, covariance, mean), covariance)
        np.testing.assert_array_equal(error_budget_covariance(mean, covariance, np.zeros(3)), covariance+np.outer(mean, mean))

    def test_uncertainty_modes_preserve_means_and_withheld_equivalence(self):
        states, _ = limit_encounter_end(example_states(), 120)
        models, _, _, _, gates = fit_confident_prefix(states, -120, 120, windows=(180, 300), folds=2)
        residual = ForecastTrajectories(states, models, -120, gates, covariance_mode='residual')
        budget = ForecastTrajectories(states, models, -120, gates, covariance_mode='second_moment')
        for name in SATELLITES:
            for method in METHODS:
                np.testing.assert_array_equal(residual.position(method, name, 0), budget.position(method, name, 0))
            mean, p = budget.local_parameters(name, 0, METHODS[2])
            np.testing.assert_allclose(budget.local_error_budget(name, 0, METHODS[2]), p+np.outer(mean, mean))
        for gate in gates.values():
            gate['apply_correction'] = False
        withheld = ForecastTrajectories(states, models, -120, gates, covariance_mode='second_moment')
        for name in SATELLITES:
            np.testing.assert_array_equal(withheld.position(METHODS[1], name, 0), withheld.position(METHODS[2], name, 0))
            np.testing.assert_array_equal(withheld.covariance(name, 0, METHODS[1]), withheld.covariance(name, 0, METHODS[2]))


if __name__ == '__main__':
    unittest.main()
