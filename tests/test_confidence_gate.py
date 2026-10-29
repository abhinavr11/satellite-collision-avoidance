"""Model-confidence mathematics, short holdout, and training-only gate choices."""
import unittest
import numpy as np

from collision_probability.confidence import candidate_correction, fit_confident_prefix
from collision_probability.data import SATELLITES
from collision_probability.encounter_forecast import (METHODS, ForecastTrajectories,
                                                     limit_encounter_end, split_times)
from test_encounter_forecast import example_states


class ConfidenceTests(unittest.TestCase):
    def test_analytic_improvement_probability_matches_full_covariance_monte_carlo(self):
        mean = np.array([5., -3., 2.])
        covariance = np.array([[9., 2., 1.], [2., 16., -1.], [1., -1., 4.]])
        correction, confidence = candidate_correction(mean, covariance)
        samples = np.random.default_rng(41).multivariate_normal(mean, covariance, 100_000)
        improved = np.sum((samples-correction)**2, axis=1) < np.sum(samples**2, axis=1)
        self.assertLess(abs(np.mean(improved)-confidence), .01)
        self.assertTrue(np.all(np.abs(correction) <= np.abs(mean)))

    def test_uncertain_or_zero_bias_does_not_meet_99_percent_gate(self):
        _, confidence = candidate_correction(np.zeros(3), np.eye(3))
        self.assertEqual(confidence, .5)
        _, confidence = candidate_correction(np.array([.1, .1, .1]), np.eye(3)*100)
        self.assertLess(confidence, .99)

    def test_short_test_discards_later_data_without_mutating_source(self):
        states = example_states()
        cropped, omitted = limit_encounter_end(states, 120)
        self.assertEqual(omitted, 480)
        self.assertEqual(states['sgp4', 'Swarm-A']['t'][-1], 600)
        self.assertEqual(cropped['sgp4', 'Swarm-A']['t'][-1], 120)
        train, test = split_times(cropped['sgp4', 'Swarm-A']['t'], -120)
        self.assertEqual(test.sum(), 240)
        self.assertFalse(np.any(train & test))
        self.assertTrue(np.any(cropped['sgp4', 'Swarm-A']['t'][test] == 0))

    def test_withheld_correction_exactly_preserves_raw_sgp4(self):
        states, _ = limit_encounter_end(example_states(), 120)
        models, _, _, _, gates = fit_confident_prefix(states, -120, 120, windows=(180, 300), folds=2)
        for gate in gates.values():
            gate['apply_correction'] = False
        curves = ForecastTrajectories(states, models, -120, gates)
        for name in SATELLITES:
            np.testing.assert_array_equal(curves.position(METHODS[1], name, curves.times),
                                          curves.position(METHODS[2], name, curves.times))
            np.testing.assert_array_equal(curves.covariance(name, 0, METHODS[1]), curves.covariance(name, 0, METHODS[2]))

    def test_changing_test_truth_cannot_select_window_or_gate(self):
        states, _ = limit_encounter_end(example_states(), 120)
        models, _, train, test, gates = fit_confident_prefix(states, -120, 120, windows=(180, 300), folds=2)
        original = ForecastTrajectories(states, models, -120, gates)
        for name in SATELLITES:
            states['reference', name]['position'][test] += 1e7
            states['reference', name]['velocity'][test] *= -9
        changed, _, _, _, changed_gates = fit_confident_prefix(states, -120, 120, windows=(180, 300), folds=2)
        self.assertEqual(models, changed)
        self.assertEqual(gates, changed_gates)
        curves = ForecastTrajectories(states, changed, -120, changed_gates)
        for name in SATELLITES:
            for method in METHODS:
                np.testing.assert_array_equal(original.position(method, name, 0), curves.position(method, name, 0))
                np.testing.assert_array_equal(original.covariance(name, 0, method), curves.covariance(name, 0, method))
            for candidate in gates[name]['candidate_windows']:
                for fold in candidate['rolling_validation_folds']:
                    self.assertLess(fold['last_fit_elapsed_seconds'], fold['validation_first_elapsed_seconds'])
                    self.assertLessEqual(fold['validation_last_elapsed_seconds'], -120)


if __name__ == '__main__':
    unittest.main()
