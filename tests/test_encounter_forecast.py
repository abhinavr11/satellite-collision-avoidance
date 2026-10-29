"""Time split, forecast-origin translation, method isolation and truth leakage."""
import unittest
import numpy as np

from collision_probability.data import SATELLITES
from collision_probability.encounter_forecast import (METHODS, split_times, shift_quadratic,
                                                     fit_training_prefix, ForecastTrajectories,
                                                     evaluate_encounter_model)
from collision_probability.model import design, rtn_basis


def example_states():
    times = np.arange(-1800., 601.)
    rate = 2*np.pi/5400
    r = 7e6*np.column_stack((np.cos(rate*times), np.sin(rate*times), np.zeros(len(times))))
    v = 7e6*rate*np.column_stack((-np.sin(rate*times), np.cos(rate*times), np.zeros(len(times))))
    basis = rtn_basis(r, v)
    rng = np.random.default_rng(99)
    x = (times+1800)/3600
    noise = rng.normal(size=(len(times), 3))
    errors = np.column_stack((10+3*x+2*x*x, 20-2*x+4*x*x, -5+x+x*x))+noise*[1., 2., .5]
    states = {}
    for name in SATELLITES:
        states['sgp4', name] = {'t': times, 'position': r.copy(), 'velocity': v.copy()}
        states['reference', name] = {'t': times, 'position': r+np.einsum('nij,nj->ni', basis, errors),
                                    'velocity': v.copy()}
    return states


class ForecastTests(unittest.TestCase):
    def test_split_is_chronological_and_contains_collision_in_test(self):
        times = np.arange(-3600., 3601.)
        train, test = split_times(times, -600)
        self.assertEqual(train.sum(), 3001)
        self.assertEqual(test.sum(), 4200)
        self.assertFalse(np.any(train & test))
        self.assertTrue(np.all(train | test))
        self.assertTrue(np.all(times[train] <= -600))
        self.assertTrue(test[np.flatnonzero(times == 0)[0]])
        self.assertTrue(np.any((times[test] < 0)))
        for cutoff in (0, 1, -600.5, -4000):
            with self.assertRaises(ValueError):
                split_times(times, cutoff)

    def test_quadratic_translation_preserves_values_and_derivatives(self):
        coefficients = np.array([[1., 2., 3.], [-5., 7., -2.]])
        origin = .833333333
        translated = shift_quadratic(coefficients, origin)
        h = np.linspace(-origin, 1., 10)
        np.testing.assert_allclose(design(h)@translated.T, design(h+origin)@coefficients.T, atol=1e-13)
        np.testing.assert_allclose(translated[:, 1], coefficients[:, 1]+2*coefficients[:, 2]*origin)

    def test_test_reference_cannot_change_fits_or_predictions(self):
        states = example_states()
        original, _, train, test = fit_training_prefix(states, -600)
        curves = ForecastTrajectories(states, original, -600)
        for name in SATELLITES:
            states['reference', name]['position'][test] += 1e6
            states['reference', name]['velocity'][test] *= -3
        changed, _, _, _ = fit_training_prefix(states, -600)
        changed_curves = ForecastTrajectories(states, changed, -600)
        for name in SATELLITES:
            self.assertEqual(original[name], changed[name])
            for method in METHODS:
                np.testing.assert_array_equal(curves.position(method, name, 0), changed_curves.position(method, name, 0))
                np.testing.assert_array_equal(curves.covariance(name, 0, method), changed_curves.covariance(name, 0, method))

    def test_three_methods_isolate_mean_correction_and_covariance_growth(self):
        states = example_states()
        model = {'fit_origin_shift_hours': 1.,
                 'mean_coefficients_about_training_start': [[2, 3, 1], [5, -1, 2], [-1, 2, 3]],
                 'sigma_coefficients_about_training_start': [[2, 1, 1], [3, 2, 1], [1, 1, 1]],
                 'correlation': [[1, .2, 0], [.2, 1, -.1], [0, -.1, 1]]}
        curves = ForecastTrajectories(states, {name: model for name in SATELLITES}, -600)
        for name in SATELLITES:
            np.testing.assert_array_equal(curves.position(METHODS[0], name, 0), curves.position(METHODS[2], name, 0))
            self.assertGreater(np.linalg.norm(curves.position(METHODS[1], name, 0)-curves.position(METHODS[2], name, 0)), 0)
            np.testing.assert_array_equal(curves.covariance(name, 0, METHODS[1]), curves.covariance(name, 0, METHODS[2]))
            at_cutoff = curves.local_parameters(name, -600, METHODS[0])
            frozen_future = curves.local_parameters(name, 0, METHODS[0])
            np.testing.assert_array_equal(at_cutoff[0], frozen_future[0])
            np.testing.assert_array_equal(at_cutoff[1], frozen_future[1])
            self.assertGreater(np.trace(curves.local_parameters(name, 0, METHODS[2])[1]), np.trace(frozen_future[1]))
            projected_at_cutoff = curves.local_parameters(name, -600, METHODS[2])
            np.testing.assert_array_equal(at_cutoff[1], projected_at_cutoff[1])

    def test_fit_coefficients_are_rebased_and_spread_is_positive(self):
        states = example_states()
        models, _, _, _ = fit_training_prefix(states, -600)
        for model in models.values():
            np.testing.assert_allclose(model['mean_coefficients_about_cutoff'],
                                       shift_quadratic(model['mean_coefficients_about_training_start'], model['fit_origin_shift_hours']))
            sigma = np.array(model['sigma_coefficients_about_cutoff'])
            self.assertTrue(np.all(sigma[:, 0] > 0))
            self.assertTrue(np.all(sigma[:, 1:] >= 0))
            _, covariance = evaluate_encounter_model(model, np.array([0, 600, 1200]))
            self.assertTrue(np.all(np.linalg.eigvalsh(covariance) > 0))


if __name__ == '__main__':
    unittest.main()
