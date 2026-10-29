"""Analytic integrals, rotations, model constraints, and closest approach."""
import unittest
import numpy as np
from scipy.stats import ncx2
from collision_probability.model import fit_model, evaluate_model, rtn_basis, score_model
from collision_probability.data import EncounterTrajectories
from collision_probability.evaluation import collision_label, binary_scores
from collision_probability.probability import (collision_probability, encounter_plane,
                                               project_encounter, closest_approach, monte_carlo)


class ProbabilityTests(unittest.TestCase):
    def test_centered_isotropic_matches_exact_integral(self):
        sigma, radius = 50., 10.
        expected = -np.expm1(-radius**2/(2*sigma**2))
        result = collision_probability([0, 0], np.eye(2)*sigma**2, radius)
        self.assertAlmostEqual(result['pc'], expected, places=12)

    def test_offset_isotropic_matches_noncentral_chi_square(self):
        expected = ncx2.cdf(10**2/50**2, 2, 50**2/50**2)
        self.assertAlmostEqual(collision_probability([50, 0], np.eye(2)*2500, 10)['pc'], expected, places=12)

    def test_rotation_invariance_with_correlated_covariance(self):
        angle = .618
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        covariance = np.array([[400., 125.], [125., 100.]])
        mean = np.array([12., -8.])
        original = collision_probability(mean, covariance, 5)['pc']
        rotated = collision_probability(rotation@mean, rotation@covariance@rotation.T, 5)['pc']
        self.assertAlmostEqual(original, rotated, places=10)

    def test_common_frame_projection_and_independent_covariance(self):
        pa, pc = np.diag([100., 400., 900.]), np.diag([25., 50., 75.])
        q, b, c, relative = project_encounter(np.array([1., 2., 3.]), np.array([4., 5., 6.]), pa, pc)
        np.testing.assert_allclose(relative, pa+pc)
        np.testing.assert_allclose(q.T@q, np.eye(2), atol=1e-14)
        np.testing.assert_allclose(q.T@np.array([4., 5., 6.]), 0, atol=1e-14)
        np.testing.assert_allclose(c, q.T@(pa+pc)@q)
        self.assertTrue(np.all(np.linalg.eigvalsh(c) > 0))

    def test_invalid_covariance_velocity_and_radius_rejected(self):
        for c in (np.zeros((2, 2)), [[1, 2], [2, 1]], [[1, .5], [0, 1]]):
            with self.assertRaises(ValueError):
                collision_probability([0, 0], c, 1)
        with self.assertRaises(ValueError):
            encounter_plane([0, 0, 0])
        with self.assertRaises(ValueError):
            collision_probability([0, 0], np.eye(2), -1)

    def test_tca_refined_between_samples_and_boundary_marked(self):
        def relative(t):
            t = np.asarray(t)
            return np.stack((15000*(t-.123456), np.ones_like(t)*3, np.zeros_like(t)), axis=-1)
        result = closest_approach(relative, np.arange(-10, 11))
        self.assertAlmostEqual(result['elapsed_seconds'], .123456, places=7)
        self.assertAlmostEqual(result['miss_distance_m'], 3, places=6)
        self.assertFalse(result['boundary_minimum'])
        boundary = closest_approach(lambda t: np.stack((np.asarray(t)+10, np.ones_like(t), np.zeros_like(t)), axis=-1), np.arange(5))
        self.assertTrue(boundary['boundary_minimum'])

    def test_seeded_monte_carlo_consistent_with_analytic_case(self):
        mean, covariance, radius = np.array([1., 0.]), np.eye(2)*4, 1.
        exact = collision_probability(mean, covariance, radius)['pc']
        result = monte_carlo(mean, covariance, radius, samples=100_000, seed=41)
        self.assertLess(abs(result['pc']-exact), 6*np.sqrt(exact*(1-exact)/100_000))

    def test_underflow_retains_log_probability(self):
        result = collision_probability([100., 0.], np.eye(2), 1)
        self.assertEqual(result['pc'], 0.)
        self.assertTrue(np.isfinite(result['log_pc']))


class OutcomeTests(unittest.TestCase):
    def test_reference_collision_noncollision_and_contact(self):
        self.assertEqual(collision_label(.286, 10), 1)
        self.assertEqual(collision_label(11, 10), 0)
        self.assertEqual(collision_label(10, 10), 1)

    def test_higher_probability_scores_better_for_collision(self):
        lower, higher = binary_scores(.1, 1), binary_scores(.9, 1)
        self.assertLess(higher['brier_score'], lower['brier_score'])
        self.assertLess(higher['log_loss'], lower['log_loss'])
        self.assertAlmostEqual(higher['brier_score'], .01)
        self.assertAlmostEqual(higher['log_loss'], -np.log(.9))

    def test_lower_probability_scores_better_for_noncollision(self):
        lower, higher = binary_scores(.1, 0), binary_scores(.9, 0)
        self.assertLess(lower['brier_score'], higher['brier_score'])
        self.assertLess(lower['log_loss'], higher['log_loss'])

    def test_endpoint_and_underflow_losses_are_explicit(self):
        self.assertEqual(binary_scores(1., 1)['log_loss'], 0.)
        self.assertEqual(binary_scores(0., 0)['log_loss'], 0.)
        self.assertTrue(binary_scores(0., 1)['log_loss_is_infinite'])
        self.assertIsNone(binary_scores(1., 0)['log_loss'])
        self.assertEqual(binary_scores(0., 1, -1000)['log_loss'], 1000.)

    def test_invalid_labels_probabilities_and_geometry_rejected(self):
        for p, label in [(-.1, 1), (1.1, 0), (.5, 2)]:
            with self.assertRaises(ValueError):
                binary_scores(p, label)
        with self.assertRaises(ValueError):
            collision_label(-1, 10)
        with self.assertRaises(ValueError):
            binary_scores(.1, 1, np.log(.2))


class ModelTests(unittest.TestCase):
    def test_scoring_rotates_inference_basis_to_reference_basis(self):
        model = {'supported_horizon_days': [.25, 2],
                 'mean_coefficients_m': [[1, 0, 0], [2, 0, 0], [3, 0, 0]],
                 'sigma_coefficients_m': [[2, 0, 0], [3, 0, 0], [4, 0, 0]],
                 'correlation': np.eye(3).tolist()}
        rotation = np.array([[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]])
        error = np.array([[-2., 1., 3.]])
        scores = score_model(model, np.array([1.]), error, rotation)
        self.assertEqual(scores['corrected']['position_rmse_m'], 0.)
        self.assertAlmostEqual(scores['uncorrected']['position_rmse_m'], np.sqrt(14))

    def test_predictions_do_not_consult_encounter_reference_positions(self):
        times = np.arange(-5., 6.)
        model = {'supported_horizon_days': [.25, 2],
                 'mean_coefficients_m': [[1, 0, 0], [2, 0, 0], [3, 0, 0]],
                 'sigma_coefficients_m': [[2, 0, 0], [3, 0, 0], [4, 0, 0]],
                 'correlation': np.eye(3).tolist()}
        states = {}
        for name in ('Swarm-A', 'Swarm-C'):
            for kind in ('reference', 'sgp4'):
                states[kind, name] = {'t': times, 'age_days': 1+times/86400,
                                      'position': np.column_stack((np.ones(11)*7e6, 7500*times, np.zeros(11))),
                                      'velocity': np.tile([0., 7500., 0.], (11, 1))}
        original = EncounterTrajectories(states, {n: model for n in ('Swarm-A', 'Swarm-C')})
        for name in ('Swarm-A', 'Swarm-C'):
            states['reference', name]['position'] = states['reference', name]['position']+100_000
        changed = EncounterTrajectories(states, {n: model for n in ('Swarm-A', 'Swarm-C')})
        np.testing.assert_array_equal(original.position('corrected', 'Swarm-A', 0), changed.position('corrected', 'Swarm-A', 0))
        np.testing.assert_array_equal(original.covariance('Swarm-A', 0), changed.covariance('Swarm-A', 0))
        finite_difference = (original.position('corrected', 'Swarm-A', .01)-original.position('corrected', 'Swarm-A', -.01))/.02
        np.testing.assert_allclose(original.velocity('corrected', 'Swarm-A', 0), finite_difference, atol=1e-5)

    def test_reference_basis_orientation_and_degeneracy(self):
        basis = rtn_basis([7e6, 0, 0], [0, 7500, 0])
        np.testing.assert_allclose(basis, np.eye(3))
        with self.assertRaises(ValueError):
            rtn_basis([1, 0, 0], [1, 0, 0])

    def test_fit_recovers_spread_and_records_sample_covariance(self):
        rng = np.random.default_rng(24)
        days = np.repeat([.25, .5, 1., 2.], 400)
        x = np.column_stack((np.ones_like(days), days, days**2))
        c = np.array([[10., 20., -3.], [-30., 7., 2.], [5., -2., 1.]])
        s = np.array([[20., 10., 5.], [50., 40., 10.], [10., 2., 1.]])
        gamma = np.array([[1., .3, -.1], [.3, 1., .2], [-.1, .2, 1.]])
        errors = x@c.T+(rng.multivariate_normal(np.zeros(3), gamma, len(days))*(x@s.T))
        model = fit_model(days, errors)
        mean, covariance = evaluate_model(model, np.array([.25, .5, 1, 2]))
        self.assertTrue(np.all(np.asarray(model['sigma_coefficients_m'])[:, 0] > 0))
        self.assertTrue(np.all(np.asarray(model['sigma_coefficients_m'])[:, 1:] >= 0))
        fitted_sigma = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))
        np.testing.assert_allclose(fitted_sigma, np.column_stack((np.ones(4), [.25, .5, 1, 2], np.square([.25, .5, 1, 2])))@s.T, rtol=.15)
        np.testing.assert_allclose(model['sample_bins_by_hours']['6.0']['sample_covariance_rtn_m2'], np.cov(errors[days == .25], rowvar=False, ddof=1))
        with self.assertRaises(ValueError):
            evaluate_model(model, .1)


if __name__ == '__main__':
    unittest.main()
