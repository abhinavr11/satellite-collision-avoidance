"""Fit within-encounter quadratics on a prefix and freeze one forecast origin.

This is intentionally separate from the historical multi-horizon model. A
single correlated trajectory arc supplies temporal residual scatter, not an
independent ensemble of forecast errors at each horizon.
"""
import numpy as np
from scipy.interpolate import CubicSpline

from .data import SATELLITES
from .model import _fit_direction, design, rtn_basis

METHODS = ('sgp4_frozen_covariance', 'corrected_projected', 'sgp4_projected_covariance')


def limit_encounter_end(states, end_seconds):
    times = states['sgp4', 'Swarm-A']['t']
    if not np.isfinite(end_seconds) or end_seconds not in times or not 0 < end_seconds <= times[-1]:
        raise ValueError('Test end must be a sampled time after the crossing')
    keep = times <= end_seconds
    cropped = {}
    for key, state in states.items():
        cropped[key] = {field: value[keep].copy() if isinstance(value, np.ndarray)
                        else [item for item, selected in zip(value, keep) if selected]
                        for field, value in state.items()}
    return cropped, int(np.count_nonzero(~keep))


def split_times(times, cutoff):
    times = np.asarray(times, float)
    if times.ndim != 1 or len(times) < 20 or np.any(np.diff(times) <= 0):
        raise ValueError('Need an increasing encounter time grid')
    if not np.isfinite(cutoff) or cutoff not in times or not times[0] < cutoff < 0 < times[-1]:
        raise ValueError('Training cutoff must be a sampled time before the collision center')
    return times <= cutoff, times > cutoff


def shift_quadratic(coefficients, origin_hours):
    """Express a(t) about a later origin: q(h) = a(origin_hours + h)."""
    c = np.asarray(coefficients, float)
    return np.stack((c[..., 0]+origin_hours*c[..., 1]+origin_hours**2*c[..., 2],
                     c[..., 1]+2*origin_hours*c[..., 2], c[..., 2]), axis=-1)


def fit_encounter_model(times, errors, cutoff, bin_seconds=60):
    """Use only samples at/before cutoff; export coefficients about h=0 there."""
    times, errors = np.asarray(times, float), np.asarray(errors, float)
    if len(times) < 20 or errors.shape != (len(times), 3) or np.any(np.diff(times) <= 0):
        raise ValueError('Need paired increasing training times and 3-D errors')
    if times[-1] != cutoff or np.any(times > cutoff) or not np.all(np.isfinite(errors)):
        raise ValueError('Training inputs must end exactly at the forecast origin')
    if bin_seconds <= 0 or (times[-1]-times[0])/bin_seconds < 3:
        raise ValueError('Need at least three training bins')
    hours = (times-times[0])/3600
    bin_count = int(np.ceil((times[-1]-times[0])/bin_seconds))
    group = np.minimum(((times-times[0])/bin_seconds).astype(int), bin_count-1)
    bins, centers, residual_spread = [], [], []
    initial_mean = design(hours)@np.linalg.lstsq(design(hours), errors, rcond=None)[0]
    for k in range(bin_count):
        mask = group == k
        if mask.sum() < 2:
            raise ValueError('Each time bin needs at least two training samples')
        covariance = np.cov(errors[mask], rowvar=False, ddof=1)
        centers.append(float(hours[mask].mean()))
        residual_spread.append(np.std((errors-initial_mean)[mask], axis=0, ddof=1))
        bins.append({'first_elapsed_seconds': float(times[mask][0]),
                     'last_elapsed_seconds': float(times[mask][-1]), 'rows': int(mask.sum()),
                     'mean_rtn_m': errors[mask].mean(axis=0).tolist(),
                     'sample_covariance_rtn_m2': covariance.tolist()})
    residual_spread = np.asarray(residual_spread)
    directions = [_fit_direction(hours, errors[:, j], (np.asarray(centers), residual_spread[:, j]))
                  for j in range(3)]
    mean = np.array([d['mean_coefficients_m'] for d in directions])
    spread = np.array([d['sigma_coefficients_m'] for d in directions])
    z = (errors-design(hours)@mean.T)/(design(hours)@spread.T)
    gamma = np.corrcoef(z, rowvar=False)
    if not np.all(np.isfinite(gamma)):
        raise ValueError('Temporal residual correlation is undefined')
    values, vectors = np.linalg.eigh(gamma)
    regularized = bool(values.min() < 1e-8)
    if regularized:
        gamma = (vectors*np.maximum(values, 1e-8))@vectors.T
        scale = np.sqrt(np.diag(gamma))
        gamma /= scale[:, None]*scale[None, :]
    np.linalg.cholesky(gamma)
    origin_hours = float(hours[-1])
    return {'time_unit': 'hours after training cutoff', 'training_rows': len(times),
            'training_first_elapsed_seconds': float(times[0]), 'training_last_elapsed_seconds': cutoff,
            'training_bins': bins, 'bin_seconds': bin_seconds,
            'fit_time_unit': 'hours after training start', 'fit_origin_shift_hours': origin_hours,
            'mean_coefficients_about_training_start': mean.tolist(),
            'sigma_coefficients_about_training_start': spread.tolist(),
            'mean_coefficients_about_cutoff': shift_quadratic(mean, origin_hours).tolist(),
            'sigma_coefficients_about_cutoff': shift_quadratic(spread, origin_hours).tolist(),
            'correlation': gamma.tolist(), 'correlation_regularized': regularized,
            'minimum_correlation_eigenvalue_before_regularization': float(values.min()),
            'optimization': [d['optimization'] for d in directions],
            'nearby_samples_are_correlated': True,
            'uncertainty_interpretation': 'Residual scatter around a temporal quadratic on one training arc; forecast extrapolation and independence are unvalidated.'}


def evaluate_encounter_model(model, lead_seconds):
    lead = np.asarray(lead_seconds, float)
    training_duration = model['fit_origin_shift_hours']*3600
    if np.any(~np.isfinite(lead)) or np.any(lead < -training_duration-1e-7):
        raise ValueError('Cannot evaluate before this model training arc')
    # Evaluate in the nonnegative fitting coordinate to avoid cancellation in
    # spread curves translated about the cutoff. Both coefficient sets are saved.
    x = design(lead/3600+model['fit_origin_shift_hours'])
    mean = x@np.array(model['mean_coefficients_about_training_start']).T
    sigma = x@np.array(model['sigma_coefficients_about_training_start']).T
    if np.any(sigma <= 0):
        raise ValueError('Nonpositive projected spread')
    covariance = sigma[..., :, None]*np.asarray(model['correlation'])*sigma[..., None, :]
    np.linalg.cholesky(covariance)
    return mean, covariance


def error_budget_covariance(mean, residual_covariance, applied_correction):
    """Second moment of error about the selected (possibly uncorrected) mean.

    This is NOT the centred covariance of a biased error. Using it as a
    covariance defines an explicit zero-mean Gaussian approximation that
    includes the unremoved bias in its RMS uncertainty budget.
    """
    remaining_bias = np.asarray(mean)-np.asarray(applied_correction)
    return np.asarray(residual_covariance)+remaining_bias[..., :, None]*remaining_bias[..., None, :]


def fit_training_prefix(states, cutoff, bin_seconds=60):
    """Slice first, then compute errors and fit; test truth is never inspected."""
    times = states['sgp4', 'Swarm-A']['t']
    train, test = split_times(times, cutoff)
    models, training_errors = {}, {}
    for name in SATELLITES:
        reference, predicted = states['reference', name], states['sgp4', name]
        basis = rtn_basis(reference['position'][train], reference['velocity'][train])
        errors = np.einsum('nji,nj->ni', basis, reference['position'][train]-predicted['position'][train])
        models[name] = fit_encounter_model(times[train], errors, cutoff, bin_seconds)
        training_errors[name] = errors
    return models, training_errors, train, test


class ForecastTrajectories:
    """Three predictions issued once at cutoff, with no test-reference updates."""
    def __init__(self, states, models, cutoff, gates=None, covariance_mode='residual'):
        self.states, self.models, self.cutoff = states, models, float(cutoff)
        self.gates = gates
        if covariance_mode not in ('residual', 'second_moment'):
            raise ValueError('Covariance mode must be residual or second_moment')
        self.covariance_mode = covariance_mode
        original_times = states['sgp4', 'Swarm-A']['t']
        _, test = split_times(original_times, cutoff)
        # Include h=0 to define the forecast origin; evaluation uses t>cutoff.
        forecast_mask = original_times >= cutoff
        self.times = original_times[test]
        self.forecast_times = original_times[forecast_mask]
        self.curves, self.predicted_velocities = {}, {}
        for name in SATELLITES:
            predicted = states['sgp4', name]
            r, v = predicted['position'][forecast_mask], predicted['velocity'][forecast_mask]
            mu, covariance = evaluate_encounter_model(models[name], self.forecast_times-cutoff)
            if gates is None:
                local_correction = mu
            else:
                from .confidence import candidate_correction
                local_correction, _ = candidate_correction(mu, covariance)
                if not gates[name]['apply_correction']:
                    local_correction = np.zeros_like(mu)
            correction = np.einsum('nij,nj->ni', rtn_basis(r, v), local_correction)
            for kind, positions in [('reference', states['reference', name]['position'][forecast_mask]),
                                    ('raw', r), ('corrected_projected', r+correction)]:
                self.curves[kind, name] = CubicSpline(self.forecast_times, positions, extrapolate=False)
            self.predicted_velocities[name] = CubicSpline(self.forecast_times, v, extrapolate=False)

    def position(self, kind, name, t):
        curve = kind if kind in ('reference', 'corrected_projected') else 'raw'
        return self.curves[curve, name](t)

    def velocity(self, kind, name, t):
        curve = kind if kind in ('reference', 'corrected_projected') else 'raw'
        return self.curves[curve, name](t, 1)

    def relative(self, kind, t):
        return self.position(kind, 'Swarm-A', t)-self.position(kind, 'Swarm-C', t)

    def relative_velocity(self, kind, t):
        return self.velocity(kind, 'Swarm-A', t)-self.velocity(kind, 'Swarm-C', t)

    def local_parameters(self, name, t, kind):
        if kind not in METHODS:
            raise ValueError('Unknown forecast technique')
        lead = np.asarray(t, float)-self.cutoff
        if np.any(lead < 0):
            raise ValueError('Forecast must be at/after its training cutoff')
        return evaluate_encounter_model(self.models[name], np.zeros_like(lead) if kind == METHODS[0] else lead)

    def local_error_budget(self, name, t, kind):
        mean, local = self.local_parameters(name, t, kind)
        if self.covariance_mode == 'residual':
            return local
        applied = np.zeros_like(mean)
        if kind == METHODS[1]:
            if self.gates is None:
                applied = mean
            elif self.gates[name]['apply_correction']:
                from .confidence import candidate_correction
                applied = candidate_correction(mean, local)[0]
        return error_budget_covariance(mean, local, applied)

    def covariance(self, name, t, kind):
        local = self.local_error_budget(name, t, kind)
        basis = rtn_basis(self.position(kind if kind != 'corrected_projected' else METHODS[2], name, t),
                          self.predicted_velocities[name](t))
        # Frozen RTN statistics still rotate into the predicted future GCRS
        # frame; rotation is a coordinate operation, not uncertainty growth.
        return basis@local@np.swapaxes(basis, -1, -2)
