"""Sections 3–4 of the simple guide; horizons are in days, errors in metres."""
import numpy as np
from scipy.optimize import minimize, lsq_linear
from scipy.stats import chi2


def rtn_basis(position, velocity):
    """Return matrices whose columns are R, T, N; supports batches."""
    r, v = np.asarray(position, dtype=float), np.asarray(velocity, dtype=float)
    radial_norm = np.linalg.norm(r, axis=-1, keepdims=True)
    normal = np.cross(r, v)
    normal_norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    if np.any(radial_norm <= 0) or np.any(normal_norm <= 1e-12):
        raise ValueError('Cannot construct RTN from a degenerate state')
    radial, normal = r/radial_norm, normal/normal_norm
    return np.stack((radial, np.cross(normal, radial), normal), axis=-1)


def design(days):
    days = np.asarray(days, dtype=float)
    return np.stack((np.ones_like(days), days, days**2), axis=-1)


def _fit_direction(days, errors, initial_spread=None):
    """Profile out mean coefficients; optimize the guide's exact marginal NLL.

    For a fixed spread curve, weighted least squares gives the unconstrained
    mean quadratic. Scaling error units improves optimizer conditioning.
    """
    x = design(days)
    scale = max(float(np.std(errors)), 1.)
    y = errors/scale

    def objective(s, return_mean=False):
        sigma = x@s
        c = np.linalg.lstsq(x/sigma[:, None], y/sigma, rcond=None)[0]
        residual = y-x@c
        value = np.sum(np.log(sigma)+.5*(residual/sigma)**2)
        gradient = x.T@(1/sigma-residual**2/sigma**3)
        return c if return_mean else (float(value), gradient)

    if initial_spread is None:
        horizons = np.unique(days)
        spread = np.array([np.std(y[days == h], ddof=1) for h in horizons])
    else:
        horizons, physical_spread = initial_spread
        spread = np.asarray(physical_spread)/scale
    initial = lsq_linear(design(horizons), spread, bounds=([1e-6, 0, 0], np.inf)).x
    trials = []
    for start in (initial, np.array([max(float(np.std(y)), .1), 0, 0])):
        result = minimize(objective, start, jac=True, method='L-BFGS-B',
                          bounds=[(1e-6, None), (0, None), (0, None)],
                          options={'maxiter': 3000, 'ftol': 1e-13, 'gtol': 1e-7, 'maxls': 50})
        if result.success and np.isfinite(result.fun):
            trials.append(result)
    if not trials:
        raise RuntimeError('Quadratic spread likelihood optimization did not converge')
    best = min(trials, key=lambda fit: fit.fun)
    return {'mean_coefficients_m': (objective(best.x, True)*scale).tolist(),
            'sigma_coefficients_m': (best.x*scale).tolist(),
            'optimization': {'success': True, 'message': str(best.message),
                             'nll_without_constant': float(best.fun+len(y)*np.log(scale)),
                             'iterations': int(best.nit)}}


def fit_model(days, errors):
    days, errors = np.asarray(days, float), np.asarray(errors, float)
    if errors.shape != (len(days), 3) or len(days) < 6 or len(np.unique(days)) < 3:
        raise ValueError('Need paired 3-D errors at at least three horizons')
    if not np.all(np.isfinite(errors)) or not np.all(np.isfinite(days)):
        raise ValueError('Nonfinite training inputs')
    if np.any(days <= 0):
        raise ValueError('Forecast horizons must be positive')
    bins = {}
    for horizon in np.unique(days):
        selected = errors[days == horizon]
        if len(selected) < 2:
            raise ValueError('Each covariance bin needs at least two errors')
        covariance = np.cov(selected, rowvar=False, ddof=1)
        bins[str(float(horizon*24))] = {
            'rows': len(selected), 'mean_rtn_m': selected.mean(axis=0).tolist(),
            'sample_covariance_rtn_m2': covariance.tolist(),
            'sample_sigma_rtn_m': np.sqrt(np.diag(covariance)).tolist()}
    directions = [_fit_direction(days, errors[:, j]) for j in range(3)]
    c = np.array([d['mean_coefficients_m'] for d in directions])
    s = np.array([d['sigma_coefficients_m'] for d in directions])
    z = (errors-design(days)@c.T)/(design(days)@s.T)
    gamma = np.corrcoef(z, rowvar=False)
    if not np.all(np.isfinite(gamma)):
        raise ValueError('Training residual correlation is undefined')
    # A sample correlation is PSD; make a numerically singular sample explicit.
    eigenvalues, vectors = np.linalg.eigh(gamma)
    floor = 1e-8
    clipped = bool(eigenvalues.min() < floor)
    if clipped:
        gamma = (vectors*np.maximum(eigenvalues, floor))@vectors.T
        diagonal = np.sqrt(np.diag(gamma))
        gamma /= diagonal[:, None]*diagonal[None, :]
    np.linalg.cholesky(gamma)
    return {'time_unit': 'days', 'position_unit': 'm',
            'supported_horizon_days': [float(days.min()), float(days.max())],
            'training_rows': len(days), 'sample_bins_by_hours': bins,
            'directions': dict(zip(('R', 'T', 'N'), directions)),
            'correlation': gamma.tolist(), 'correlation_eigenvalue_floor_applied': clipped,
            'correlation_minimum_eigenvalue_before_floor': float(eigenvalues.min()),
            'mean_coefficients_m': c.tolist(), 'sigma_coefficients_m': s.tolist()}


def evaluate_model(model, days):
    days = np.asarray(days, float)
    lower, upper = model['supported_horizon_days']
    if np.any(~np.isfinite(days)) or np.any(days < lower-1e-10) or np.any(days > upper+1e-10):
        raise ValueError('Requested horizon is outside the fitted range')
    x = design(days)
    mean = x@np.array(model['mean_coefficients_m']).T
    sigma = x@np.array(model['sigma_coefficients_m']).T
    if np.any(sigma <= 0):
        raise ValueError('Nonpositive model standard deviation')
    covariance = sigma[..., :, None]*np.asarray(model['correlation'])*sigma[..., None, :]
    np.linalg.cholesky(covariance)
    return mean, covariance


def score_model(model, days, errors, predicted_to_reference=None):
    mean, covariance = evaluate_model(model, days)
    if predicted_to_reference is not None:
        # Scoring residuals use reference RTN, but inference rotates model
        # parameters with the predicted basis. Account for this exactly.
        rotation = np.asarray(predicted_to_reference)
        mean = np.einsum('nij,nj->ni', rotation, mean)
        covariance = rotation@covariance@np.swapaxes(rotation, -1, -2)
    scores = {}
    for name, residual in [('uncorrected', errors), ('corrected', errors-mean)]:
        d2 = np.einsum('ni,ni->n', residual, np.linalg.solve(covariance, residual[..., None])[..., 0])
        nll = .5*(3*np.log(2*np.pi)+np.linalg.slogdet(covariance)[1]+d2)
        scores[name] = {'rows': len(errors), 'position_rmse_m': float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
                        'mean_negative_log_likelihood': float(nll.mean()),
                        'coverage_95_percent': float(np.mean(d2 <= chi2.ppf(.95, 3))),
                        'coverage_by_level': {str(q): float(np.mean(d2 <= chi2.ppf(q, 3)))
                                              for q in (.5, .8, .9, .95, .99)}}
    return scores
