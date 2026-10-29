"""Sections 5–6: relative covariance, encounter plane and Gaussian disk integral."""
from functools import lru_cache
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize_scalar


def checked_covariance(covariance, dimension):
    covariance = np.asarray(covariance, float)
    if covariance.shape != (dimension, dimension) or not np.all(np.isfinite(covariance)):
        raise ValueError('Covariance has the wrong shape or nonfinite entries')
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-10):
        raise ValueError('Covariance must be symmetric')
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError('Covariance must be positive definite') from exc
    return covariance


def encounter_plane(relative_velocity):
    velocity = np.asarray(relative_velocity, float)
    speed = np.linalg.norm(velocity)
    if velocity.shape != (3,) or not np.isfinite(speed) or speed < 1e-9:
        raise ValueError('Encounter plane requires nonzero relative velocity')
    q3 = velocity/speed
    axis = np.eye(3)[np.argmin(np.abs(q3))]
    q1 = np.cross(q3, axis)
    q1 /= np.linalg.norm(q1)
    return np.column_stack((q1, np.cross(q3, q1)))


@lru_cache(maxsize=16)
def _legendre(order):
    return np.polynomial.legendre.leggauss(order)


def _disk_quadrature(mean, covariance, radius, order):
    xr, wr = _legendre(order)
    xt, wt = _legendre(2*order)
    rho, theta = radius*(xr+1)/2, np.pi*(xt+1)
    points = np.stack((rho[:, None]*np.cos(theta), rho[:, None]*np.sin(theta)), axis=-1)
    residual = points-mean
    whitened = np.linalg.solve(np.linalg.cholesky(covariance), residual.reshape(-1, 2).T).T
    log_density = -.5*np.sum(whitened**2, axis=1).reshape(order, 2*order)
    log_density -= np.log(2*np.pi)+.5*np.linalg.slogdet(covariance)[1]
    weights = (wr*radius/2*rho)[:, None]*(wt*np.pi)[None, :]
    log_probability = float(logsumexp(log_density+np.log(weights)))
    return float(np.exp(log_probability)), log_probability


def collision_probability(mean, covariance, radius, rtol=1e-6, atol=1e-12, max_order=512):
    mean = np.asarray(mean, float)
    covariance = checked_covariance(covariance, 2)
    if mean.shape != (2,) or not np.all(np.isfinite(mean)):
        raise ValueError('Encounter-plane mean must have two finite components')
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('Combined hard-body radius must be positive')
    previous = None
    history = []
    for order in (16, 32, 64, 128, 256, 512):
        if order > max_order:
            break
        value, log_value = _disk_quadrature(mean, covariance, radius, order)
        history.append({'radial_order': order, 'angular_order': 2*order, 'pc': value})
        # Compare log values too: two underflowed zeroes do not prove convergence.
        if previous is not None:
            difference = abs(value-previous[0])
            log_agreement = abs(log_value-previous[1]) <= rtol
            if difference <= atol+rtol*abs(value) and (value > atol or log_agreement):
                if value > 1+atol:
                    raise RuntimeError('Quadrature returned probability greater than one')
                return {'pc': min(value, 1.), 'log_pc': log_value, 'converged': True,
                        'absolute_difference_last_refinement': difference, 'history': history}
        previous = value, log_value
    raise RuntimeError(f'Collision integral did not converge: {history}')


def monte_carlo(mean, covariance, radius, samples=1_000_000, seed=20261029):
    covariance = checked_covariance(covariance, 2)
    if samples <= 0 or radius <= 0:
        raise ValueError('Positive sample count and radius required')
    rng = np.random.default_rng(seed)
    hits = 0
    for start in range(0, samples, 100_000):
        points = rng.multivariate_normal(mean, covariance, size=min(100_000, samples-start))
        hits += int(np.count_nonzero(np.sum(points**2, axis=1) <= radius**2))
    p = hits/samples
    z = 1.959963984540054
    denominator = 1+z*z/samples
    center = (p+z*z/(2*samples))/denominator
    width = z*np.sqrt(p*(1-p)/samples+z*z/(4*samples*samples))/denominator
    return {'samples': samples, 'hits': hits, 'pc': p, 'seed': seed,
            'wilson_95_interval': [max(0., center-width), min(1., center+width)],
            'zero_hits_is_not_proof_of_zero_probability': True}


def closest_approach(relative_position, sample_times):
    times = np.asarray(sample_times, float)
    if times.ndim != 1 or len(times) < 3 or np.any(np.diff(times) <= 0):
        raise ValueError('Need an increasing time grid')
    values = np.sum(relative_position(times)**2, axis=-1)
    candidates = [(float(values[0]), float(times[0])), (float(values[-1]), float(times[-1]))]
    indices = np.flatnonzero((values[1:-1] <= values[:-2]) & (values[1:-1] <= values[2:]))+1
    for i in indices:
        optimum = minimize_scalar(lambda t: float(np.sum(relative_position(t)**2)),
                                  bounds=(times[i-1], times[i+1]), method='bounded',
                                  options={'xatol': 1e-9})
        if not optimum.success:
            raise RuntimeError('Closest-approach minimization failed')
        candidates.append((float(optimum.fun), float(optimum.x)))
    distance_squared, tca = min(candidates)
    return {'elapsed_seconds': tca, 'miss_distance_m': float(np.sqrt(distance_squared)),
            'boundary_minimum': tca in (times[0], times[-1]), 'local_minima_refined': len(indices)}


def project_encounter(relative_position, relative_velocity, covariance_a, covariance_c):
    relative_covariance = checked_covariance(covariance_a, 3)+checked_covariance(covariance_c, 3)
    q = encounter_plane(relative_velocity)
    b, c = q.T@relative_position, q.T@relative_covariance@q
    checked_covariance(c, 2)
    return q, b, c, relative_covariance
