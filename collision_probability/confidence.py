"""Training-only selection and confidence gating for encounter corrections."""
import numpy as np
from scipy.stats import norm

from .data import SATELLITES
from .model import rtn_basis
from .encounter_forecast import fit_encounter_model, evaluate_encounter_model, split_times


def candidate_correction(mean, covariance):
    """The earlier experiment's variance shrinkage, with full covariance."""
    mean, covariance = np.asarray(mean), np.asarray(covariance)
    variance = np.diagonal(covariance, axis1=-2, axis2=-1)
    weights = mean**2/(mean**2+variance)
    correction = weights*mean
    projected_variance = np.einsum('...i,...ij,...j->...', correction, covariance, correction)
    numerator = np.sum(mean*correction, axis=-1)-.5*np.sum(correction**2, axis=-1)
    denominator = np.sqrt(np.maximum(projected_variance, 0))
    confidence = np.full(np.shape(numerator), .5, dtype=float)
    np.divide(numerator, denominator, out=confidence, where=denominator > 0)
    confidence = np.where(denominator > 0, norm.cdf(confidence), .5)
    return correction, confidence


def _training_object(states, name, mask):
    reference, predicted = states['reference', name], states['sgp4', name]
    ref_basis = rtn_basis(reference['position'][mask], reference['velocity'][mask])
    pred_basis = rtn_basis(predicted['position'][mask], predicted['velocity'][mask])
    difference = reference['position'][mask]-predicted['position'][mask]
    ref_errors = np.einsum('nji,nj->ni', ref_basis, difference)
    pred_errors = np.einsum('nji,nj->ni', pred_basis, difference)
    return ref_errors, pred_errors


def fit_confident_prefix(states, cutoff, forecast_end, threshold=.99,
                         windows=(300, 600, 900, 1800), folds=6, bin_seconds=60,
                         validation_horizon_seconds=None):
    """Choose a fit window by rolling validation inside the training prefix.

    Confidence is a conditional Gaussian model probability, not an empirical
    99% guarantee. The complete test horizon must pass the frozen gate; there
    are no within-test on/off switches and no use of test truth in selection.
    """
    if not .5 < threshold < 1:
        raise ValueError('Confidence threshold must be strictly between .5 and 1')
    times = states['sgp4', 'Swarm-A']['t']
    train, test = split_times(times, cutoff)
    horizon = float(forecast_end-cutoff if validation_horizon_seconds is None else validation_horizon_seconds)
    if horizon <= 0 or (validation_horizon_seconds is None and forecast_end != times[-1]):
        raise ValueError('Forecast end must match the short dataset window')
    origins = cutoff-horizon*np.arange(folds, 0, -1)
    if any(origin not in times for origin in origins):
        raise ValueError('Rolling origins must fall on sampled training times')
    models, gates, errors = {}, {}, {}
    for name in SATELLITES:
        # Reference data are sliced to train BEFORE errors or window selection.
        training_times = times[train]
        reference_errors, predicted_errors = _training_object(states, name, train)
        errors[name] = reference_errors
        candidates, backtests = [], {}
        for window in windows:
            if window < 3*bin_seconds or origins[0]-window < training_times[0]:
                continue
            records, forecast_errors, forecast_covariances, true_errors = [], [], [], []
            for origin in origins:
                fit_mask = (training_times >= origin-window) & (training_times <= origin)
                validation = (training_times > origin) & (training_times <= origin+horizon)
                model = fit_encounter_model(training_times[fit_mask], reference_errors[fit_mask], float(origin), bin_seconds)
                mu, covariance = evaluate_encounter_model(model, training_times[validation]-origin)
                correction, confidence = candidate_correction(mu, covariance)
                actual = predicted_errors[validation]
                residual = actual-correction
                before = np.linalg.norm(actual, axis=1)
                after = np.linalg.norm(residual, axis=1)
                records.append({'origin_elapsed_seconds': float(origin),
                                'last_fit_elapsed_seconds': float(training_times[fit_mask][-1]),
                                'validation_first_elapsed_seconds': float(training_times[validation][0]),
                                'validation_last_elapsed_seconds': float(training_times[validation][-1]),
                                'rows': int(validation.sum()),
                                'raw_rmse_m': float(np.sqrt(np.mean(before**2))),
                                'candidate_rmse_m': float(np.sqrt(np.mean(after**2))),
                                'improved_fraction': float(np.mean(after < before)),
                                'minimum_model_confidence': float(np.min(confidence))})
                # Forecast-mean errors, rather than in-sample curve-fit errors,
                # calibrate how much trust to put in the tiny fitted spreads.
                forecast_errors.append(actual-mu)
                forecast_covariances.append(covariance)
                true_errors.append(actual)
            residuals = np.concatenate(forecast_errors)
            covariance = np.concatenate(forecast_covariances)
            actual = np.concatenate(true_errors)
            sigma = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))
            standardized = residuals/sigma
            inflation = np.maximum(1., np.sqrt(np.mean(standardized**2, axis=0)))
            gamma = np.corrcoef(standardized, rowvar=False)
            values, vectors = np.linalg.eigh(gamma)
            gamma = (vectors*np.maximum(values, 1e-6))@vectors.T
            d = np.sqrt(np.diag(gamma))
            gamma /= d[:, None]*d[None, :]
            calibrated_covariance = (sigma*inflation)[:, :, None]*gamma*(sigma*inflation)[:, None, :]
            # Re-score the variance-shrunk candidate with calibrated spread.
            # This remains wholly inside the outer training prefix.
            forecast_means = actual-residuals
            corrected, confidence = candidate_correction(forecast_means, calibrated_covariance)
            cursor = 0
            for record in records:
                count = record['rows']
                before = np.linalg.norm(actual[cursor:cursor+count], axis=1)
                after = np.linalg.norm(actual[cursor:cursor+count]-corrected[cursor:cursor+count], axis=1)
                record['calibrated_candidate_rmse_m'] = float(np.sqrt(np.mean(after**2)))
                record['calibrated_improved_fraction'] = float(np.mean(after < before))
                record['calibrated_minimum_model_confidence'] = float(np.min(confidence[cursor:cursor+count]))
                cursor += count
            candidate_rmse = float(np.sqrt(np.mean(np.sum((actual-corrected)**2, axis=1))))
            passes = all(r['calibrated_candidate_rmse_m'] < r['raw_rmse_m'] and
                         r['calibrated_improved_fraction'] >= threshold for r in records)
            entry = {'window_seconds': int(window), 'rolling_validation_folds': records,
                     'validation_candidate_rmse_m': candidate_rmse,
                     'validation_raw_rmse_m': float(np.sqrt(np.mean(np.sum(actual**2, axis=1)))),
                     'passes_validation_position_check': passes,
                     'spread_inflation_rtn': inflation.tolist(),
                     'calibrated_correlation': gamma.tolist()}
            candidates.append(entry)
            backtests[int(window)] = entry
        if not candidates:
            raise ValueError('Training prefix too short for the specified rolling backtests')
        passing = [c for c in candidates if c['passes_validation_position_check']]
        selected = min(passing or candidates, key=lambda c: c['validation_candidate_rmse_m'])
        window = selected['window_seconds']
        fit_mask = training_times >= cutoff-window
        model = fit_encounter_model(training_times[fit_mask], reference_errors[fit_mask], cutoff, bin_seconds)
        inflation = np.array(selected['spread_inflation_rtn'])
        model['uncalibrated_sigma_coefficients_about_training_start'] = model['sigma_coefficients_about_training_start']
        model['uncalibrated_sigma_coefficients_about_cutoff'] = model['sigma_coefficients_about_cutoff']
        for key in ('sigma_coefficients_about_training_start', 'sigma_coefficients_about_cutoff'):
            model[key] = (np.asarray(model[key])*inflation[:, None]).tolist()
        model['uncalibrated_correlation'] = model['correlation']
        model['correlation'] = selected['calibrated_correlation']
        model['validation_spread_inflation_rtn'] = inflation.tolist()
        model['selected_window_seconds'] = window
        model['covariance_calibration_uses_training_backtests_only'] = True
        lead = times[(times >= cutoff) & (times <= cutoff+horizon)]-cutoff
        mu, covariance = evaluate_encounter_model(model, lead)
        _, confidence = candidate_correction(mu, covariance)
        confidence_passes = bool(np.min(confidence) >= threshold)
        gate = {'threshold': threshold, 'selection_uses_test_reference': False,
                'selected_window_seconds': window, 'candidate_windows': candidates,
                'validation_position_check_passed': selected['passes_validation_position_check'],
                'minimum_test_horizon_model_confidence': float(np.min(confidence)),
                'model_confidence_check_passed': confidence_passes,
                'apply_correction': bool(selected['passes_validation_position_check'] and confidence_passes),
                'decision_frozen_at_elapsed_seconds': cutoff,
                'correction_is_variance_shrunk': True,
                'interpretation': 'Conditional Gaussian confidence plus training-only rolling checks; not an empirically established 99% guarantee.'}
        model['training_prefix_rows_available'] = len(training_times)
        models[name], gates[name] = model, gate
    return models, errors, train, test, gates
