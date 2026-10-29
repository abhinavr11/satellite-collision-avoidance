"""Choose a longer lead without observing the collision test outcome."""
import numpy as np

from .data import SATELLITES
from .confidence import fit_confident_prefix


def select_balanced_window(states, leads=(120, 180, 240, 300, 360), threshold=.99,
                           max_rmse_ratio=.25, min_folds=3, max_folds=6,
                           windows=(300, 600, 900, 1800), bin_seconds=60):
    """Longest candidate passing common-prefix position checks for both objects.

    Every candidate is evaluated using reference data only through the earliest
    proposed outer cutoff. No candidate test Pc or test reference error enters
    the selection. More forecasts can fail the gate after the final refit; such
    a failure is kept rather than selecting another window from test outcomes.
    """
    leads = tuple(sorted(set(int(v) for v in leads)))
    if not leads or min(leads) <= 0:
        raise ValueError('Positive candidate leads required')
    common_cutoff = -float(max(leads))
    start = float(states['sgp4', 'Swarm-A']['t'][0])
    minimum_fit = min(windows)
    candidates = []
    for lead in leads:
        horizon = 2*lead
        folds = min(max_folds, int((common_cutoff-start-minimum_fit)//horizon))
        if folds < min_folds:
            candidates.append({'lead_seconds': lead, 'eligible': False,
                               'reason': 'Insufficient history for nonoverlapping validation blocks'})
            continue
        end = common_cutoff+horizon
        try:
            models, _, _, _, gates = fit_confident_prefix(
                states, common_cutoff, end, threshold, windows=windows, folds=folds,
                bin_seconds=bin_seconds, validation_horizon_seconds=horizon)
        except (ValueError, RuntimeError) as exc:
            candidates.append({'lead_seconds': lead, 'eligible': False, 'reason': str(exc)})
            continue
        objects = {}
        for name in SATELLITES:
            gate = gates[name]
            chosen = next(v for v in gate['candidate_windows'] if v['window_seconds'] == gate['selected_window_seconds'])
            ratio = chosen['validation_candidate_rmse_m']/chosen['validation_raw_rmse_m']
            objects[name] = {'selected_fit_window_seconds': gate['selected_window_seconds'],
                             'position_rmse_ratio': ratio,
                             'model_confidence': gate['minimum_test_horizon_model_confidence'],
                             'gate_passed': gate['apply_correction'],
                             'passes_substantial_position_improvement': bool(ratio <= max_rmse_ratio)}
        eligible = all(v['gate_passed'] and v['passes_substantial_position_improvement'] for v in objects.values())
        candidates.append({'lead_seconds': lead, 'test_duration_seconds': horizon,
                           'validation_folds': folds, 'eligible': eligible, 'objects': objects,
                           'latest_validation_truth_elapsed_seconds': common_cutoff})
    passing = [v for v in candidates if v['eligible']]
    if not passing:
        raise RuntimeError(f'No candidate passes the fixed position criteria: {candidates}')
    chosen = max(passing, key=lambda v: v['lead_seconds'])
    return {'selected_lead_seconds': chosen['lead_seconds'],
            'training_cutoff_seconds': -chosen['lead_seconds'],
            'test_end_seconds': chosen['lead_seconds'],
            'final_rolling_validation_folds': chosen['validation_folds'],
            'selection_reference_cutoff_seconds': common_cutoff,
            'confidence_threshold': threshold, 'maximum_validation_rmse_ratio': max_rmse_ratio,
            'selection_rule': 'Longest lead passing training-only position improvement and confidence checks for both satellites',
            'selection_uses_test_truth': False, 'selection_uses_pc': False, 'candidates': candidates}
