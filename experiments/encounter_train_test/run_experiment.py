"""Three probability forecasts issued at the end of one encounter training arc."""
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import csv
import importlib.metadata
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from collision_probability.data import SATELLITES, ensure_encounter, load_encounter, fingerprint
from collision_probability.encounter_forecast import (METHODS, fit_training_prefix, ForecastTrajectories,
                                                     evaluate_encounter_model, limit_encounter_end)
from collision_probability.confidence import fit_confident_prefix, candidate_correction
from collision_probability.window_selection import select_balanced_window
from collision_probability.model import rtn_basis
from collision_probability.evaluation import binary_scores
from experiments.collision_probability.run_experiment import evaluate_encounter, reference_outcome

LABELS = dict(zip(METHODS, ('SGP4 + frozen covariance', 'Gated correction + projected covariance', 'SGP4 + projected covariance')))
COLORS = dict(zip(METHODS, ('#bd6036', '#226ca5', '#7954a1')))
SHORT_LABELS = dict(zip(METHODS, ('frozen raw', 'gated', 'projected raw')))


def utc(center, seconds):
    return (center+timedelta(seconds=float(seconds))).isoformat(timespec='microseconds').replace('+00:00', 'Z')


def diagnostics(states, curves, train, test, half_window):
    """Retrospective validation only; called after models/predictions are fixed."""
    times = states['sgp4', 'Swarm-A']['t']
    observed, metrics = {}, {}
    for name in SATELLITES:
        ref, pred = states['reference', name], states['sgp4', name]
        basis = rtn_basis(ref['position'], ref['velocity'])
        observed[name] = np.einsum('nji,nj->ni', basis, ref['position']-pred['position'])
        metrics[name] = {}
        for method in METHODS:
            residual = ref['position'][test]-curves.position(method, name, times[test])
            event = np.abs(times[test]) <= half_window
            metrics[name][method] = {
                'test_position_rmse_m': float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
                'selected_event_position_rmse_m': float(np.sqrt(np.mean(np.sum(residual[event]**2, axis=1))))}
        before = np.linalg.norm(ref['position'][test]-curves.position(METHODS[2], name, times[test]), axis=1)
        after = np.linalg.norm(ref['position'][test]-curves.position(METHODS[1], name, times[test]), axis=1)
        changed = np.linalg.norm(curves.position(METHODS[1], name, times[test])-curves.position(METHODS[2], name, times[test]), axis=1) > 1e-8
        metrics[name]['correction_audit'] = {
            'test_samples': int(test.sum()), 'changed_samples': int(changed.sum()),
            'changed_and_improved': int(np.count_nonzero(changed & (after < before-1e-8))),
            'changed_and_worsened': int(np.count_nonzero(changed & (after > before+1e-8))),
            'unchanged_samples': int(np.count_nonzero(~changed)),
            'raw_error_at_reference_center_m': float(np.linalg.norm(curves.position(METHODS[2], name, 0)-curves.position('reference', name, 0))),
            'corrected_error_at_reference_center_m': float(np.linalg.norm(curves.position(METHODS[1], name, 0)-curves.position('reference', name, 0))),
            'test_truth_used_for_gate': False}
    return observed, metrics


def save_tables(output, metadata, states, models, curves, train, test, observed):
    times = states['sgp4', 'Swarm-A']['t']
    center = datetime.fromisoformat(metadata['synthetic_center_utc'].replace('Z', '+00:00'))
    with (output/'time_split.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['time_utc', 'elapsed_seconds', 'lead_from_training_end_seconds', 'split'])
        for t, training in zip(times, train):
            writer.writerow([utc(center, t), t, t-curves.cutoff, 'train' if training else 'test'])
    with (output/'error_samples.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['satellite', 'time_utc', 'elapsed_seconds', 'lead_seconds', 'split', 'error_r_m', 'error_t_m', 'error_n_m'])
        for name in SATELLITES:
            for t, error, training in zip(times, observed[name], train):
                writer.writerow([name, utc(center, t), t, t-curves.cutoff, 'train' if training else 'test', *error])
    with (output/'forecast_parameters.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['satellite', 'time_utc', 'elapsed_seconds', 'lead_seconds',
                         'projected_mu_r_m', 'projected_mu_t_m', 'projected_mu_n_m',
                         'projected_sigma_r_m', 'projected_sigma_t_m', 'projected_sigma_n_m',
                         'frozen_mu_r_m', 'frozen_mu_t_m', 'frozen_mu_n_m',
                         'frozen_sigma_r_m', 'frozen_sigma_t_m', 'frozen_sigma_n_m'])
        for name in SATELLITES:
            mean0, covariance0 = evaluate_encounter_model(models[name], 0)
            mean, covariance = evaluate_encounter_model(models[name], curves.forecast_times-curves.cutoff)
            for i, t in enumerate(curves.forecast_times):
                writer.writerow([name, utc(center, t), t, t-curves.cutoff, *mean[i],
                                 *np.sqrt(np.diag(covariance[i])), *mean0, *np.sqrt(np.diag(covariance0))])
    with (output/'confidence_audit.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['satellite', 'time_utc', 'elapsed_seconds', 'lead_seconds', 'model_improvement_probability',
                         'correction_allowed_by_frozen_gate', 'applied_correction_norm_m', 'raw_position_error_m',
                         'corrected_position_error_m', 'position_improved_afterward'])
        for name in SATELLITES:
            mu, covariance = evaluate_encounter_model(models[name], curves.times-curves.cutoff)
            _, confidence = candidate_correction(mu, covariance)
            for i, second in enumerate(curves.times):
                raw, corrected, truth = (curves.position(kind, name, second) for kind in (METHODS[2], METHODS[1], 'reference'))
                before, after = np.linalg.norm(raw-truth), np.linalg.norm(corrected-truth)
                writer.writerow([name, utc(center, second), second, second-curves.cutoff, confidence[i],
                                 curves.gates[name]['apply_correction'] if curves.gates is not None else True,
                                 np.linalg.norm(corrected-raw), before, after, bool(after < before)])
    with (output/'forecast_trajectories.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        columns = ['time_utc', 'elapsed_seconds', 'lead_seconds']
        for name in SATELLITES:
            for kind in ('reference', *METHODS):
                columns += [name.lower().replace('-', '_')+'_'+kind+'_'+axis+'_m' for axis in ('x', 'y', 'z')]
        writer.writerow(columns)
        for t in curves.times:
            writer.writerow([utc(center, t), t, t-curves.cutoff,
                             *[float(value) for name in SATELLITES for kind in ('reference', *METHODS)
                               for value in curves.position(kind, name, t)]])


def plots(summary, states, models, curves, observed, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    times = np.linspace(-8, 8, 1601)
    for method in METHODS:
        axes[0, 0].semilogy(times, np.linalg.norm(curves.relative(method, times), axis=1),
                            color=COLORS[method], label=LABELS[method],
                            linestyle='--' if method == METHODS[2] else '-')
        closest = summary['encounter_results'][method]['closest_approach']
        axes[0, 0].scatter(closest['elapsed_seconds'], closest['miss_distance_m'], color=COLORS[method], s=25, zorder=5)
    axes[0, 0].semilogy(times, np.linalg.norm(curves.relative('reference', times), axis=1), color='#368357', label='Reference ground truth')
    axes[0, 0].axhline(summary['combined_radius_m'], color='red', linestyle=':', label='Collision distance')
    axes[0, 0].set(xlabel='Seconds from saved crossing', ylabel='Separation (m)', title='All forecasts issued at the same training cutoff')
    axes[0, 0].legend(fontsize=8)
    names = ['Frozen / raw', 'Projected / gated', 'Projected / raw']
    pcs = [summary['encounter_results'][m]['quadrature']['pc'] for m in METHODS]
    bars = axes[0, 1].bar(names, pcs, color=[COLORS[m] for m in METHODS])
    axes[0, 1].bar_label(bars, labels=[f'{pc:.4g}' if pc > 0 else f'Underflow\nlog Pc={summary["encounter_results"][m]["quadrature"]["log_pc"]:.1f}' for m, pc in zip(METHODS, pcs)], padding=3, fontsize=8)
    axes[0, 1].set(ylabel='Predicted Pc', title=f"Reference outcome y={summary['ground_truth']['collision_label']}")
    losses = [summary['outcome_scores'][m]['log_loss'] for m in METHODS]
    bars = axes[1, 0].bar(names, losses, color=[COLORS[m] for m in METHODS])
    axes[1, 0].bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    axes[1, 0].set(ylabel='Log loss (lower is better)', title='One-event scores; general calibration is untested')
    for name, style in zip(SATELLITES, ('-', '--')):
        leads = np.linspace(0, curves.forecast_times[-1]-curves.cutoff, 200)
        for method in METHODS:
            budget = curves.local_error_budget(name, curves.cutoff+leads, method)
            axes[1, 1].plot(leads/60, np.sqrt(np.trace(budget, axis1=-2, axis2=-1)), style,
                            color=COLORS[method], alpha=.8, label=name+' / '+SHORT_LABELS[method])
    axes[1, 1].axvline(-curves.cutoff/60, color='black', linestyle=':', label='Reference crossing')
    axes[1, 1].set(xlabel='Minutes after training cutoff', ylabel='sqrt(trace(Gaussian error budget)) (m)',
                   title='Unremoved bias included when second-moment mode is used')
    axes[1, 1].legend(fontsize=7)
    fig.suptitle('Within-encounter train/test: three frozen-origin forecasts', fontsize=14)
    fig.savefig(output/'results.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout='constrained')
    full_times = states['sgp4', 'Swarm-A']['t']
    for row, name in enumerate(SATELLITES):
        shown = full_times >= models[name]['training_first_elapsed_seconds']
        lead = full_times[shown]-curves.cutoff
        training = full_times[shown] <= curves.cutoff
        mean, covariance = evaluate_encounter_model(models[name], lead)
        sigma = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))
        for j, direction in enumerate(('R', 'T', 'N')):
            ax = axes[row, j]
            ax.plot(lead[training]/60, observed[name][shown][training, j], color='#368357', label='Observed train error')
            ax.plot(lead[~training]/60, observed[name][shown][~training, j], color='#666666', label='Held-out error')
            ax.plot(lead/60, mean[:, j], color='#226ca5', label='Quadratic mean')
            ax.fill_between(lead/60, mean[:, j]-sigma[:, j], mean[:, j]+sigma[:, j], color='#226ca5', alpha=.15, label='Mean ± sigma')
            ax.axvline(0, color='black', linestyle=':')
            ax.axvline(-curves.cutoff/60, color='red', linestyle=':')
            ax.set(title=name+' / '+direction, xlabel='Minutes from training cutoff', ylabel='RTN error (m)')
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('Recent fitting windows selected inside training; test truth shown only afterward', fontsize=14)
    fig.savefig(output/'quadratic_forecast.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for ax, name in zip(axes, SATELLITES):
        reference = curves.position('reference', name, curves.times)
        for method, label, color in [(METHODS[2], 'Original SGP4', '#bd6036'),
                                      (METHODS[1], 'Gated correction', '#226ca5')]:
            error = np.linalg.norm(curves.position(method, name, curves.times)-reference, axis=1)
            ax.semilogy(curves.times, error, label=label, color=color)
        ax.axvline(0, color='black', linestyle=':', label='Reference crossing')
        ax.set(title=name, xlabel='Seconds from reference crossing', ylabel='3-D position error (m; log scale)')
        ax.legend(fontsize=8)
    fig.suptitle('Held-out position errors; truth used only for this retrospective audit', fontsize=12)
    fig.savefig(output/'position_errors.png', dpi=180)
    plt.close(fig)
    selection = summary.get('window_selection')
    if selection is not None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
        candidates = [c for c in selection['candidates'] if 'objects' in c]
        minutes = [c['lead_seconds']/60 for c in candidates]
        for name, color in zip(SATELLITES, ('#226ca5', '#bd6036')):
            axes[0].plot(minutes, [c['objects'][name]['position_rmse_ratio'] for c in candidates], 'o-', color=color, label=name)
            axes[1].plot(minutes, [100*c['objects'][name]['model_confidence'] for c in candidates], 'o-', color=color, label=name)
        axes[0].axhline(selection['maximum_validation_rmse_ratio'], color='black', linestyle=':', label='Maximum allowed ratio')
        axes[1].axhline(100*selection['confidence_threshold'], color='black', linestyle=':', label='Required model confidence')
        for ax in axes:
            ax.axvline(selection['selected_lead_seconds']/60, color='#368357', linestyle='--', label='Selected lead')
            ax.set(xlabel='Candidate forecast lead (minutes)', xticks=minutes)
            ax.legend(fontsize=7)
        axes[0].set(ylabel='Corrected / raw position RMSE', title='Training-only position forecasts; lower is better', ylim=(0, .3))
        axes[1].set(ylabel='Conditional model confidence (%)', title='Both satellites must pass the existing gate')
        fig.suptitle('Lead selection without evaluating candidate test Pc', fontsize=13)
        fig.savefig(output/'window_selection.png', dpi=180)
        plt.close(fig)


def probability_text(result):
    value = result['quadrature']['pc']
    if value > 0:
        return f'{value:.9g}'
    log_value = result['quadrature']['log_pc']
    exponent = int(np.floor(log_value/np.log(10)))
    mantissa = np.exp(log_value-exponent*np.log(10))
    return f'{mantissa:.4g}e{exponent} (stored Pc underflows)'


def report(summary, output):
    split, truth = summary['split'], summary['ground_truth']
    gates, selection = summary['confidence_gates'], summary['window_selection']
    winner = min(METHODS, key=lambda m: summary['outcome_scores'][m]['log_loss'])
    lines = ['# Balanced forecast window and explicit uncertainty baselines', '',
             f"**Selected lead: {abs(summary['training_cutoff_elapsed_seconds'])/60:g} minutes. Test duration: {(summary['split']['test_samples_per_satellite'])/60:g} minutes. Best score on this constructed event: {LABELS[winner]}.**", '',
             '## Window choice without test-probability selection', '',
             f"Training prefix: {split['training_first_utc']} through **{split['training_last_utc']}** ({split['training_samples_per_satellite']} timestamps per object). Test: **{split['test_first_utc']} through {split['test_last_utc']}** ({split['test_samples_per_satellite']} timestamps). Later source data are excluded.", '']
    if selection is not None:
        lines += [f"Candidate leads were 2, 3, 4, 5 and 6 minutes, with symmetric test windows. All selection reference data end at elapsed time {selection['selection_reference_cutoff_seconds']:g} s, the earliest candidate training cutoff. No candidate test reference outcome, test position error or Pc is used to choose the split.", '',
                  'The fixed rule chooses the longest lead for which both objects pass the 99% confidence gate and training-only rolling validation reduces position RMSE by at least 75%. Between three and six nonoverlapping validation blocks are used, depending on available history. A final refit can still withhold correction; the test result is never used to select another split.', '',
                  '| Candidate lead (min) | Training validation blocks | A: corrected/raw RMSE | C: corrected/raw RMSE | Eligible for both |', '|---|---:|---:|---:|---|']
        for candidate in selection['candidates']:
            objects = candidate.get('objects', {})
            a, c = objects.get('Swarm-A', {}), objects.get('Swarm-C', {})
            lines.append(f"| {candidate['lead_seconds']/60:g} | {candidate.get('validation_folds', 0)} | {a.get('position_rmse_ratio', float('nan')):.4f} | {c.get('position_rmse_ratio', float('nan')):.4f} | {candidate['eligible']} |")
        lines += ['', 'Four- to six-minute leads may have useful average position forecasts but fail at least one satellite’s existing 99% model gate. They are not adopted just to obtain more appealing probabilities.', '']
    else:
        lines += ['The window was explicitly fixed by command-line arguments; no automatic lead selection was performed.', '']
    lines += ['## Why the raw-SGP4 probabilities changed', '',
              f"This run uses **{summary['covariance_mode']}** uncertainty mode. Changing the window alone is not an appropriate way to repair implausible probabilities.", '',
              'The fitted Gaussian error model has bias mu and centred covariance P. P describes residual scatter after removing the fitted bias. Using only that narrow P around an uncorrected trajectory ignores the unremoved systematic offset. For an explicit zero-mean Gaussian baseline, the error second moment about the displayed mean is S = P + (mu−d)(mu−d)^T, where d is the applied correction. For raw SGP4 d=0; for a fully applied mean correction d=mu and S=P. A gated variance-shrunk correction leaves its small unapplied bias in S.', '',
              '**S is a second moment, not the true centred covariance of a biased error.** Assigning N(displayed mean,S) is a stated zero-mean Gaussian approximation and a different uncertainty assumption. The independent-object baseline adds the two object error budgets. These probabilities are conditional on that model; moderate-looking numbers do not prove calibration.', '',
              ('The original centred-residual-only comparison is retained below to isolate correction with the same residual covariance.' if summary['residual_covariance_controls'] else 'Residual mode uses the original common centred covariance directly; the second-moment model is an explicit alternative.'), '',
              '## Three primary forecasts', '',
              '| Technique | Displayed position mean | Local Gaussian error budget in second-moment mode |', '|---|---|---|',
              ('| SGP4 + frozen error budget | Original SGP4 | P(0) + mu(0)mu(0)^T, frozen |' if summary['covariance_mode'] == 'second_moment' else '| SGP4 + frozen covariance | Original SGP4 | P(0), frozen |'),
              ('| Gated correction + projected error budget | SGP4 + U(t)d(h) when permitted | P(h) + (mu(h)−d(h))(mu(h)−d(h))^T |' if summary['covariance_mode'] == 'second_moment' else '| Gated correction + projected covariance | SGP4 + U(t)d(h) when permitted | P(h) |'),
              ('| SGP4 + projected error budget | Original SGP4 | P(h) + mu(h)mu(h)^T |' if summary['covariance_mode'] == 'second_moment' else '| SGP4 + projected covariance | Original SGP4 | P(h) |'), '',
              'All matrices rotate with the predicted RTN basis into common GCRS. The original TLE epochs are preserved. Gates, fitting windows and covariance calibration are frozen at the forecast origin. Reference trajectories supply only a binary collision label, with no reference covariance or Pc.', '',
              f"Reference minimum separation: **{truth['closest_approach']['miss_distance_m']:.6f} m**. Assumed radii A={summary['radii_m']['Swarm-A']:g} m and C={summary['radii_m']['Swarm-C']:g} m give a {summary['combined_radius_m']:g} m combined collision distance. Outcome y={truth['collision_label']}.", '',
              '| Technique | Predicted miss (m) | Pc | Brier score | Log loss |', '|---|---:|---:|---:|---:|']
    for method in METHODS:
        result, score = summary['encounter_results'][method], summary['outcome_scores'][method]
        lines.append(f"| {LABELS[method]} | {result['closest_approach']['miss_distance_m']:.6f} | {probability_text(result)} | {score['brier_score']:.8f} | {score['log_loss']:.6f} |")
    if summary['residual_covariance_controls']:
        lines += ['', '## Original residual-covariance-only controls', '',
                  'Same selected window, models, gates and mean trajectories; only the Gaussian uncertainty assumption changes back to centred P for every method.', '',
                  '| Technique | Pc using only centred residual covariance | log(Pc) |', '|---|---:|---:|']
        for method, result in summary['residual_covariance_controls'].items():
            lines.append(f"| {LABELS[method]} | {probability_text(result)} | {result['quadrature']['log_pc']:.6f} |")
    lines += ['', 'Brier score is (Pc−y)^2 and binary log loss is −y log(Pc)−(1−y) log(1−Pc); lower is better. A higher Pc scores better for this selected collision case, not necessarily for other events. Retained log(Pc) distinguishes probabilities that underflow floating-point storage.', '',
              '## Confidence gating and held-out position accuracy', '',
              f"The threshold remains {100*summary['confidence_threshold']:g}% conditional Gaussian confidence that a variance-shrunk correction reduces position error. Candidate fitting windows are 5, 10, 15 and 30 minutes. Training-only rolling position forecasts choose the fit window and inflate residual spreads using their forecast errors. The gate must also improve RMSE and at least the threshold fraction of positions in every training validation block. It never uses test truth or Pc.", '',
              '| Satellite | Selected recent fit (s) | Minimum model confidence | Correction allowed |', '|---|---:|---:|---|']
    if gates is not None:
        for name, gate in gates.items():
            lines.append(f"| {name} | {gate['selected_window_seconds']} | {gate['minimum_test_horizon_model_confidence']:.10f} | {gate['apply_correction']} |")
    lines += ['', 'The following position audit is computed after predictions are frozen and never changes the gate:', '',
              '| Satellite | Changed | Improved | Worsened | Raw error at crossing (m) | Corrected error (m) |', '|---|---:|---:|---:|---:|---:|']
    for name, metrics in summary['held_out_position_diagnostics'].items():
        a = metrics['correction_audit']
        lines.append(f"| {name} | {a['changed_samples']} | {a['changed_and_improved']} | {a['changed_and_worsened']} | {a['raw_error_at_reference_center_m']:.6f} | {a['corrected_error_at_reference_center_m']:.6f} |")
    lines += ['', '## Limits and reproducibility', '',
              '- This is the same previously inspected, constructed collision case. It is not independent confirmation of probability accuracy. The 99% gate is model confidence, not an empirical success guarantee.',
              '- Reference states are interpolated and nearby samples/validation blocks are correlated. Gaussian errors, independent objects and illustrative radii remain assumptions. More moderate Pc values do not establish that the uncertainty is calibrated.',
              '- Training-only lead selection maximizes usable lead subject to fixed position criteria. Neither the test Pc nor a requirement that correction win enters selection. Longer failed candidates are retained in window_selection.json.',
              '- Tests perturb all reference data after the common selection cutoff and check that the selected lead is unchanged. Additional tests check second moments analytically and preserve means across uncertainty modes.',
              '- Source and model hashes, exact time split, gate and position audit, projected parameters, primary and residual-only matrices, quadrature refinements and Monte Carlo checks are saved.', '',
              '```sh', 'python experiments/encounter_train_test/run_experiment.py --data-dir /path/to/data',
              '# Reproduce the previous two-minute, residual-covariance-only experiment:',
              'python experiments/encounter_train_test/run_experiment.py --data-dir /path/to/data --training-cutoff-s -120 --test-end-s 120 --covariance-mode residual',
              '```', '', '![Primary forecasts](results.png)', '', '![Position errors](position_errors.png)', '', '![Recent quadratic fits](quadratic_forecast.png)', '']
    if selection is not None:
        lines += ['![Training-only window choice](window_selection.png)', '']
    (output/'report.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(os.environ.get('SGP4_DATA_DIR', ROOT/'data')))
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--training-cutoff-s', type=float)
    parser.add_argument('--window-policy', choices=('balanced', 'fixed'))
    parser.add_argument('--covariance-mode', choices=('second_moment', 'residual'), default='second_moment')
    parser.add_argument('--test-end-s', type=float)
    parser.add_argument('--confidence-threshold', type=float, default=.99)
    parser.add_argument('--no-confidence-gate', action='store_true')
    parser.add_argument('--bin-seconds', type=int, default=60)
    parser.add_argument('--radius-a-m', type=float, default=5.)
    parser.add_argument('--radius-c-m', type=float, default=5.)
    parser.add_argument('--mc-samples', type=int, default=1_000_000)
    parser.add_argument('--encounter-half-window-s', type=float, default=60.)
    args = parser.parse_args()
    if not np.all(np.isfinite([args.radius_a_m, args.radius_c_m])) or min(args.radius_a_m, args.radius_c_m, args.mc_samples, args.bin_seconds) <= 0:
        parser.error('Radii, samples and bin size must be positive')
    explicit_window = args.training_cutoff_s is not None or args.test_end_s is not None
    policy = args.window_policy or ('fixed' if explicit_window else 'balanced')
    if policy == 'balanced' and explicit_window:
        parser.error('Specify a fixed window or request balanced selection, not both')
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    bundle = ensure_encounter(args.data_dir.expanduser().resolve())
    metadata, states = load_encounter(bundle)
    selection = None
    if policy == 'balanced':
        selection = select_balanced_window(states, threshold=args.confidence_threshold, bin_seconds=args.bin_seconds)
        args.training_cutoff_s = selection['training_cutoff_seconds']
        args.test_end_s = selection['test_end_seconds']
        folds = selection['final_rolling_validation_folds']
        print(f"Training-only window choice: {selection['selected_lead_seconds']/60:g}-minute lead, {2*selection['selected_lead_seconds']/60:g}-minute test.", flush=True)
    else:
        args.training_cutoff_s = -120. if args.training_cutoff_s is None else args.training_cutoff_s
        args.test_end_s = 120. if args.test_end_s is None else args.test_end_s
        folds = 6
    if not (2 <= args.encounter_half_window_s < -args.training_cutoff_s and args.encounter_half_window_s < args.test_end_s):
        parser.error('Selected encounter window must be entirely inside the holdout')
    states, unused_samples = limit_encounter_end(states, args.test_end_s)
    if args.no_confidence_gate:
        models, training_errors, train, test = fit_training_prefix(states, args.training_cutoff_s, args.bin_seconds)
        gates = None
    else:
        models, training_errors, train, test, gates = fit_confident_prefix(
            states, args.training_cutoff_s, args.test_end_s, args.confidence_threshold,
            folds=folds, bin_seconds=args.bin_seconds)
    curves = ForecastTrajectories(states, models, args.training_cutoff_s, gates, covariance_mode=args.covariance_mode)
    radius = args.radius_a_m+args.radius_c_m
    results = {}
    for method in METHODS:
        result = evaluate_encounter(curves, method, metadata, radius, args.mc_samples, args.encounter_half_window_s)
        tca = result['closest_approach']['elapsed_seconds']
        result['forecast_lead_seconds'] = tca-curves.cutoff
        result['parameters_at_tca'] = {}
        for name in SATELLITES:
            mean, covariance = curves.local_parameters(name, tca, method)
            candidate, confidence = candidate_correction(mean, covariance)
            allowed = bool(method == METHODS[1] and (gates is None or gates[name]['apply_correction']))
            applied = (mean if gates is None else candidate) if allowed else np.zeros(3)
            result['parameters_at_tca'][name] = {'mean_error_rtn_m': mean.tolist(),
                                                'covariance_rtn_m2': covariance.tolist(),
                                                'error_budget_rtn_m2': curves.local_error_budget(name, tca, method).tolist(),
                                                'mean_correction_applied': allowed,
                                                'applied_correction_rtn_m': applied.tolist(),
                                                'candidate_improvement_probability': float(confidence)}
        results[method] = result
        print(f"{method}: lead={result['forecast_lead_seconds']:.3f} s; miss={result['closest_approach']['miss_distance_m']:.6f} m; Pc={probability_text(result)}", flush=True)
    controls = {}
    if args.covariance_mode == 'second_moment':
        residual_curves = ForecastTrajectories(states, models, args.training_cutoff_s, gates, covariance_mode='residual')
        for method in METHODS:
            controls[method] = evaluate_encounter(residual_curves, method, metadata, radius, args.mc_samples, args.encounter_half_window_s)
    truth = reference_outcome(curves, metadata, radius, args.encounter_half_window_s)
    scores = {m: binary_scores(r['quadrature']['pc'], truth['collision_label'], r['quadrature']['log_pc']) for m, r in results.items()}
    observed, metrics = diagnostics(states, curves, train, test, args.encounter_half_window_s)
    times = states['sgp4', 'Swarm-A']['t']
    center = datetime.fromisoformat(metadata['synthetic_center_utc'].replace('Z', '+00:00'))
    summary = {'experiment': 'within_encounter_training_prefix', 'training_cutoff_elapsed_seconds': curves.cutoff,
               'forecast_reference_utc': utc(center, curves.cutoff),
               'split': {'training_first_utc': utc(center, times[train][0]), 'training_last_utc': utc(center, times[train][-1]),
                         'test_first_utc': utc(center, times[test][0]), 'test_last_utc': utc(center, times[test][-1]),
                         'training_samples_per_satellite': int(train.sum()), 'test_samples_per_satellite': int(test.sum()), 'unused_samples_after_test': unused_samples},
               'confidence_gates': gates, 'confidence_threshold': args.confidence_threshold,
               'window_policy': policy, 'window_selection': selection, 'covariance_mode': args.covariance_mode,
               'residual_covariance_controls': controls,
               'radii_m': dict(zip(SATELLITES, (args.radius_a_m, args.radius_c_m))), 'combined_radius_m': radius,
               'encounter_half_window_seconds': args.encounter_half_window_s, 'models': models,
               'cutoff_parameters': {name: {'mean_error_rtn_m': evaluate_encounter_model(models[name], 0)[0].tolist(),
                                            'covariance_rtn_m2': evaluate_encounter_model(models[name], 0)[1].tolist()} for name in SATELLITES},
               'ground_truth': truth, 'encounter_results': results, 'outcome_scores': scores,
               'held_out_position_diagnostics': metrics, 'test_reference_used_for_fitting': False,
               'historical_calibration_pairs_used': False, 'single_event_evaluation_only': True,
               'forecast_projection_is_extrapolation': True, 'frozen_covariance_frame': 'RTN, rotated to future predicted GCRS',
               'input_bundle_id': metadata['dataset_id'], 'input_bundle_sha256': fingerprint(bundle/'encounter.json'),
               'input_trajectory_hashes': {k: metadata['outputs'][k]['sha256'] for k in ('reference', 'sgp4')},
               'software_versions': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'astropy', 'sgp4')}}
    if selection is not None:
        (output/'window_selection.json').write_text(json.dumps(selection, indent=2, allow_nan=False)+'\n')
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    save_tables(output, metadata, states, models, curves, train, test, observed)
    plots(summary, states, models, curves, observed, output)
    report(summary, output)
    print('Saved encounter-prefix train/test forecasts, coefficients, comparison and diagnostics.', flush=True)


if __name__ == '__main__':
    main()
