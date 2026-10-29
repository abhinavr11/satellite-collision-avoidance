"""Run Sections 3–6 of the guide on the fixed shifted Swarm A/C encounter."""
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import csv
import json
import importlib.metadata
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from collision_probability.data import (SATELLITES, ensure_encounter, load_encounter,
                                       calibration_errors, EncounterTrajectories, fingerprint)
from collision_probability.evaluation import collision_label, binary_scores
from collision_probability.model import fit_model, score_model, evaluate_model
from collision_probability.probability import (closest_approach, project_encounter,
                                              collision_probability, monte_carlo)


def short_encounter_check(curves, kind, tca, q, covariance, plane_covariance, radius):
    velocity = curves.relative_velocity(kind, tca)
    speed = np.linalg.norm(velocity)
    line = velocity/speed
    duration = float((3*np.sqrt(line@covariance@line)+radius)/speed)
    inside = curves.times[0] <= tca-duration and tca+duration <= curves.times[-1]
    if not inside:
        return {'supported': False, 'reason': 'Three-sigma encounter extends beyond available trajectory window',
                'three_sigma_half_duration_s': duration}
    offsets = np.array([-duration, -.5*duration, .5*duration, duration])
    relative = curves.relative(kind, tca)
    nonlinear = curves.relative(kind, tca+offsets)-relative-offsets[:, None]*velocity
    nonlinear_plane_error = float(np.max(np.linalg.norm(nonlinear@q, axis=1)))
    scale = max(radius, float(np.sqrt(np.linalg.eigvalsh(plane_covariance).min())))
    nearby_v = curves.relative_velocity(kind, tca+offsets)
    direction_change = float(np.max(np.linalg.norm(nearby_v/np.linalg.norm(nearby_v, axis=1)[:, None]-line, axis=1)))
    covariance_change = max(float(np.linalg.norm(sum(curves.covariance(n, tca+dt, kind) for n in SATELLITES)-covariance)/np.linalg.norm(covariance))
                            for dt in offsets)
    return {'supported': bool(nonlinear_plane_error < .1*scale and direction_change < .05 and covariance_change < .1),
            'three_sigma_half_duration_s': duration,
            'max_projected_nonlinearity_m': nonlinear_plane_error,
            'nonlinearity_scale_m': scale, 'max_velocity_direction_change': direction_change,
            'max_fractional_covariance_change': covariance_change,
            'criteria': 'projected curvature < 10% of max(radius, minimum plane sigma); direction change < .05; covariance change < 10%',
            'interpretation': 'Local numerical diagnostic, not independent validation of Gaussian tails or temporal uncertainty.'}


def evaluate_encounter(curves, kind, metadata, radius, samples, half_window):
    assessment_times = curves.times[np.abs(curves.times) <= half_window]
    closest = closest_approach(lambda t: curves.relative(kind, t), assessment_times)
    t = closest['elapsed_seconds']
    if closest['boundary_minimum']:
        raise RuntimeError(f'{kind} closest approach is at the window boundary; extend the window')
    relative, velocity = curves.relative(kind, t), curves.relative_velocity(kind, t)
    pa, pc = (curves.covariance(n, t, kind) for n in SATELLITES)
    q, b, covariance, relative_covariance = project_encounter(relative, velocity, pa, pc)
    integral = collision_probability(b, covariance, radius)
    simulation = monte_carlo(b, covariance, radius, samples=samples)
    sigma = np.sqrt(max(integral['pc']*(1-integral['pc']), 1/samples)/samples)
    if abs(simulation['pc']-integral['pc']) > 6*sigma+3/samples:
        raise RuntimeError('Monte Carlo disagrees with quadrature beyond sampling tolerance')
    center = datetime.fromisoformat(metadata['synthetic_center_utc'].replace('Z', '+00:00'))
    return {'closest_approach': closest, 'tca_utc': (center+timedelta(seconds=t)).isoformat(),
            'relative_mean_gcrs_m': relative.tolist(), 'relative_velocity_gcrs_m_s': velocity.tolist(),
            'relative_speed_m_s': float(np.linalg.norm(velocity)),
            'relative_covariance_gcrs_m2': relative_covariance.tolist(),
            'object_covariances_gcrs_m2': {n: curves.covariance(n, t, kind).tolist() for n in SATELLITES},
            'encounter_plane_basis': q.tolist(), 'plane_mean_m': b.tolist(),
            'plane_covariance_m2': covariance.tolist(), 'plane_sigma_eigenaxes_m': np.sqrt(np.linalg.eigvalsh(covariance)).tolist(),
            'along_velocity_mean_m': float(relative@velocity/np.linalg.norm(velocity)),
            'quadrature': integral, 'monte_carlo': simulation,
            'short_encounter_diagnostic': short_encounter_check(curves, kind, t, q, relative_covariance, covariance, radius)}


def reference_outcome(curves, metadata, radius, half_window):
    """Ground truth is an outcome from reference geometry, without covariance."""
    assessment_times = curves.times[np.abs(curves.times) <= half_window]
    closest = closest_approach(lambda t: curves.relative('reference', t), assessment_times)
    if closest['boundary_minimum']:
        raise RuntimeError('Reference minimum is at the assessment boundary; extend the window')
    label = collision_label(closest['miss_distance_m'], radius)
    center = datetime.fromisoformat(metadata['synthetic_center_utc'].replace('Z', '+00:00'))
    return {'collision_label': label, 'outcome': 'collision' if label else 'no_collision',
            'closest_approach': closest, 'combined_collision_distance_m': radius,
            'tca_utc': (center+timedelta(seconds=closest['elapsed_seconds'])).isoformat(),
            'reference_treated_as_exact': True, 'covariance_assigned': False,
            'interpretation': 'Deterministic outcome in the constructed encounter under the stated physical radii.'}


def write_trajectories(curves, metadata, output):
    center = datetime.fromisoformat(metadata['synthetic_center_utc'].replace('Z', '+00:00'))
    names = ['time_utc', 'elapsed_seconds']
    data = []
    for name in SATELLITES:
        for kind in ('reference', 'uncorrected', 'corrected'):
            prefix = name.lower().replace('-', '_')+'_'+kind
            names += [prefix+'_'+axis+'_m' for axis in ('x', 'y', 'z')]
            data.append(curves.position(kind, name, curves.times))
    with (output/'encounter_trajectories.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        for i, t in enumerate(curves.times):
            writer.writerow([(center+timedelta(seconds=float(t))).isoformat(), float(t),
                             *[f'{v:.9f}' for positions in data for v in positions[i]]])


def plots(summary, curves, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout='constrained')
    colors = {'uncorrected': '#bd6036', 'corrected': '#226ca5'}
    labels = {'uncorrected': 'Uncorrected', 'corrected': 'Bias corrected'}
    times = np.linspace(-8, 8, 1601)
    for kind in colors:
        axes[0, 0].semilogy(times, np.linalg.norm(curves.relative(kind, times), axis=1), color=colors[kind], label=labels[kind])
    axes[0, 0].semilogy(times, np.linalg.norm(curves.relative('reference', times), axis=1),
                        color='#368357', label='Reference: ground-truth path')
    axes[0, 0].axhline(summary['combined_radius_m'], color='red', linestyle=':', label='Collision distance')
    axes[0, 0].set(xlabel='Seconds from reference crossing', ylabel='Separation (m)', title='Reference trajectory determines collision outcome')
    axes[0, 0].legend(fontsize=8)
    results = summary['encounter_results']
    bars = axes[0, 1].bar([labels[k] for k in colors], [results[k]['quadrature']['pc'] for k in colors], color=list(colors.values()))
    axes[0, 1].bar_label(bars, fmt='%.6g', padding=3, fontsize=9)
    axes[0, 1].set(ylabel='Predicted collision probability', ylim=(0, max(r['quadrature']['pc'] for r in results.values())*1.2),
                   title=f"Two forecasts; reference outcome y={summary['ground_truth']['collision_label']}")
    losses = [summary['outcome_scores'][k]['log_loss'] for k in colors]
    bars = axes[1, 0].bar([labels[k] for k in colors], losses, color=list(colors.values()))
    axes[1, 0].bar_label(bars, fmt='%.6f', padding=3, fontsize=9)
    axes[1, 0].set(ylabel='Log loss (lower is better)', ylim=(0, max(losses)*1.15),
                   title='Forecast scores against the binary reference outcome')
    for name, style in [('Swarm-A', '-'), ('Swarm-C', '--')]:
        reference = curves.position('reference', name, times)
        for kind in ('uncorrected', 'corrected'):
            error = np.linalg.norm(curves.position(kind, name, times)-reference, axis=1)
            axes[1, 1].plot(times, error, style, color=colors[kind], label=name+' '+labels[kind])
    axes[1, 1].set(xlabel='Seconds from reference crossing', ylabel='Position error versus reference (m)', title='Position errors on this encounter')
    axes[1, 1].legend(fontsize=8)
    fig.suptitle('Constructed Swarm A/C encounter: correction versus no correction', fontsize=14)
    fig.savefig(output/'results.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), layout='constrained')
    horizon = np.linspace(.25, 2, 100)
    for row, name in enumerate(SATELLITES):
        model = summary['models'][name]
        mean, covariance = evaluate_model(model, horizon)
        sigma = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))
        bins = model['sample_bins_by_hours']
        for j, direction in enumerate(('R', 'T', 'N')):
            ax = axes[row, j]
            ax.plot(horizon*24, mean[:, j], label='Fitted mean')
            ax.plot(horizon*24, sigma[:, j], label='Fitted sigma')
            ax.scatter([float(k) for k in bins], [b['mean_rtn_m'][j] for b in bins.values()], marker='o', s=20)
            ax.scatter([float(k) for k in bins], [b['sample_sigma_rtn_m'][j] for b in bins.values()], marker='x', s=25)
            ax.set(title=name+' / '+direction, xlabel='TLE age (h)', ylabel='Metres')
    axes[0, 0].legend(fontsize=8)
    fig.savefig(output/'calibration.png', dpi=170)
    plt.close(fig)


def report(summary, output):
    truth = summary['ground_truth']
    scores = summary['outcome_scores']
    winner = min(scores, key=lambda name: scores[name]['brier_score'])
    lines = ['# Collision probability: forecasts versus a binary reference outcome', '',
             f"**Reference outcome: {truth['outcome']} (y={truth['collision_label']}). The {winner} forecast has the lower Brier score and log loss for this one encounter.**", '',
             'This implements Sections 3–6 of [the simple guide](../../docs/Collision_Avoidance_Simple_Guide.md). Only the corrected and uncorrected prediction models receive covariances and collision probabilities. Reference trajectories supply a deterministic collision/no-collision label.', '',
             'The input is the fixed time-shifted Swarm A/C case. Original GPS-based precise-orbit and SGP4 trajectories share synthetic timestamps while retaining native times and TLE ages. This is a constructed encounter, not an observed historical collision.', '',
             '## Ground truth from the reference trajectories', '',
             f"The continuous reference minimum separation is **{truth['closest_approach']['miss_distance_m']:.6f} m**. Treating the reference trajectories as exact, y = 1 if minimum separation <= R_A + R_C, otherwise y = 0.", '',
             f"The illustrative radius assumptions are A = {summary['radii_m']['Swarm-A']:g} m and C = {summary['radii_m']['Swarm-C']:g} m. Their sum is the **{summary['combined_radius_m']:g} m collision distance**; it is not the radius of each satellite. These values are configurable and are not measured spacecraft dimensions.", '',
             f"The reference outcome is therefore **{truth['outcome']}**, with label **{truth['collision_label']}**. No reference covariance or reference Pc is calculated.", '',
             f"All methods assess the same preselected crossing within ±{summary['encounter_half_window_seconds']:g} s of its saved center. The full two-hour trajectories contain three separate approaches; their minima are retained as separate geometry diagnostics.", '',
             '## Forecasts and outcome scores', '',
             '| Method | TCA offset (s) | Predicted miss (m) | Pc | Brier score | Log loss |',
             '|---|---:|---:|---:|---:|---:|']
    for name, result in summary['encounter_results'].items():
        score = scores[name]
        lines.append(f"| {name} | {result['closest_approach']['elapsed_seconds']:.6f} | {result['closest_approach']['miss_distance_m']:.6f} | {result['quadrature']['pc']:.9g} | {score['brier_score']:.10f} | {score['log_loss']:.9f} |")
    lines += ['', 'Brier score is (Pc - y)^2. Binary log loss is -y log(Pc) - (1-y) log(1-Pc). Lower is better for both. Because this encounter has y=1, a higher Pc produces a better score on this event. For an encounter with y=0, a lower Pc scores better.', '',
              'One selected collision case does not reveal the true forecast probability or establish that either method is generally calibrated. Validation needs many independently evaluated collision and non-collision encounters. The small improvement here also does not erase the fact that both forecasts assign very low probability to this selected collision.', '',
              'Each forecast has its own refined TCA and encounter plane. The methods share the same learned covariance function and radii; the projected matrices can differ because their TCA and plane differ. The reference geometry is not used to tune either correction or covariance.', '',
              'A larger Euclidean miss can still yield a larger Pc under anisotropic covariance: the full projected mean and covariance determine the integral.', '',
              '## Fitting and held-out checks', '',
              'Only Swarm A/C calibration forecasts are used: issue dates January 1–17, 2025 train the model; January 18–20 form the gap; January 21–31 are held out. Training reference targets precede test issue dates. Encounter reference positions never fit model parameters.', '',
              'States are converted to GCRS before subtraction. Training uses reference RTN errors and full K-1 sample covariance in each horizon bin. Marginal Gaussian likelihood fits quadratic means and standard deviations with positive sigma intercepts and nonnegative growth coefficients. Standardized training residuals supply a full correlation matrix, giving P_RTN = D Gamma D. Inference rotates these parameters with the predicted RTN basis.', '',
              'The correction adds U mu(tau) to SGP4. The uncorrected method keeps the original mean with the same centred covariance, making this a controlled mean-correction comparison. It is not a separately refitted zero-mean covariance baseline.', '',
              '| Satellite | Technique | Held-out rows | Position RMSE (m) | Gaussian NLL | 95% coverage |',
              '|---|---|---:|---:|---:|---:|']
    for name, methods in summary['held_out_scores'].items():
        for method, score in methods.items():
            lines.append(f"| {name} | {method} | {score['rows']} | {score['position_rmse_m']:.3f} | {score['mean_negative_log_likelihood']:.3f} | {100*score['coverage_95_percent']:.2f}% |")
    lines += ['', 'At the saved reference crossing center:', '',
              '| Satellite | Original position error (m) | Corrected position error (m) |', '|---|---:|---:|']
    for name, errors in summary['position_errors_at_reference_center_m'].items():
        lines.append(f"| {name} | {errors['uncorrected']:.3f} | {errors['corrected']:.3f} |")
    lines += ['', 'A better probability score on this event does not imply that individual position errors improve. Both position and probability results should be inspected.', '',
              '## Numerical checks and limits', '',
              '- Positive-definite covariances, polar Gauss–Legendre quadrature refinement and seeded Monte Carlo cross-check the two forecast probabilities. Monte Carlo draws are hypothetical prediction outcomes, not ground-truth labels.',
              '- Mean velocities are derivatives of continuous position curves, including the changing bias. The supplied SGP4 velocities construct the predicted RTN basis.',
              '- Short-encounter duration, projected curvature, direction change and covariance variation are recorded per method.',
              '- Independent object errors are assumed; cross-object covariance is unverified. Gaussian tails, reference error and the illustrative size assumptions remain limitations.',
              '- '+('Supplied maneuver intervals were screened against forecast intervals.' if summary['calibration_checks']['maneuver_log_supplied'] else 'No verified maneuver log was supplied. All residuals were retained without clipping; maneuver contamination remains unresolved.'),
              '- No avoidance maneuvers are implemented.', '',
              '## Outputs and rerun', '',
              '`summary.json` separates ground_truth, encounter_results and outcome_scores. It also records fitted models, covariance matrices, integration/Monte Carlo checks, held-out scores and source hashes. `prediction_errors.json` contains reference-basis errors. `encounter_trajectories.csv` stores both objects’ reference, original and corrected positions at shared timestamps.', '',
              '```sh', 'python experiments/collision_probability/run_experiment.py --data-dir /path/to/data',
              'python scripts/check_collision_results.py', '```', '',
              'The core downloader supplies every raw input. Missing shifted bundles are reconstructed from the checked-in recipe. Override radius assumptions with --radius-a-m and --radius-c-m. Optional maneuver exclusions use --maneuver-intervals and a CSV with satellite,start_utc,end_utc.', '',
              '![Forecast and outcome comparison](results.png)', '', '![Quadratic calibration](calibration.png)', '']
    (output/'report.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(os.environ.get('SGP4_DATA_DIR', ROOT/'data')))
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--radius-a-m', type=float, default=5.)
    parser.add_argument('--radius-c-m', type=float, default=5.)
    parser.add_argument('--mc-samples', type=int, default=1_000_000)
    parser.add_argument('--encounter-half-window-s', type=float, default=120.)
    parser.add_argument('--maneuver-intervals', type=Path)
    args = parser.parse_args()
    if not np.isfinite(args.radius_a_m+args.radius_c_m) or min(args.radius_a_m, args.radius_c_m) <= 0 or args.mc_samples < 1:
        parser.error('Positive finite radii and positive Monte Carlo sample count required')
    data_dir, output = args.data_dir.expanduser().resolve(), args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    bundle = ensure_encounter(data_dir)
    metadata, states = load_encounter(bundle)
    if not np.isfinite(args.encounter_half_window_s) or not 2 <= args.encounter_half_window_s <= 3600:
        parser.error('Encounter half-window must be between 2 and 3600 seconds')
    print('Computing reference-basis calibration errors for Swarm A and C.', flush=True)
    rows, checks = calibration_errors(data_dir, args.maneuver_intervals)
    models, scores = {}, {}
    for name in SATELLITES:
        train = [r for r in rows if r['satellite'] == name and r['split'] == 'train']
        test = [r for r in rows if r['satellite'] == name and r['split'] == 'test']
        models[name] = fit_model(np.array([float(r['horizon_hours'])/24 for r in train]),
                                 np.array([r['error_rtn_m'] for r in train]))
        scores[name] = score_model(models[name], np.array([float(r['horizon_hours'])/24 for r in test]),
                                   np.array([r['error_rtn_m'] for r in test]),
                                   np.array([r['predicted_to_reference_rtn_rotation'] for r in test]))
    curves = EncounterTrajectories(states, models)
    radius = args.radius_a_m+args.radius_c_m
    results = {}
    for label, kind in [('uncorrected', 'uncorrected'), ('corrected', 'corrected')]:
        results[label] = evaluate_encounter(curves, kind, metadata, radius, args.mc_samples, args.encounter_half_window_s)
        print(f"{label}: miss={results[label]['closest_approach']['miss_distance_m']:.6f} m; Pc={results[label]['quadrature']['pc']:.8g}", flush=True)
    truth = reference_outcome(curves, metadata, radius, args.encounter_half_window_s)
    outcome_scores = {name: binary_scores(result['quadrature']['pc'], truth['collision_label'],
                                         result['quadrature']['log_pc']) for name, result in results.items()}
    summary = {'method': 'Simple guide Sections 3–6; full quadratic bias, constrained sigma, full covariance and encounter-plane Pc',
               'radii_m': dict(zip(SATELLITES, (args.radius_a_m, args.radius_c_m))), 'combined_radius_m': radius,
               'radius_source': 'Illustrative configurable assumptions; not measured spacecraft dimensions',
               'input_bundle_id': metadata['dataset_id'], 'input_bundle_sha256': fingerprint(bundle/'encounter.json'),
               'input_trajectory_hashes': {k: metadata['outputs'][k]['sha256'] for k in ('reference', 'sgp4')},
               'split_counts': {split: sum(r['split'] == split for r in rows) for split in ('train', 'gap', 'test')},
               'encounter_half_window_seconds': args.encounter_half_window_s,
               'full_trajectory_minima': {label: closest_approach(lambda t, k=kind: curves.relative(k, t), curves.times)
                                          for label, kind in [('uncorrected', 'uncorrected'), ('corrected', 'corrected'), ('reference', 'reference')]},
               'models': models, 'held_out_scores': scores, 'calibration_checks': checks,
               'position_errors_at_reference_center_m': {
                   name: {kind: float(np.linalg.norm(curves.position(kind, name, 0)-curves.position('reference', name, 0)))
                          for kind in ('uncorrected', 'corrected')} for name in SATELLITES},
               'ground_truth': truth, 'encounter_results': results, 'outcome_scores': outcome_scores,
               'independent_object_errors_assumed': True, 'correction_covariance_shared_with_uncorrected': True,
               'single_event_evaluation_only': True,
               'software_versions': {package: importlib.metadata.version(package)
                                     for package in ('numpy', 'scipy', 'astropy', 'sgp4', 'pyerfa')}}
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    (output/'prediction_errors.json').write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n')
    write_trajectories(curves, metadata, output)
    plots(summary, curves, output)
    report(summary, output)
    print('Saved probability comparison, fitted models, trajectories, plots and report.', flush=True)


if __name__ == '__main__':
    main()
