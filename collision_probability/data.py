"""Reference-RTN calibration data and reproducible shifted encounter inputs."""
from pathlib import Path
from datetime import datetime, timedelta
import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys

import numpy as np
from astropy import units as u
from astropy.coordinates import ITRS, TEME, GCRS, CartesianRepresentation, CartesianDifferential
from astropy.time import Time
from astropy.utils import iers
from scipy.interpolate import CubicSpline

from .model import rtn_basis, evaluate_model

ROOT = Path(__file__).resolve().parents[1]
SATELLITES = ('Swarm-A', 'Swarm-C')


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ensure_encounter(data_dir, rebuild=False):
    """Reconstruct from the ordinary core download; never fetch extra data."""
    data_dir = Path(data_dir)
    bundle = data_dir/'encounters/swarm_ac_shifted_v1'
    if (bundle/'encounter.json').exists() and not rebuild:
        load_encounter(bundle)
        return bundle
    recipe = json.loads((ROOT/'config/swarm_ac_shifted_encounter.json').read_text())
    (bundle/'sources').mkdir(parents=True, exist_ok=True)
    for entry in [recipe['earth_orientation']]+[o['reference_sp3'] for o in recipe['objects']]:
        source = data_dir/entry['original_data_path']
        if 'reference_orbits/' in entry['original_data_path'] and fingerprint(source) != entry['sha256']:
            raise ValueError(f'Reference source checksum mismatch: {source.name}')
        shutil.copy2(source, bundle/entry['path'])
        entry['sha256'] = fingerprint(source)
    for obj in recipe['objects']:
        entry = obj['tle']
        source = data_dir/entry['original_data_path']
        lines = source.read_text().splitlines()
        i = 2*entry['original_pair_index_zero_based']
        target = bundle/entry['path']
        target.write_text('\n'.join(lines[i:i+2])+'\n')
        if fingerprint(target) != entry['sha256']:
            raise ValueError(f'Selected TLE checksum mismatch: {target.name}')
    for source in (ROOT/'scripts/encounter_bundle').iterdir():
        if source.is_file():
            shutil.copy2(source, bundle/source.name)
    (bundle/'encounter.json').write_text(json.dumps(recipe, indent=2)+'\n')
    subprocess.run([sys.executable, str(bundle/'rebuild.py')], check=True)
    subprocess.run([sys.executable, str(bundle/'plot_trajectories.py')], check=True)
    return bundle


def load_encounter(bundle):
    bundle = Path(bundle)
    metadata = json.loads((bundle/'encounter.json').read_text())
    if metadata['coordinate_frame'] != 'GCRS' or metadata['time_scale'] != 'UTC':
        raise ValueError('Expected GCRS positions on a common UTC timeline')
    result = {}
    for kind in ('reference', 'sgp4'):
        entry = metadata['outputs'][kind]
        source = bundle/entry['path']
        if fingerprint(source) != entry['sha256']:
            raise ValueError(f'Encounter table checksum mismatch: {kind}')
        with source.open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != entry['rows']:
            raise ValueError('Unexpected encounter row count')
        for obj in metadata['objects']:
            name = obj['satellite']
            if name not in SATELLITES:
                raise ValueError('Encounter must contain only Swarm A and C')
            selected = [r for r in rows if r['satellite'] == name]
            t = np.array([float(r['elapsed_seconds']) for r in selected])
            position = np.array([[float(r[k]) for k in ('x_m', 'y_m', 'z_m')] for r in selected])
            velocity = np.array([[float(r[k]) for k in ('vx_m_s', 'vy_m_s', 'vz_m_s')] for r in selected])
            if len(t) != entry['rows_per_satellite'] or np.any(np.diff(t) <= 0):
                raise ValueError('Encounter samples must be complete and increasing')
            if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
                raise ValueError('Nonfinite encounter states')
            for r in selected:
                native = datetime.fromisoformat(r['source_time_utc'].replace('Z', '+00:00'))
                shifted = datetime.fromisoformat(r['time_utc'].replace('Z', '+00:00'))
                if native+timedelta(seconds=obj['source_to_synthetic_shift_seconds']) != shifted:
                    raise ValueError('Incorrect native-to-synthetic time mapping')
            result[kind, name] = {'t': t, 'position': position, 'velocity': velocity,
                                  'age_days': np.array([float(r['tle_age_hours'])/24 for r in selected]),
                                  'time_utc': [r['time_utc'] for r in selected],
                                  'source_time_utc': [r['source_time_utc'] for r in selected]}
    anchor = result['reference', 'Swarm-A']
    for state in result.values():
        if not np.array_equal(state['t'], anchor['t']) or state['time_utc'] != anchor['time_utc']:
            raise ValueError('All four trajectories must share identical timestamps')
    for name in SATELLITES:
        if result['reference', name]['source_time_utc'] != result['sgp4', name]['source_time_utc']:
            raise ValueError('Prediction and reference source times differ')
    return metadata, result


def calibration_errors(data_dir, maneuver_intervals=None):
    """Recompute reference-basis GCRS errors; never reuse predicted-basis errors."""
    data_dir = Path(data_dir)
    spec = importlib.util.spec_from_file_location('original_pilot', ROOT/'experiments/sgp4_uncertainty_pilot/run_experiment.py')
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    pilot.DATA = data_dir
    with (data_dir/'calibration_pairs_index.csv').open(newline='') as handle:
        rows = [r for r in csv.DictReader(handle) if r['satellite'] in SATELLITES]
    excluded = 0
    if maneuver_intervals:
        with Path(maneuver_intervals).open(newline='') as handle:
            intervals = list(csv.DictReader(handle))
        kept = []
        for row in rows:
            overlap = any(row['satellite'] == interval['satellite'] and
                          pilot.parse_utc(interval['start_utc']) <= pilot.parse_utc(row['comparison_time_utc']) and
                          pilot.parse_utc(interval['end_utc']) >= pilot.parse_utc(row['tle_epoch_utc'])
                          for interval in intervals)
            excluded += int(overlap)
            if not overlap:
                kept.append(row)
        rows = kept
    times = Time([r['comparison_time_utc'] for r in rows], scale='utc')
    reference_r, reference_v, predicted_r, predicted_v, checks = [], [], [], [], []
    for i, row in enumerate(rows):
        sat = pilot.read_tles(row['tle_file'])[int(row['tle_pair_index_zero_based'])]
        if sat.satnum != int(row['norad_cat_id']):
            raise ValueError('Calibration satellite identity mismatch')
        code, r, v = sat.sgp4(times[i].jd1, times[i].jd2)
        if code:
            raise RuntimeError(f'SGP4 error {code}')
        rr, vv, check, frame = pilot.reference_at(row['reference_file'], float(times[i].unix), pilot.REFERENCE_MAP[row['satellite']])
        reference_r.append(rr)
        reference_v.append(vv)
        predicted_r.append(np.asarray(r)*1000)
        predicted_v.append(np.asarray(v)*1000)
        checks.append(check)
        row['reference_frame'] = frame
    table = iers.IERS_A.open(str(data_dir/'earth_orientation/finals2000A.all'))
    with iers.earth_orientation_table.set(table):
        def transform(r, v, frame):
            state = CartesianRepresentation(np.asarray(r).T*u.m,
                                            differentials=CartesianDifferential(np.asarray(v).T*u.m/u.s))
            original = frame(state, obstime=times)
            converted = original.transform_to(GCRS(obstime=times))
            back = converted.transform_to(frame(obstime=times))
            error = float(np.max(np.linalg.norm(back.cartesian.xyz.to_value(u.m).T-r, axis=1)))
            if error > 1e-4:
                raise RuntimeError('Calibration frame round trip failed')
            return (converted.cartesian.xyz.to_value(u.m).T,
                    converted.cartesian.differentials['s'].d_xyz.to_value(u.m/u.s).T, error)
        rr, vv, ref_roundtrip = transform(reference_r, reference_v, ITRS)
        pr, pv, pred_roundtrip = transform(predicted_r, predicted_v, TEME)
    basis = rtn_basis(rr, vv)
    predicted_basis = rtn_basis(pr, pv)
    predicted_to_reference = np.einsum('nji,njk->nik', basis, predicted_basis)
    errors = np.einsum('nji,nj->ni', basis, rr-pr)
    np.testing.assert_allclose(np.linalg.norm(errors, axis=1), np.linalg.norm(rr-pr, axis=1), atol=1e-7)
    for row, error, rotation in zip(rows, errors, predicted_to_reference):
        issue = pilot.parse_utc(row['tle_epoch_utc'])
        row['split'] = 'train' if issue < pilot.TRAIN_END else 'test' if issue >= pilot.TEST_START else 'gap'
        row['error_rtn_m'] = error.tolist()
        row['predicted_to_reference_rtn_rotation'] = rotation.tolist()
    latest = max(pilot.parse_utc(r['comparison_time_utc']) for r in rows if r['split'] == 'train')
    earliest = min(pilot.parse_utc(r['tle_epoch_utc']) for r in rows if r['split'] == 'test')
    if latest >= earliest:
        raise ValueError('Training target times overlap test issue times')
    sources = {r[k] for r in rows for k in ('tle_file', 'reference_file')}
    sources.update(['calibration_pairs_index.csv', 'earth_orientation/finals2000A.all'])
    return rows, {'reference_rtn_basis': True, 'coordinate_frame': 'GCRS',
                  'max_interpolation_crosscheck_m': max(checks),
                  'reference_frame_roundtrip_m': ref_roundtrip, 'sgp4_frame_roundtrip_m': pred_roundtrip,
                  'latest_training_target_utc': latest.isoformat(), 'earliest_test_issue_utc': earliest.isoformat(),
                  'excluded_maneuver_forecasts': excluded, 'maneuver_log_supplied': bool(maneuver_intervals),
                  'source_hashes': {p: fingerprint(data_dir/p) for p in sorted(sources)}}


class EncounterTrajectories:
    """Continuous mean positions; velocities are derivatives of these positions."""
    def __init__(self, states, models):
        self.states, self.models = states, models
        self.times = states['sgp4', 'Swarm-A']['t']
        self.position_curves, self.velocity_curves, self.ages = {}, {}, {}
        for name in SATELLITES:
            pred = states['sgp4', name]
            self.ages[name] = CubicSpline(self.times, pred['age_days'], extrapolate=False)
            self.velocity_curves[name] = CubicSpline(self.times, pred['velocity'], extrapolate=False)
            correction, _ = evaluate_model(models[name], pred['age_days'])
            basis = rtn_basis(pred['position'], pred['velocity'])
            corrected = pred['position']+np.einsum('nij,nj->ni', basis, correction)
            for kind, positions in [('reference', states['reference', name]['position']),
                                    ('uncorrected', pred['position']), ('corrected', corrected)]:
                self.position_curves[kind, name] = CubicSpline(self.times, positions, extrapolate=False)

    def position(self, kind, name, t):
        return self.position_curves[kind, name](t)

    def velocity(self, kind, name, t):
        return self.position_curves[kind, name](t, 1)

    def relative(self, kind, t):
        return self.position(kind, 'Swarm-A', t)-self.position(kind, 'Swarm-C', t)

    def relative_velocity(self, kind, t):
        return self.velocity(kind, 'Swarm-A', t)-self.velocity(kind, 'Swarm-C', t)

    def covariance(self, name, t, kind=None):
        # This function is shared by both techniques; reference geometry never fits it.
        _, local = evaluate_model(self.models[name], self.ages[name](t))
        basis = rtn_basis(self.position('uncorrected', name, t), self.velocity_curves[name](t))
        return basis@local@basis.T
