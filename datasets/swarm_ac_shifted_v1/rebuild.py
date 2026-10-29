"""Rebuild the two trajectory tables offline from the bundled source files."""
from pathlib import Path
import csv
from datetime import datetime, timedelta
import hashlib
import importlib.metadata
import json

import numpy as np
from scipy.interpolate import CubicHermiteSpline, BarycentricInterpolator
from astropy import units as u
from astropy.coordinates import CartesianRepresentation, CartesianDifferential, ITRS, TEME, GCRS
from astropy.time import Time
from astropy.utils import iers
from sgp4.api import Satrec, WGS72

ROOT = Path(__file__).resolve().parent
FIELDS = ['satellite', 'norad_id', 'elapsed_seconds', 'time_utc', 'source_time_utc',
          'tle_age_hours', 'x_m', 'y_m', 'z_m', 'vx_m_s', 'vy_m_s', 'vz_m_s']


def parse(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def stamp(value):
    return value.isoformat(timespec='microseconds').replace('+00:00', 'Z')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_reference(path, expected_id, center):
    """Use seconds from the source center to avoid large absolute-time rounding."""
    epochs, positions, velocities, ids = [], [], [], set()
    with path.open() as handle:
        header = next(handle)
        frame = header[46:51].strip()
        assert frame in {'IGS20', 'IGb20'}
        gps_confirmed = False
        for line in handle:
            if line.startswith('%c M'):
                assert line.split()[3] == 'GPS'
                gps_confirmed = True
            elif line.startswith('* '):
                parts = line[1:].split()
                gps = datetime(*map(int, parts[:5]), tzinfo=center.tzinfo)
                gps += timedelta(seconds=float(parts[5]))
                epochs.append((gps - timedelta(seconds=18) - center).total_seconds())
            elif line.startswith('P'):
                ids.add(line[1:4])
                positions.append([float(line[k:k+14])*1000 for k in (4, 18, 32)])
            elif line.startswith('V'):
                velocities.append([float(line[k:k+14])*0.1 for k in (4, 18, 32)])
    assert gps_confirmed and ids == {expected_id}
    epochs, positions, velocities = map(np.asarray, (epochs, positions, velocities))
    assert len(epochs) == len(positions) == len(velocities) == 8640
    np.testing.assert_allclose(np.diff(epochs), 10, rtol=0, atol=1e-8)
    return epochs, positions, velocities, frame


def gcrs_states(position, velocity, times, source_frame):
    representation = CartesianRepresentation(
        position.T*u.m, differentials=CartesianDifferential(velocity.T*u.m/u.s))
    source = source_frame(representation, obstime=times)
    target = source.transform_to(GCRS(obstime=times))
    r = target.cartesian.xyz.to_value(u.m).T
    v = target.cartesian.differentials['s'].d_xyz.to_value(u.m/u.s).T
    back = target.transform_to(source_frame(obstime=times))
    roundtrip = float(np.max(np.linalg.norm(back.cartesian.xyz.to_value(u.m).T-position, axis=1)))
    assert roundtrip < 1e-4, roundtrip
    return r, v, roundtrip


def main():
    metadata = json.loads((ROOT/'encounter.json').read_text())
    iers.conf.auto_download = False
    iers.conf.auto_max_age = None
    window = metadata['window']
    elapsed = np.arange(window['start_elapsed_seconds'],
                        window['stop_elapsed_seconds']+1, window['step_seconds'], dtype=float)
    center = parse(metadata['synthetic_center_utc'])
    synthetic_dates = [stamp(center+timedelta(seconds=float(dt))) for dt in elapsed]
    tables = {'reference': [], 'sgp4': []}
    state_data, checks = {}, {}
    inputs = [metadata['earth_orientation']]
    inputs += [obj[key] for obj in metadata['objects'] for key in ('reference_sp3', 'tle')]
    for entry in inputs:
        assert digest(ROOT/entry['path']) == entry['sha256'], entry['path']
    table = iers.IERS_A.open(str(ROOT/metadata['earth_orientation']['path']))
    with iers.earth_orientation_table.set(table):
        for obj in metadata['objects']:
            name = obj['satellite']
            native_center = parse(obj['source_center_utc'])
            source_dates = [native_center+timedelta(seconds=float(dt)) for dt in elapsed]
            times = Time(source_dates, scale='utc')
            t, r, v, frame = read_reference(ROOT/obj['reference_sp3']['path'], obj['sp3_id'], native_center)
            assert t[0] <= elapsed[0] and elapsed[-1] <= t[-1]
            curve = CubicHermiteSpline(t, r, v, extrapolate=False)
            ref_r, ref_v = curve(elapsed), curve(elapsed, 1)
            interpolation_errors = []
            for dt in elapsed[::60]:
                k = int(np.searchsorted(t, dt)-1)
                lo = max(0, min(k-3, len(t)-8))
                polynomial = BarycentricInterpolator(t[lo:lo+8]-dt, r[lo:lo+8], axis=0)
                interpolation_errors.append(float(np.linalg.norm(polynomial(0)-curve(dt))))
            assert max(interpolation_errors) < 0.01
            lines = (ROOT/obj['tle']['path']).read_text().splitlines()
            assert len(lines) == 2
            sat = Satrec.twoline2rv(*lines, WGS72)
            assert sat.satnum == obj['norad_id']
            epoch = Time(sat.jdsatepoch, sat.jdsatepochF, format='jd', scale='utc')
            ages = (times-epoch).to_value(u.hour)
            codes, pred_r, pred_v = sat.sgp4_array(times.jd1, times.jd2)
            assert not np.any(codes), np.unique(codes)
            obj['tle']['source_epoch_utc'] = stamp(epoch.to_datetime(timezone=native_center.tzinfo))
            obj['tle']['synthetic_epoch_utc_for_age_bookkeeping_only'] = stamp(
                epoch.to_datetime(timezone=native_center.tzinfo)+timedelta(seconds=obj['source_to_synthetic_shift_seconds']))
            obj['tle']['age_at_center_hours'] = float(ages[np.flatnonzero(elapsed == 0)[0]])
            checks[name] = {'source_frame': frame,
                            'max_interpolation_crosscheck_m': max(interpolation_errors)}
            for kind, source_r, source_v, source_frame in (
                    ('reference', ref_r, ref_v, ITRS),
                    ('sgp4', pred_r*1000, pred_v*1000, TEME)):
                rr, vv, error = gcrs_states(source_r, source_v, times, source_frame)
                checks[name][kind+'_frame_roundtrip_m'] = error
                step = window['step_seconds']
                finite_difference = (rr[:-4]-8*rr[1:-3]+8*rr[3:-1]-rr[4:])/(12*step)
                speed_error = float(np.max(np.linalg.norm(finite_difference-vv[2:-2], axis=1)))
                native_difference = (source_r[:-4]-8*source_r[1:-3]+8*source_r[3:-1]-source_r[4:])/(12*step)
                native_error = float(np.max(np.linalg.norm(native_difference-source_v[2:-2], axis=1)))
                # SGP4 returns its own velocity approximation. Preserve it and
                # check that frame conversion adds no material inconsistency.
                assert speed_error < max(0.02, native_error+0.005), (kind, name, speed_error, native_error)
                checks[name][kind+'_native_velocity_finite_difference_max_m_s'] = native_error
                checks[name][kind+'_velocity_finite_difference_max_m_s'] = speed_error
                state_data[kind, name] = (rr, vv)
                for i, dt in enumerate(elapsed):
                    tables[kind].append([name, obj['norad_id'], int(dt), synthetic_dates[i],
                                         stamp(source_dates[i]), f'{ages[i]:.12f}',
                                         *[f'{x:.9f}' for x in rr[i]], *[f'{x:.9f}' for x in vv[i]]])
    center_index = int(np.flatnonzero(elapsed == 0)[0])
    metrics = {}
    for kind in tables:
        tables[kind].sort(key=lambda row: (row[2], row[0]))
        path = ROOT/(kind+'_trajectories.csv')
        with path.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(FIELDS)
            writer.writerows(tables[kind])
        a, av = state_data[kind, 'Swarm-A']
        c, cv = state_data[kind, 'Swarm-C']
        distances = np.linalg.norm(a-c, axis=1)
        metrics[kind] = {'center_separation_m': float(distances[center_index]),
                          'center_relative_speed_m_s': float(np.linalg.norm(av[center_index]-cv[center_index])),
                          'sampled_minimum_separation_m': float(distances.min()),
                          'sampled_minimum_elapsed_seconds': float(elapsed[distances.argmin()])}
    assert abs(metrics['reference']['center_separation_m']-
               metadata['reference_center_check_from_feasibility']['separation_m']) < 0.01
    metadata['outputs'] = {kind: {'path': kind+'_trajectories.csv',
                                 'rows': len(tables[kind]), 'rows_per_satellite': len(elapsed),
                                 'sha256': digest(ROOT/(kind+'_trajectories.csv'))} for kind in tables}
    metadata['metrics'] = metrics
    metadata['validation'] = checks
    metadata['software_versions'] = {package: importlib.metadata.version(package)
                                      for package in ('numpy', 'scipy', 'astropy', 'sgp4', 'pyerfa')}
    (ROOT/'encounter.json').write_text(json.dumps(metadata, indent=2)+'\n')
    from export_aligned import export_aligned
    export_aligned(ROOT)
    print(json.dumps({'outputs': metadata['outputs'], 'metrics': metrics, 'checks': checks}, indent=2))


if __name__ == '__main__':
    main()
