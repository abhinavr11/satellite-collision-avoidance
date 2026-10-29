"""Load the standalone Swarm A/C encounter bundle; requires only NumPy."""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parent


def load_metadata():
    return json.loads((ROOT/'encounter.json').read_text())


def load_trajectories(kind='reference'):
    """Return {satellite: arrays} on the common synthetic timeline.

    kind is 'reference' or 'sgp4'. Position/velocity are N x 3 GCRS arrays
    in m and m/s. Original timestamps remain in source_time_utc.
    """
    if kind not in ('reference', 'sgp4'):
        raise ValueError("kind must be 'reference' or 'sgp4'")
    metadata = load_metadata()
    entry = metadata['outputs'][kind]
    path = ROOT/entry['path']
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
        raise ValueError(f'Trajectory checksum mismatch: {path}')
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != entry['rows']:
        raise ValueError('Unexpected trajectory row count')
    result = {}
    for obj in metadata['objects']:
        selected = [row for row in rows if row['satellite'] == obj['satellite']]
        result[obj['satellite']] = {
            'elapsed_seconds': np.array([float(row['elapsed_seconds']) for row in selected]),
            'time_utc': np.array([row['time_utc'] for row in selected]),
            'source_time_utc': np.array([row['source_time_utc'] for row in selected]),
            'tle_age_hours': np.array([float(row['tle_age_hours']) for row in selected]),
            'position_m': np.array([[float(row[k]) for k in ('x_m', 'y_m', 'z_m')] for row in selected]),
            'velocity_m_s': np.array([[float(row[k]) for k in ('vx_m_s', 'vy_m_s', 'vz_m_s')] for row in selected]),
        }
    return result


if __name__ == '__main__':
    for kind in ('reference', 'sgp4'):
        trajectories = load_trajectories(kind)
        a, c = trajectories['Swarm-A'], trajectories['Swarm-C']
        assert np.array_equal(a['time_utc'], c['time_utc'])
        i = int(np.flatnonzero(a['elapsed_seconds'] == 0)[0])
        distance = np.linalg.norm(a['position_m'][i]-c['position_m'][i])
        print(f'{kind}: {len(a["elapsed_seconds"])} samples per satellite; center separation {distance:.6f} m')
