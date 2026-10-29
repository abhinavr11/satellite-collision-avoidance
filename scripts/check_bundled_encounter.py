"""Verify the versioned encounter inputs, source hashes and aligned export."""
from pathlib import Path
import csv
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from collision_probability.data import BUNDLED_ENCOUNTER, SATELLITES, fingerprint, load_encounter


def validate_bundle(bundle=BUNDLED_ENCOUNTER):
    bundle = Path(bundle)
    metadata, states = load_encounter(bundle)
    sources = [metadata['earth_orientation']]
    sources += [obj[key] for obj in metadata['objects'] for key in ('reference_sp3', 'tle')]
    for entry in sources + list(metadata['outputs'].values()):
        if fingerprint(bundle / entry['path']) != entry['sha256']:
            raise ValueError(f"Checksum mismatch: {entry['path']}")
    anchor = states['reference', SATELLITES[0]]
    window = metadata['window']
    expected = np.arange(window['start_elapsed_seconds'],
                         window['stop_elapsed_seconds'] + window['step_seconds'],
                         window['step_seconds'])
    np.testing.assert_array_equal(anchor['t'], expected)
    with (bundle / metadata['outputs']['aligned']['path']).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(expected):
        raise ValueError('Aligned table has a different number of timestamps')
    if [row['time_utc'] for row in rows] != anchor['time_utc']:
        raise ValueError('Aligned table has different common UTC timestamps')
    np.testing.assert_array_equal([float(row['elapsed_seconds']) for row in rows], expected)
    for name, prefix in zip(SATELLITES, ('swarm_a', 'swarm_c')):
        for kind in ('reference', 'sgp4'):
            state = states[kind, name]
            if [row[prefix + '_source_time_utc'] for row in rows] != state['source_time_utc']:
                raise ValueError('Aligned table lost native timestamps')
            np.testing.assert_allclose([float(row[prefix + '_tle_age_hours']) / 24 for row in rows],
                                       state['age_days'], rtol=0, atol=0)
            for array, fields in [('position', ('x_m', 'y_m', 'z_m')),
                                  ('velocity', ('vx_m_s', 'vy_m_s', 'vz_m_s'))]:
                aligned = [[float(row[f'{prefix}_{kind}_{field}']) for field in fields] for row in rows]
                np.testing.assert_array_equal(aligned, state[array])
    return {'dataset_id': metadata['dataset_id'], 'source_hashes_verified': len(sources),
            'trajectory_tables_verified': 3, 'samples_per_trajectory': len(expected),
            'common_timestamps': True, 'aligned_values_match': True}


if __name__ == '__main__':
    print(json.dumps(validate_bundle(), indent=2))
