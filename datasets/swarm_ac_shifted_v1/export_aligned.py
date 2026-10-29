"""Save all four trajectories together, with exactly one row per shared time."""
from pathlib import Path
import csv
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parent
STATE_FIELDS = ('x_m', 'y_m', 'z_m', 'vx_m_s', 'vy_m_s', 'vz_m_s')


def export_aligned(root=ROOT):
    root = Path(root)
    metadata = json.loads((root/'encounter.json').read_text())
    tables = {}
    for kind in ('reference', 'sgp4'):
        entry = metadata['outputs'][kind]
        path = root/entry['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError(f'Checksum mismatch: {path}')
        with path.open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        table = {(row['time_utc'], row['satellite']): row for row in rows}
        if len(table) != len(rows) or len(rows) != entry['rows']:
            raise ValueError(f'Duplicate or missing rows in {kind}')
        tables[kind] = table
    if tables['reference'].keys() != tables['sgp4'].keys():
        raise ValueError('Reference and SGP4 timestamps or satellite identities differ')
    names = ('Swarm-A', 'Swarm-C')
    times = sorted({time for time, _ in tables['reference']})
    expected = metadata['outputs']['reference']['rows_per_satellite']
    if len(times) != expected or len(tables['reference']) != 2*expected:
        raise ValueError('Expected two complete satellite trajectories')
    columns = ['time_utc', 'elapsed_seconds']
    for name in names:
        prefix = name.lower().replace('-', '_')
        columns.extend([prefix+'_source_time_utc', prefix+'_tle_age_hours'])
        for kind in ('reference', 'sgp4'):
            columns.extend(prefix+'_'+kind+'_'+field for field in STATE_FIELDS)
    output = root/'aligned_trajectories.csv'
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for time in times:
            paired = {'time_utc': time}
            for name in names:
                prefix = name.lower().replace('-', '_')
                ref, pred = (tables[kind][time, name] for kind in ('reference', 'sgp4'))
                for field in ('source_time_utc', 'elapsed_seconds', 'norad_id', 'tle_age_hours'):
                    if ref[field] != pred[field]:
                        raise ValueError(f'{name} reference/SGP4 mismatch: {field}')
                if 'elapsed_seconds' in paired and paired['elapsed_seconds'] != ref['elapsed_seconds']:
                    raise ValueError('Satellite elapsed times differ')
                paired['elapsed_seconds'] = ref['elapsed_seconds']
                paired[prefix+'_source_time_utc'] = ref['source_time_utc']
                paired[prefix+'_tle_age_hours'] = ref['tle_age_hours']
                for kind, row in (('reference', ref), ('sgp4', pred)):
                    for field in STATE_FIELDS:
                        if not math.isfinite(float(row[field])):
                            raise ValueError('Nonfinite trajectory state')
                        paired[prefix+'_'+kind+'_'+field] = row[field]
            writer.writerow(paired)
    metadata['schema_version'] = '1.1'
    metadata['outputs']['aligned'] = {
        'path': output.name, 'rows': len(times), 'columns': columns,
        'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'description': 'One row per shared synthetic UTC; GPS-based precise reference and SGP4 states for both satellites.',
        'reference_source_type': 'ESA processed GPS-based precise orbit (SP3), interpolated to one-second spacing',
        'all_four_trajectories_share_exact_timestamps': True,
        'state_values_identical_to_separate_tables': True,
    }
    (root/'encounter.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Saved {output.name}: {len(times)} timestamps, all four trajectories aligned.')


if __name__ == '__main__':
    export_aligned()
