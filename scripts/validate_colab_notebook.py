"""Execute the demo against a clone of the staged tree; save successful outputs.

Requires numerical dependencies plus requirements-notebook-qa.txt. No push,
branch switch, or access to the original project's bulk data is performed.
"""
from pathlib import Path
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/Shifted_Swarm_Demo.ipynb'


def run(*args, cwd=ROOT):
    return subprocess.check_output(args, cwd=cwd, stderr=subprocess.STDOUT, text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', default=sys.executable, help='Python with numerical and notebook QA dependencies')
    args = parser.parse_args()
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code':
            compile(cell.source, f'notebook-cell-{index}', 'exec')
            cell.outputs = []
            cell.execution_count = None
    with tempfile.TemporaryDirectory(prefix='swarm-notebook-qa-') as temporary:
        base = Path(temporary)
        source = base / 'source'
        source.mkdir()
        run('git', 'checkout-index', '--all', '--prefix', str(source) + os.sep)
        run('git', 'init', '-b', 'main', cwd=source)
        run('git', 'add', '.', cwd=source)
        run('git', '-c', 'user.name=Notebook QA', '-c', 'user.email=notebook-qa@example.invalid',
            '-c', 'commit.gpgsign=false', 'commit', '-m', 'Staged notebook validation snapshot', cwd=source)
        kernels = base / 'kernels'
        spec = kernels / 'satellite-demo-qa'
        spec.mkdir(parents=True)
        (spec / 'kernel.json').write_text(json.dumps({
            'argv': [str(Path(args.python).resolve()), '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'Satellite demo validation', 'language': 'python'}))
        manager = KernelManager(kernel_name='satellite-demo-qa',
                                kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
        environment = dict(os.environ, SATELLITE_DEMO_REPO_URL=source.as_uri(),
                           SATELLITE_DEMO_WORKSPACE=str(base / 'workspace'), MPLBACKEND='Agg')
        client = NotebookClient(notebook, km=manager, timeout=600,
                                resources={'metadata': {'path': str(base)}})
        client.on_cell_start = lambda cell, cell_index, **_: print(
            f'Cell {cell_index + 1}/{len(notebook.cells)}: {cell.source.splitlines()[0][:100]}', flush=True)
        client.execute(env=environment)
        project = base / 'workspace/satellite-collision-avoidance'
        run(args.python, 'scripts/check_bundled_encounter.py', cwd=project)
        summary = json.loads((project / 'demo_outputs/notebook_results.json').read_text())
        for cell in notebook.cells:
            cell.metadata.pop('execution', None)
            for output in cell.get('outputs', []):
                if output.output_type == 'error':
                    raise RuntimeError('Notebook contains a failed cell')
                fragments = [''.join(output.get('text', []))]
                fragments += [''.join(value) for key, value in output.get('data', {}).items() if key.startswith('text/')]
                if re.search(r'/(?:Users|home|private/var|var/folders)/', '\n'.join(fragments)):
                    raise RuntimeError('Notebook output contains a machine-specific path')
        nbformat.validate(notebook)
        nbformat.write(notebook, NOTEBOOK)
        print(json.dumps({'validation': 'all cells executed against isolated staged Git clone',
                          'hosted_colab_executed': False,
                          'dataset_id': summary['dataset_id'],
                          'collision_probabilities': {name: value['quadrature']['pc']
                              for name, value in summary['collision_results'].items()},
                          'position_results': summary['position_results']}, indent=2))


if __name__ == '__main__':
    main()
