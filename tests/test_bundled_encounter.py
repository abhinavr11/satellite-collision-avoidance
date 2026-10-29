"""Packaging checks: a clone must have independent, byte-verifiable inputs."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from collision_probability.data import BUNDLED_ENCOUNTER, ROOT, ensure_encounter
from scripts.check_bundled_encounter import validate_bundle


class BundledEncounterTests(unittest.TestCase):
    def test_complete_bundle_and_aligned_export(self):
        result = validate_bundle()
        self.assertEqual(result['samples_per_trajectory'], 7201)

    def test_default_works_without_external_source_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / 'does-not-exist'
            with patch('collision_probability.data.shutil.copy2', side_effect=AssertionError('Unexpected source-cache access')):
                self.assertEqual(ensure_encounter(missing), BUNDLED_ENCOUNTER)
            self.assertFalse(missing.exists())

    def test_notebook_is_portable_executable_python(self):
        notebook = json.loads((ROOT / 'notebooks/Shifted_Swarm_Demo.ipynb').read_text())
        self.assertEqual(notebook['nbformat'], 4)
        for index, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), f'cell-{index}', 'exec')
                self.assertFalse(any(output['output_type'] == 'error' for output in cell.get('outputs', [])))


if __name__ == '__main__':
    unittest.main()
