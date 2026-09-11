"""Check provenance-preserving migration without requiring LAMMPS."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import run_campaign30 as c


class UpgradeTests(unittest.TestCase):
    def setup_case(self, root):
        (root/'source_snapshot').mkdir()
        (root/'checkpoints').mkdir()
        old = {}
        for name in ['run_campaign30.py', 'in.generate_equal_porosity_psd.lammps', 'solver.py']:
            p = root/'source_snapshot'/name
            p.write_text('original '+name)
            old[name] = c.digest(p)
        (root/'result.dat').write_text('accepted packing')
        c.write_json(root/'checkpoints'/'dem_18427.json',
                     {'outputs':{'result.dat':c.digest(root/'result.dat')}})
        previous = {'source_sha256':old, 'settings':{'max_ke_J':1e-12}, 'python':'same'}
        current = copy.deepcopy(previous)
        current['source_sha256'].update({k:'new' for k in list(old)[:2]})
        current['settings']['relaxation_extension'] = {'block_steps':50000}
        c.write_json(root/'campaign_manifest.json', previous)
        return previous, current, {k:old[k] for k in list(old)[:2]}

    def test_preserves_outputs_and_original_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            previous, current, hashes = self.setup_case(root)
            before = (root/'checkpoints'/'dem_18427.json').read_bytes()
            with patch.object(c, 'RELAX_PREDECESSOR', hashes):
                c.upgrade_relaxation(root, previous, current)
            self.assertEqual((root/'checkpoints'/'dem_18427.json').read_bytes(), before)
            self.assertEqual(json.loads((root/'campaign_manifest.json').read_text()), current)
            archive = root/'provenance_before_relaxation_extension'
            self.assertEqual(json.loads((archive/'campaign_manifest.json').read_text()), previous)
            self.assertEqual((root/'result.dat').read_text(), 'accepted packing')

    def test_rejects_changed_environment_or_output(self):
        for kind in ['environment', 'output', 'source']:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                previous, current, hashes = self.setup_case(root)
                if kind == 'environment':
                    current['python'] = 'changed'
                elif kind == 'output':
                    (root/'result.dat').write_text('tampered')
                else:
                    (root/'source_snapshot'/'solver.py').write_text('tampered')
                with patch.object(c, 'RELAX_PREDECESSOR', hashes):
                    with self.assertRaises(RuntimeError):
                        c.upgrade_relaxation(root, previous, current)
                self.assertEqual(json.loads((root/'campaign_manifest.json').read_text()), previous)


if __name__ == '__main__':
    unittest.main()
