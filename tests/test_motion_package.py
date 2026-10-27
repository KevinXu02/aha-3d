"""A fresh source checkout must prepare executable motion code without PromptHMR."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MotionPackageTests(unittest.TestCase):
    def command(self, *args, cwd=ROOT):
        result = subprocess.run([sys.executable, *map(str, args)], cwd=cwd,
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result

    def test_both_prepare_clis_default_to_packaged_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / 'bodies'
            for rel in ('motion_native.npz', 'provenance.json', 'hmr4d_results.pt',
                        'preprocess/bbx.pt', 'preprocess/vitpose.pt'):
                p = run / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text('fixture')
            fixture = root / 'input'
            fixture.write_text('fixture')
            fixture.with_name('input.json').write_text('{}')
            out = root / 'postopt'
            self.command(ROOT / 'tools/gvhmr/world_postopt.py', 'prepare',
                         '--gvhmr-run', run, '--camera-tracks', fixture,
                         '--smplx-model', fixture, '--python', sys.executable,
                         '--out', out, '--actor-id', 'actor', '--gpu', 0)
            self.assertEqual([s['id'] for s in json.loads((out / 'experiment.json').read_text())['stages']],
                             ['adapt', 'v2', 'measure'])
            self.assertTrue((out / 'upstream/pipeline/postprocessing_v2.py').is_file())
            self.assertIn('Selected PyTorch3D v0.4.0',
                          (out / 'upstream/prompt_hmr/utils/rotation_conversions.py').read_text())

            bundle = root / 'bundle'
            bundle.mkdir()
            for name in ('inputs.json', 'cameras.json', 'camera_review.json'):
                (bundle / name).write_text('{}')
            repo = root / 'gvhmr'
            for rel in ('inputs/checkpoints/gvhmr/gvhmr_b1b2.ckpt', 'tools/demo/demo.py'):
                p = repo / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text('fixture')
            sam3 = root / 'sam3'
            (sam3 / 'sam3').mkdir(parents=True)
            (sam3 / 'sam3/__init__.py').write_text('')
            # A source-bound scene prior is required by the public full-chain CLI.
            import hashlib
            prior = root / 'ground.json'
            prior.write_text(json.dumps(dict(source_video_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
                coordinate_frame='pi3x-aligned-world', scale=1, accepted=True,
                evidence='test-only prior', plane=[0, 0, 1, 0], upright=[0, 0, 1])))
            full = root / 'full'
            prepare_args = [ROOT / 'tools/gvhmr/world_pipeline.py', 'prepare',
                         '--video', fixture, '--pi3x-bundle', bundle,
                         '--lifecycle-review', fixture, '--sam3', sam3,
                         '--sam3-python', sys.executable, '--sam3-checkpoint', fixture,
                         '--gvhmr-repo', repo,
                         '--pi3', root, '--pi3-checkpoint', fixture,
                         '--smplx-model', fixture, '--python', sys.executable,
                         '--pmpose-root', root, '--pmpose-checkpoint', fixture,
                         '--out', full, '--actor-id', 'actor', '--gpu', 0,
                         '--box', 1, 2, 3, 4, '--scene-ground', prior]
            self.command(*prepare_args)
            graph = json.loads((full / 'experiment.json').read_text())
            self.assertEqual([s['id'] for s in graph['stages']],
                             ['track', 'lifecycle', 'bodies', 'camera', 'adapt', 'v2', 'measure', 'review'])
            provenance = json.loads((full / 'upstream_provenance.json').read_text())
            self.assertTrue(all(Path(p['path']).is_relative_to(ROOT / 'tools/gvhmr/world_backend')
                                for p in provenance.values()))
            self.assertTrue((full / 'snapshot/tools/gvhmr/person_reconstruct.py').is_file())
            self.assertTrue((full / 'snapshot/tools/gvhmr/person_tracking.py').is_file())
            self.assertFalse((full / 'upstream/experiments/postopt_ab/run_samurai.py').exists())
            self.assertFalse((full / 'snapshot/tools/gvhmr/samurai_reconstruct.py').exists())
            for removed in ('--tracker', '--samurai', '--samurai-checkpoint', '--samurai-deps'):
                result = subprocess.run([sys.executable, *map(str, prepare_args), removed, 'samurai'],
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('unrecognized arguments: '+removed, result.stderr)

    @unittest.skipUnless(importlib.util.find_spec('pytorch3d'), 'requires the GVHMR runtime')
    def test_backend_snapshot_imports_without_neighboring_checkout(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            backend = Path(tmp) / 'backend'
            shutil.copytree(ROOT / 'tools/gvhmr/world_backend', backend)
            code = '''
import importlib.util, pathlib, sys, torch
root = pathlib.Path(sys.argv[1])
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, root / path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
harness = load('harness', 'experiments/postopt_ab/run_ab.py')
harness._stub_prompt_hmr()
optimizer = load('optimizer', 'pipeline/postprocessing_v2.py')
adapter = load('adapter', 'experiments/postopt_ab/build_results.py')
load('baseline', 'experiments/postopt_ab/build_gvhmr_global.py')
metrics = load('metrics', 'experiments/postopt_ab/metrics.py')
load('export', 'experiments/postopt_ab/export_viewer_data.py')
assert callable(optimizer.post_optimization_v2)
aa = torch.tensor([[0., 0., 0.], [0.3, -0.8, 0.6], [3.14159265, 0., 0.]])
rot = adapter.axis_angle_to_matrix(aa)
restored = adapter.axis_angle_to_matrix(adapter.matrix_to_axis_angle(rot))
torch.testing.assert_close(rot, restored)
torch.testing.assert_close(metrics._rc().axis_angle_to_matrix(aa), rot)
assert pathlib.Path(sys.modules['prompt_hmr.utils.rotation_conversions'].__file__).is_relative_to(root)
'''
            env = dict(os.environ)
            env.pop('PYTHONPATH', None)
            subprocess.run([sys.executable, '-c', code, str(backend)], cwd=tmp,
                           env=env, check=True, capture_output=True, text=True)
            result = self.command(backend / 'experiments/postopt_ab/run_ab.py', '--help', cwd=tmp)
            self.assertIn('--versions {v2}', result.stdout)


if __name__ == '__main__':
    unittest.main()
