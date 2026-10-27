"""Contract checks for the optional world-postopt integration (no GPU required)."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('world_postopt', ROOT / 'tools/gvhmr/world_postopt.py')
postopt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(postopt)


class WorldPostoptTests(unittest.TestCase):
    def test_preparation_preserves_venv_and_pins_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            upstream, run = root / 'upstream', root / 'gvhmr'
            for rel in postopt.UPSTREAM_FILES:
                p = upstream / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('# source\n')
            for rel in ('motion_native.npz', 'provenance.json', 'hmr4d_results.pt',
                        'preprocess/bbx.pt', 'preprocess/vitpose.pt', 'active_inference/hmr4d_results.pt'):
                p = run / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('input')
            model, camera = root / 'model.npz', root / 'camera.npz'
            model.write_text('model'); camera.write_text('camera')
            Path(str(camera) + '.json').write_text('{}')
            python = root / 'venv/bin/python'; python.parent.mkdir(parents=True)
            python.symlink_to('/usr/bin/python3')
            dest = root / 'prepared'
            postopt.prepare(argparse.Namespace(out=dest, prompt_hmr=upstream, python=python,
                smplx_model=model, camera_tracks=camera, gvhmr_run=run, actor_id='actor', gpu=0))
            config = json.loads((dest / 'experiment.json').read_text())
            self.assertEqual([s['id'] for s in config['stages']], ['adapt', 'v2', 'measure'])
            self.assertEqual(config['stages'][0]['argv'][0], str(python))
            self.assertIn('active_hmr', config['stages'][0]['inputs'])
            self.assertTrue(all('sha256' in c for s in config['stages'] for c in s['code']))
            self.assertFalse(config['stages'][1]['params']['postopt_optimize_scale'])
            from aha3d.workflow.human_runner import load_config
            load_config(dest / 'experiment.json')
            with self.assertRaises(FileExistsError):
                postopt.prepare(argparse.Namespace(out=dest))

    def test_rejects_actor_sparse_camera_and_timing_mismatch(self):
        native = dict(source_actor_id='actor', fps=30., frame_times_seconds=np.arange(4) / 30,
                      track_active=np.array([1, 1, 1, 0]))
        camera = dict(c2w=np.tile(np.eye(4), (4, 1, 1)), time_seconds=np.arange(4) / 30,
                      camera_source='pi3x_dense_single_pass')
        provenance = dict(actor_id='actor', tracker='sam3', camera_estimator='pi3x')
        self.assertEqual(postopt.validate_inputs(native, camera, provenance, 'actor'), 30.)
        for change, expected in [({'camera_source': 'pi3x_interpolated'}, 'dense'),
                                 ({'time_seconds': np.arange(4) / 24}, 'timestamps')]:
            with self.assertRaisesRegex(ValueError, expected):
                postopt.validate_inputs(native, dict(camera, **change), provenance, 'actor')
        with self.assertRaisesRegex(ValueError, 'identity'):
            postopt.validate_inputs(native, camera, provenance, 'other')

    def test_camera_path_is_not_a_motion_quality_gate(self):
        def row(path, slip):
            return dict(camera=dict(path_m=path), people={1: dict(contact_slip_mps_mean=slip,
                root_accel_rms=2., penetration_cm_mean=.2, contact_height_cm_median=1.)})
        table=dict(gvhmr_global=row(1., .05), v2=row(30., .03), lifted_init=row(1., .7))
        decision=postopt.assess(table)
        self.assertFalse(decision['camera_path_is_quality_criterion'])
        self.assertAlmostEqual(decision['metric_deltas_vs_gvhmr']['v2']['1']['contact_slip_mps_mean'], -.02)
        self.assertFalse(decision['allow_scene_import'])

    def test_person_extrinsics_inherit_shared_intrinsics(self):
        result={'camera_world': {'img_focal':1000, 'Rcw':'shared'},
                'people': {1: {'camera_world': {'Rcw':'person'}}}}
        camera=postopt.person_camera(result,1)
        self.assertEqual(camera,{'img_focal':1000,'Rcw':'person'})
        self.assertEqual(result['camera_world']['Rcw'],'shared')

    def test_original_camera_diagnostic_overrides_every_person(self):
        source={'camera_world': {'id':'top'}, 'people': {1: {'camera_world': {'id':'a'}}, 2: {'camera_world': {'id':'b'}}}}
        camera={'id':'original'}
        fixed=postopt.with_camera(source,camera)
        self.assertTrue(all(p['camera_world']==camera for p in fixed['people'].values()))
        self.assertEqual(source['people'][1]['camera_world']['id'],'a')

    def test_global_orientation_is_required_without_silent_camera_fallback(self):
        result={'people': {1: {'frames':[0,1], 'smplx_cam': {}}}}
        with self.assertRaisesRegex(ValueError,'global-branch'):
            postopt.require_global_orientation(result, postopt.optimizer_config())
        result['people'][1]['smplx_cam']['global_orient_world']=np.zeros((2,3))
        postopt.require_global_orientation(result, postopt.optimizer_config())
        result['people'][1]['smplx_cam']['global_orient_world'][0,0]=np.nan
        with self.assertRaises(ValueError):
            postopt.require_global_orientation(result, postopt.optimizer_config())

    def test_corrected_optimizer_configuration(self):
        cfg=postopt.optimizer_config()
        self.assertEqual(cfg['postopt_orient_source'],'global')
        self.assertEqual(cfg['postopt_acc'],.1)
        self.assertEqual(cfg['postopt_lr_v2'],.01)
        self.assertEqual(cfg['postopt_cont_vel'],1000.)
        self.assertNotIn('postopt_lr',cfg)


if __name__ == '__main__':
    unittest.main()
