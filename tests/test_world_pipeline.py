"""Full-chain graph wiring must reconstruct inputs rather than adopt caches."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/gvhmr'))
import world_pipeline as pipeline


class WorldPipelineTests(unittest.TestCase):
    def test_review_stage_dispatches_renderer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            params = root / 'params.json'
            params.write_text(json.dumps(dict(snapshot=temp, upstream=temp)))
            with patch.object(pipeline, 'review') as render:
                pipeline.stage(argparse.Namespace(params=params, input=[], out=root / 'result', kind='review'))
                render.assert_called_once_with(root / 'result', root, {})

    @patch('tools.gvhmr.scene_ground.load_prior', return_value={})
    @patch.object(pipeline.post, 'fingerprint', side_effect=lambda p: {
        'path': str(Path(p).resolve()), 'sha256': '0' * 64})
    def test_full_graph_fresh_stages_and_v2_only(self, fingerprint, load_prior):
        # Graph wiring only: supplied identities/prior validation are mocked;
        # this test performs no asset checksum or scene acceptance check.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            upstream = root / 'upstream'
            for rel in (*pipeline.post.UPSTREAM_FILES, *('experiments/postopt_ab/' + f for f in pipeline.FRONTEND_FILES)):
                p = upstream / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('# source')
            model = root / 'model'; model.write_text('model')
            bundle = root / 'bundle'; bundle.mkdir()
            for name in ('inputs.json', 'cameras.json', 'camera_review.json'):
                (bundle / name).write_text('{}')
            gvhmr = root / 'gvhmr'
            for rel in ('inputs/checkpoints/gvhmr/gvhmr_b1b2.ckpt', 'tools/demo/demo.py'):
                p = gvhmr / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('asset')
            dest = root / 'prepared'
            args = argparse.Namespace(out=dest, prompt_hmr=upstream, python=Path(sys.executable),
                smplx_model=model, actor_id='actor', gpu=0, video=model, pi3x_bundle=bundle,
                lifecycle_review=model, sam3=root/'sam3', sam3_checkpoint=model, sam3_python=Path('/sam3/python'),
                pose_detector='pmpose', pmpose_python=Path(sys.executable), pmpose_root=root/'bmp',
                pmpose_checkpoint=model, pmpose_variant='PMPose-h', pmpose_ld_preload='',
                gvhmr_repo=gvhmr, pi3=root, pi3_checkpoint=model, box=[1, 2, 3, 4], prompt_frame=0)
            pipeline.prepare(args)
            cfg = json.loads((dest / 'experiment.json').read_text())
            self.assertEqual([s['id'] for s in cfg['stages']],
                             ['track', 'lifecycle', 'bodies', 'camera', 'adapt', 'v2', 'measure', 'review'])
            from aha3d.workflow.human_runner import load_config
            load_config(dest / 'experiment.json')
            stages = {s['id']: s for s in cfg['stages']}
            self.assertEqual(stages['bodies']['inputs']['masks'], {'stage': 'track', 'output': 'masks'})
            self.assertEqual(stages['adapt']['inputs']['bodies'], {'stage': 'bodies', 'output': 'run'})
            self.assertEqual(stages['adapt']['inputs']['camera'], {'stage': 'camera', 'output': 'camera'})
            self.assertEqual(stages['bodies']['params']['pose_detector'], 'pmpose')
            self.assertEqual(stages['bodies']['inputs']['pose_model']['path'], str(model))
            self.assertEqual(stages['measure']['inputs']['gvhmr'], {'stage': 'adapt', 'output': 'gvhmr'})
            self.assertEqual(stages['review']['inputs']['gvhmr'], {'stage': 'adapt', 'output': 'gvhmr'})
            self.assertEqual(stages['v2']['params']['optimizer']['postopt_orient_source'], 'global')
            self.assertNotIn('v1', stages)
            self.assertFalse(stages['v2']['params']['optimizer']['postopt_optimize_scale'])
            self.assertNotEqual(stages['bodies']['params']['snapshot'], str(ROOT))
            self.assertEqual(stages['track']['params']['tracker'], 'sam3')
            self.assertEqual(stages['track']['params']['sam3_python'], '/sam3/python')
            self.assertEqual(stages['track']['inputs']['checkpoint']['path'], str(model))
            self.assertFalse(any(k.startswith('samurai') for k in stages['track']['params']))
            prior = root/'ground.json'
            prior.write_text(json.dumps(dict(source_video_sha256=pipeline.post.fingerprint(model)['sha256'],
                coordinate_frame='pi3x-aligned-world', scale=1, accepted=True, evidence='test reviewed floor',
                plane=[0, 0, 1, 2], upright=[0, 0, 1])))
            args.out = root/'scene-prepared'; args.scene_ground = prior
            # The normal scene-ground graph needs no Kimodo arguments or assets.
            pipeline.prepare(args)
            load_config(args.out/'experiment.json')
            scene_stages = {s['id']: s for s in json.loads((args.out/'experiment.json').read_text())['stages']}
            self.assertEqual(list(scene_stages), list(stages))
            for sid in ('camera', 'adapt', 'review'):
                self.assertEqual(scene_stages[sid]['inputs']['bodies'], {'stage': 'bodies', 'output': 'run'})
            for sid in ('bodies', 'adapt'):
                self.assertEqual(scene_stages[sid]['inputs']['scene_ground']['path'], str(prior))
            self.assertFalse(scene_stages['bodies']['params']['kimodo_completion'])
            self.assertNotIn('kimodo_python', scene_stages['bodies']['params'])

            args.out = root/'completion-prepared'; args.kimodo_completion = True
            args.completion_prompt = ' '
            with self.assertRaisesRegex(ValueError, '--completion-prompt'):
                pipeline.prepare(args)
            self.assertFalse(args.out.exists())
            checkpoint = root/'kimodo'; checkpoint.mkdir()
            for name in ('config.yaml', 'model.safetensors'):
                (checkpoint/name).write_text('fixture')
            args.kimodo_python = Path(sys.executable); args.kimodo_checkpoint = checkpoint
            args.kimodo_upstream = root/'upstream-kimodo'; args.completion_prompt = 'A person walks.'
            args.completion_context = 3
            pipeline.prepare(args)
            load_config(args.out/'experiment.json')
            scene_stages = {s['id']: s for s in json.loads((args.out/'experiment.json').read_text())['stages']}
            self.assertIn('complete', scene_stages)
            self.assertEqual(scene_stages['camera']['inputs']['bodies'], {'stage': 'complete', 'output': 'run'})
            self.assertEqual(scene_stages['adapt']['inputs']['bodies'], {'stage': 'complete', 'output': 'run'})
            self.assertEqual(scene_stages['bodies']['inputs']['scene_ground']['path'], str(prior))
            self.assertEqual(scene_stages['complete']['params']['completion_context'], 3)


if __name__ == '__main__':
    unittest.main()
