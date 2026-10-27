"""SAM3 dispatch, prompts and truthful source-tracker metadata (no asset hashing)."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools/gvhmr'))
from tools.gvhmr.sam3_tracking import normalized_box, tracking_orders, tracker_name
import world_pipeline as pipeline


class SAM3TrackingTests(unittest.TestCase):
    def test_box_uses_raster_normalization(self):
        np.testing.assert_allclose(normalized_box([320,180,960,720],1280,720),[.25,.25,.75,1])
        for bad in ([1,2,1,4], [-1,0,10,20], [0,0,1281,720], [0,0,float('nan'),3]):
            with self.assertRaises(ValueError):normalized_box(bad,1280,720)

    def test_anchor_streams_cover_both_directions(self):
        self.assertEqual(tracking_orders(2,5),[[2,3,4],[2,1,0]])
        self.assertEqual(tracking_orders(0,3),[[0,1,2]])
        with self.assertRaises(ValueError):tracking_orders(5,5)

    def test_requires_explicit_sam3_provenance(self):
        self.assertEqual(tracker_name({'tracker':np.array('SAM3')}),'SAM3')
        for archive in ({}, {'tracker':'SAMURAI'}, {'tracker':'YOLO'}):
            with self.subTest(archive=archive), self.assertRaises(ValueError):
                tracker_name(archive)

    def test_sam3_stage_dispatches_separate_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); params=root/'params.json'
            params.write_text(json.dumps(dict(snapshot=str(ROOT),upstream=temp,tracker='sam3',
                sam3_python='/runtime/sam3/python',sam3='/runtime/sam3/src',actor_id='actor',
                prompt_frame=12,box=[1,2,30,40],source_video_sha256='existing-graph-binding')))
            import cv2
            with patch.object(cv2,'VideoCapture') as capture, patch.object(pipeline.subprocess,'run') as run:
                capture.return_value.get.side_effect=[1280,720,30]
                pipeline.stage(argparse.Namespace(params=params,input=['video=source.mp4','checkpoint=model.pt'],out=root/'result',kind='track'))
            argv=run.call_args.args[0]
            self.assertEqual(argv[0],'/runtime/sam3/python')
            self.assertIn('sam3_tracking.py',argv[1])
            self.assertEqual(argv[argv.index('--source-video-sha256')+1],'existing-graph-binding')
            self.assertIn('--prompt-frame',argv)

    def test_bodies_keep_original_bbox_policy_and_pmpose(self):
        import tools.gvhmr.world_tracking as policy
        import tools.gvhmr.person_reconstruct as reconstruct
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);params=root/'params.json'
            params.write_text(json.dumps(dict(snapshot=str(ROOT),upstream=temp,tracker='sam3',
                actor_id='actor',pi3x_bundle=temp,gvhmr_repo=temp,gpu=0,
                pose_detector='pmpose',pmpose_python='/pose/python',pmpose_root=temp,
                pmpose_variant='PMPose-h',pmpose_ld_preload='')))
            with patch.object(policy,'install_policy') as install, patch.object(reconstruct,'main') as main:
                pipeline.stage(argparse.Namespace(params=params,input=['video=v','masks=m','lifecycle=l','pose_model=p'],out=root/'result',kind='bodies'))
                install.assert_called_once();main.assert_called_once()
                self.assertIn('--person-masks',sys.argv)
                self.assertEqual(sys.argv[sys.argv.index('--pose-detector')+1],'pmpose')

    def test_person_masks_cli_rejects_removed_alias(self):
        from tools.gvhmr.person_reconstruct import parser
        base=['--video','v','--output','o','--actor-id','a','--pi3x-bundle','b']
        self.assertEqual(parser().parse_args(base+['--person-masks','m']).person_masks,Path('m'))
        with self.assertRaises(SystemExit), patch('sys.stderr'):
            parser().parse_args(base+['--samurai-masks','m'])

    def test_tracking_stage_rejects_legacy_or_missing_provenance(self):
        import cv2
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name, tracker in [('legacy','samurai'), ('missing',None)]:
                params=root/(name+'.json')
                data=dict(snapshot=str(ROOT),upstream=temp)
                if tracker is not None:
                    data['tracker']=tracker
                params.write_text(json.dumps(data))
                with patch.object(cv2,'VideoCapture') as capture, patch.object(pipeline.subprocess,'run') as run:
                    capture.return_value.get.side_effect=[1280,720,30]
                    with self.assertRaisesRegex(ValueError,'explicit SAM3'):
                        pipeline.stage(argparse.Namespace(params=params,input=['video=source.mp4'],out=root/name,kind='track'))
                    run.assert_not_called()


if __name__=='__main__':unittest.main()
