"""Terminal activity, exact prefix and inactive consumer contracts; no model inference."""
import json
from pathlib import Path
import tempfile
import unittest
import importlib.util
import os
import numpy as np
from tools.gvhmr.track_lifecycle import (track_active_mask, source_frame_mapping,
    load_lifecycle, edge_exit_proposal, pad_inactive, write_lossless_prefix, normalized_source_times)

PINNED_REPO = Path(os.environ.get('GVHMR_ROOT', Path(__file__).resolve().parents[1]/'.runtime/gvhmr-bedlam2'))

class LifecycleTests(unittest.TestCase):
    def test_activity_is_not_missing_observation_support(self):
        np.testing.assert_array_equal(track_active_mask(None,7),np.ones(7,bool))
        for bad in ([True,False,True], [False,True,True], [False,False,False], [1,1,0]):
            with self.assertRaises(ValueError):track_active_mask(bad,3)
        np.testing.assert_array_equal(track_active_mask([True,True,False],3),[True,True,False])
        with self.assertRaises(ValueError):source_frame_mapping([0,1,3],3)
        with self.assertRaises(ValueError):source_frame_mapping([10,11,12],3)
        with self.assertRaisesRegex(ValueError,'zero-origin'):normalized_source_times(np.arange(7)/30+.5)
        with self.assertRaises(ValueError):normalized_source_times(np.arange(7)/24)

    def test_bound_review_and_terminal_frame(self):
        t=np.arange(7)/30;data=dict(schema_version=1,source_video_sha256='a'*64,actor_id='person',
            time_seconds=t.tolist(),image_size=[16,12],reviewer='reviewer',reason='Source boundary exit',
            evidence_frames=[4,5,6],terminal_exit_frame=5)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'review.json'
            def run(value):
                p.write_text(json.dumps(value));return load_lifecycle(p,times=t,image_size=[16,12],source_sha256='a'*64,actor_id='person')
            active,receipt=run(data);np.testing.assert_array_equal(active,[True]*5+[False]*2)
            self.assertEqual(receipt['terminal_exit_frame'],5)
            active,_=run(dict(data,terminal_exit_frame=None));self.assertTrue(active.all())
            for key,value in [('source_video_sha256','b'*64),('actor_id','wrong'),('image_size',[12,16]),('time_seconds',(t+.001).tolist()),('terminal_exit_frame',5.0),('evidence_frames',[4]),('reviewer','')]:
                with self.subTest(key=key),self.assertRaises(ValueError):run(dict(data,**{key:value}))

    def test_edge_proposal_is_review_only_and_ignores_initial_interior_gaps(self):
        masks=np.zeros((12,10,10),bool);masks[2:5,3:7,3:7]=True;masks[8:,3:7,3:7]=True
        self.assertIsNone(edge_exit_proposal(masks)['candidate'])
        masks[8:]=False
        self.assertIsNone(edge_exit_proposal(masks)['candidate']) # internal disappearance
        masks[4,3:8,9]=True
        proposal=edge_exit_proposal(masks);self.assertEqual(proposal['candidate'],5);self.assertTrue(proposal['requires_review'])

    def test_inactive_padding(self):
        active=np.arange(8)<5
        a=np.arange(15,dtype=float).reshape(5,3);padded=pad_inactive(a,active)
        np.testing.assert_array_equal(padded[:5],a);np.testing.assert_array_equal(padded[5:],np.repeat(a[-1:],3,axis=0))
        self.assertTrue(np.isnan(pad_inactive(a,active,fill='nan')[5:]).all())
        with self.assertRaises(ValueError):pad_inactive(padded,active)

    def test_projection_excludes_inactive_high_confidence_rows(self):
        from tools.gvhmr.export_motion_quality import projection_arrays
        n=6;kp=np.ones((n,17,3));j=np.ones((n,17,3));active=np.arange(n)<4
        obs,pred=projection_arrays(kp,j,np.eye(3),np.arange(n)/30,[16,12],track_active=active)
        self.assertTrue(np.isnan(pred['uv'][4:]).all());self.assertFalse(obs['detected'][4:].any())
        self.assertTrue(np.isfinite(pred['uv'][:4]).all())

    def test_ground_ignores_inactive_penetration_and_stance(self):
        from tools.gvhmr.audit_native_ground import native_ground_arrays
        v=np.ones((6,4,3));v[4:,:,1]=-100.;j=np.ones((6,22,3));active=np.arange(6)<4
        arrays,summary=native_ground_arrays(v,j,np.array([[0,1,2],[1,2,3]]),[np.array([0,1]),np.array([2,3])],np.arange(6)/30,0,track_active=active)
        self.assertEqual(summary['mesh']['maximum_penetration_m'],0)
        self.assertTrue(np.isnan(arrays['mesh_max_penetration_m'][4:]).all())
        self.assertTrue(np.isnan(arrays['left_vertex_speed_m_s'][3:]).all())
        with self.assertRaisesRegex(ValueError,'after terminal'):
            native_ground_arrays(v,j,np.array([[0,1,2],[1,2,3]]),[np.array([0,1]),np.array([2,3])],np.arange(6)/30,0,track_active=active,stationary=np.ones((6,2),bool))

    def test_exact_lossless_prefix_source_pixels(self):
        import av
        from fractions import Fraction
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);src=p/'input.avi'
            with av.open(str(src),'w') as out:
                stream=out.add_stream('ffv1',rate=30);stream.width=16;stream.height=12;stream.pix_fmt='bgr0'
                for i in range(8):
                    rgb=np.random.default_rng(i).integers(0,256,(12,16,3),dtype=np.uint8)
                    frame=av.VideoFrame.from_ndarray(rgb,format='rgb24');frame.pts=i;frame.time_base=Fraction(1,30)
                    for packet in stream.encode(frame):out.mux(packet)
                for packet in stream.encode():out.mux(packet)
            receipt=write_lossless_prefix(src,p/'prefix.avi',np.arange(8)<5)
            self.assertEqual(receipt['frames'],5);self.assertTrue(receipt['lossless_rgb_verified'])
            from tools.gvhmr.tracking_evidence import video_timeline
            np.testing.assert_array_equal(video_timeline(p/'prefix.avi')['time_seconds'],np.arange(5)/30)

    @unittest.skipUnless((PINNED_REPO/'hmr4d').is_dir() and all(importlib.util.find_spec(k) is not None for k in ('torch','colorlog','pytorch3d','pytorch_lightning')),
                         'Optional pinned-helper integration requires separate GVHMR runtime')
    def test_real_adapter_stops_box_smoothing_and_preserves_missing_prefix_rows(self):
        import av,torch
        from tools.gvhmr.person_tracking import prepare_gvhmr_boxes
        from tools.gvhmr.tracking_evidence import video_timeline,file_sha256,validated_detected
        repo=PINNED_REPO
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);video=root/'source.avi'
            with av.open(str(video),'w') as out:
                stream=out.add_stream('ffv1',rate=30);stream.width=16;stream.height=12;stream.pix_fmt='bgr0'
                for i in range(10):
                    for packet in stream.encode(av.VideoFrame.from_ndarray(np.full((12,16,3),i*20,np.uint8),format='rgb24')):out.mux(packet)
                for packet in stream.encode():out.mux(packet)
            timeline=video_timeline(video);m=np.zeros((10,12,16),bool);m[2:5,2:10,4:12]=True;m[8,1:12,:]=True
            archive=root/'masks.npz';active=np.arange(10)<7
            np.savez_compressed(archive,masks=m,time_seconds=timeline['time_seconds'],image_size=timeline['image_size'],source_video_sha256=np.array(file_sha256(video)),actor_id=np.array('target'),tracker=np.array('SAM3'))
            result=prepare_gvhmr_boxes(archive,video,'target',repo,root/'boxes',track_active=active)
            prefix=torch.load(root/'boxes/bbx.pt',weights_only=True)['bbx_xyxy'];self.assertEqual(len(prefix),7)
            full=np.load(root/'boxes/person_boxes.npz');self.assertTrue(np.isnan(full['smoothed_bbox_xyxy'][7:]).all())
            self.assertTrue(np.isfinite(full['smoothed_bbox_xyxy'][:7]).all())
            raw=json.loads((root/'boxes/raw_tracking.json').read_text());support=validated_detected(raw,timeline['time_seconds'],timeline['image_size'],file_sha256(video),full['smoothed_bbox_xyxy'])
            np.testing.assert_array_equal(support,[False,False,True,True,True,False,False,False,False,False])
            self.assertEqual(result['active_frames'],7);self.assertTrue(full['raw_mask_nonempty'][8])
            full.close()


if __name__=='__main__':unittest.main()
