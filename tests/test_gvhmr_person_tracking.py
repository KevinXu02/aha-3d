"""CPU contracts: mask evidence, real pinned numerical crop path, fail-closed injection."""
from pathlib import Path
import copy
import json
import tempfile
import importlib.util
import os
import unittest
import numpy as np
from tools.gvhmr.person_tracking import mask_boxes,validate_archive,forbid_default_tracker,dense_upstream_boxes,upstream_helpers

PINNED_REPO = Path(os.environ.get('GVHMR_ROOT', Path(__file__).resolve().parents[1]/'.runtime/gvhmr-bedlam2'))
PINNED_RUNTIME_AVAILABLE = ((PINNED_REPO/'hmr4d').is_dir()
    and all(importlib.util.find_spec(name) is not None for name in
            ('torch','av','einops','scipy','colorlog','pytorch_lightning','pytorch3d')))
PINNED_RUNTIME_REASON = 'Integration needs pinned checkout and dedicated GVHMR runtime dependencies; do not install these into shared core'

class MaskEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.times=np.arange(7)/30;self.mask=np.zeros((7,12,16),bool);self.mask[2,2:8,3:7]=True;self.mask[4,4:12,8:16]=True
        self.data=dict(masks=self.mask,time_seconds=self.times,image_size=np.array([16,12]),source_video_sha256=np.array('a'*64),actor_id=np.array('reviewed_actor'),tracker=np.array('SAM3'))
    def validate(self,data):return validate_archive(data,self.times,[16,12],'a'*64,'reviewed_actor')
    def test_raw_mask_extent_and_empty_support(self):
        box,valid,area=self.validate(self.data);np.testing.assert_array_equal(box[2],[3,2,7,8]);np.testing.assert_array_equal(box[4],[8,4,16,12]);self.assertTrue(np.isnan(box[0]).all());self.assertEqual(valid.sum(),2);self.assertEqual(area[2],24)
    def test_hash_actor_and_exact_time_binding(self):
        for key,value in [('source_video_sha256',np.array('b'*64)),('actor_id',np.array('wrong_actor')),('time_seconds',self.times+.001),('image_size',np.array([12,16]))]:
            d=dict(self.data);d[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.validate(d)
    def test_missing_rows_and_all_empty_rejected(self):
        for mask in [self.mask[:-1],np.zeros_like(self.mask),self.mask.astype(np.uint8)]:
            d=dict(self.data,masks=mask)
            with self.assertRaises(ValueError):self.validate(d)
    def test_claimed_boxes_cannot_replace_mask_evidence(self):
        boxes,_,area=self.validate(self.data);d=dict(self.data,bbox_xyxy_boundary=boxes.copy(),mask_area_pixels=area.copy());self.validate(d);d['bbox_xyxy_boundary'][2,0]+=1
        with self.assertRaises(ValueError):self.validate(d)
        d=dict(self.data,mask_area_pixels=area+1)
        with self.assertRaises(ValueError):self.validate(d)
    def test_default_tracker_guard_restores_even_on_exception(self):
        class Tracker:
            def __init__(self):self.called=True
        original=Tracker.__init__
        with self.assertRaisesRegex(RuntimeError,'Default Tracker invoked'):
            with forbid_default_tracker(Tracker):Tracker()
        self.assertIs(Tracker.__init__,original);self.assertTrue(Tracker().called)
    @unittest.skipUnless(PINNED_RUNTIME_AVAILABLE,PINNED_RUNTIME_REASON)
    def test_actual_pinned_gap_smoothing_and_crop_math(self):
        repo=PINNED_REPO;boxes,valid,_=self.validate(self.data)
        dense,xys,filled=dense_upstream_boxes(boxes,valid,upstream_helpers(repo))
        expected=np.array([[3,2,7,8],[3,2,7,8],[3,2,7,8],[5.5,3,11.5,10],[8,4,16,12],[8,4,16,12],[8,4,16,12]],np.float32)
        np.testing.assert_array_equal(filled.numpy(),expected)
        for _ in range(2):expected=np.stack([np.convolve(np.pad(expected[:,j],(2,2),mode='edge'),np.ones(5)/5,mode='valid') for j in range(4)],axis=-1)
        np.testing.assert_allclose(dense.numpy(),expected,atol=2e-6,rtol=0)
        # Upstream fits192:256 aspect then returns its larger dimension: max(height,width*4/3)*1.2.
        width=expected[:,2]-expected[:,0];height=expected[:,3]-expected[:,1];wanted=np.c_[(expected[:,:2]+expected[:,2:])/2,np.maximum(height,width*4/3)*1.2]
        np.testing.assert_allclose(xys.numpy(),wanted,atol=3e-6,rtol=0)

    @unittest.skipUnless(PINNED_RUNTIME_AVAILABLE,PINNED_RUNTIME_REASON)
    def test_fresh_video_injection_binds_exporter_support_and_refuses_overwrite(self):
        import av,torch
        from fractions import Fraction
        from tools.gvhmr.person_tracking import prepare_gvhmr_boxes
        from tools.gvhmr.tracking_evidence import file_sha256,video_timeline,validated_detected
        repo=PINNED_REPO
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);video=root/'source.mp4';archive=root/'masks.npz'
            with av.open(str(video),'w') as out:
                stream=out.add_stream('libx264',rate=30);stream.width=16;stream.height=12;stream.pix_fmt='yuv420p'
                for i in range(7):
                    pixels=np.full((12,16,3),i*20,np.uint8);frame=av.VideoFrame.from_ndarray(pixels,format='rgb24');frame.pts=i;frame.time_base=Fraction(1,30)
                    for packet in stream.encode(frame):out.mux(packet)
                for packet in stream.encode():out.mux(packet)
            timeline=video_timeline(video);digest=file_sha256(video)
            np.savez_compressed(archive,**dict(self.data,source_video_sha256=np.array(digest),time_seconds=timeline['time_seconds']))
            result=prepare_gvhmr_boxes(archive,video,'reviewed_actor',repo,root/'injected')
            dense=torch.load(root/'injected/bbx.pt',weights_only=True)['bbx_xyxy'].numpy()
            sidecar=json.loads((root/'injected/raw_tracking.json').read_text())
            support=validated_detected(sidecar,timeline['time_seconds'],timeline['image_size'],digest,dense)
            np.testing.assert_array_equal(support,[False,False,True,False,True,False,False])
            self.assertEqual(result['injected_bbx_file_sha256'],file_sha256(root/'injected/bbx.pt'))
            self.assertFalse(result['default_tracker_invoked']);self.assertEqual(sidecar['tracker'],'SAM3')
            with self.assertRaises(FileExistsError):prepare_gvhmr_boxes(archive,video,'reviewed_actor',repo,root/'injected')

if __name__=='__main__':unittest.main()
