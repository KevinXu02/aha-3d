import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from tools.gvhmr import pmpose


class PMPoseTests(unittest.TestCase):
    def test_default_and_explicit_control(self):
        p = argparse.ArgumentParser(); pmpose.add_arguments(p)
        self.assertEqual(p.parse_args([]).pose_detector, 'pmpose')
        self.assertEqual(p.parse_args(['--pose-detector', 'vitpose']).pose_detector, 'vitpose')

    def test_joint_mapping_and_invalid_outputs(self):
        value = np.zeros((23, 3)); value[:, 0] = np.arange(23); value[:, 2] = .5
        np.testing.assert_array_equal(pmpose.coco17(value)[:, 0], np.arange(17))
        for invalid in (value[:17], np.full((23, 3), np.nan), value + 2):
            with self.assertRaises(ValueError):
                pmpose.coco17(invalid)

    def test_requires_mask_without_silent_fallback(self):
        with patch.object(pmpose.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'requires source-person masks'):
                pmpose.extract(argparse.Namespace(person_masks=None), None, None, None, 3)
            run.assert_not_called()

    def test_active_prefix_keeps_real_masks_and_zeroes_empty_frames(self):
        import os
        import sys
        import torch
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'mmpose').mkdir()
            boxes = np.tile([0, 0, 4, 4], (3, 1)).astype(np.float32)
            torch.save({'bbx_xyxy': torch.from_numpy(boxes)}, root/'bbx.pt')
            masks = np.zeros((5, 4, 4), dtype=bool)
            masks[1, 1, 1] = True; masks[2:] = True
            np.savez(root/'masks.npz', masks=masks)
            prediction = np.zeros((1, 23, 3), dtype=np.float32)
            prediction[..., 2] = .7
            model = MagicMock(); model.predict.return_value = (prediction,)
            cap = MagicMock(); cap.isOpened.return_value = True
            cap.read.return_value = (True, np.zeros((4, 4, 3), dtype=np.uint8))
            a = argparse.Namespace(root=root, video=root/'source.mp4', boxes=root/'bbx.pt',
                masks=root/'masks.npz', output=root/'kp.pt', checkpoint=root/'model.pth', frames=3, variant='PMPose-h')
            cwd, oldpath = Path.cwd(), sys.path[:]
            try:
                with patch.dict(sys.modules, {'cv2': SimpleNamespace(VideoCapture=lambda _: cap),
                                              'mmpretrain': SimpleNamespace(),
                                              'pmpose': SimpleNamespace(PMPose=lambda **kw: model)}):
                    pmpose.worker(a)
            finally:
                os.chdir(cwd); sys.path[:] = oldpath
            kp = torch.load(root/'kp.pt', weights_only=True).numpy()
            self.assertEqual(kp.shape, (3, 17, 3))
            self.assertTrue((kp[0] == 0).all())
            self.assertEqual(model.predict.call_count, 2)
            np.testing.assert_array_equal(model.predict.call_args_list[0].kwargs['masks'][0], masks[1])
            self.assertEqual(cap.read.call_count, 3)

    def test_uses_venv_path_without_resolving_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); python = root/'python'; python.symlink_to('/usr/bin/python3')
            output = root/'kp.pt'; Path(str(output)+'.json').write_text('{"detector":"pmpose"}')
            a = argparse.Namespace(person_masks=root/'masks.npz', pmpose_python=python,
                pmpose_root=root, pmpose_checkpoint=root/'model.pth', pmpose_variant='PMPose-h', pmpose_ld_preload='')
            with patch.object(pmpose.subprocess, 'run') as run:
                pmpose.extract(a, root/'source.mp4', root/'bbx.pt', output, 3)
            self.assertEqual(run.call_args.args[0][0], str(python))


if __name__ == '__main__':
    unittest.main()
