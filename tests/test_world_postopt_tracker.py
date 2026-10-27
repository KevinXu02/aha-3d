import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('world_postopt', ROOT / 'tools/gvhmr/world_postopt.py')
postopt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(postopt)


class TrackerProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.native = dict(source_actor_id='actor', fps=30., frame_times_seconds=np.arange(4) / 30,
                           track_active=np.array([1, 1, 1, 0]))
        self.camera = dict(c2w=np.tile(np.eye(4), (4, 1, 1)), time_seconds=np.arange(4) / 30,
                           camera_source='pi3x_dense_single_pass')

    def test_sam3_tracking_is_accepted(self):
        provenance = dict(actor_id='actor', tracker='sam3', camera_estimator='pi3x')
        self.assertEqual(postopt.validate_inputs(self.native, self.camera, provenance, 'actor'), 30.)

    def test_unknown_tracker_or_camera_estimator_is_rejected(self):
        for provenance in (dict(actor_id='actor', tracker='yolo', camera_estimator='pi3x'),
                           dict(actor_id='actor', tracker='samurai', camera_estimator='pi3x'),
                           dict(actor_id='actor', camera_estimator='pi3x'),
                           dict(actor_id='actor', tracker='sam3', camera_estimator='dpvo')):
            with self.assertRaisesRegex(ValueError, 'tracking and Pi3X'):
                postopt.validate_inputs(self.native, self.camera, provenance, 'actor')


if __name__ == '__main__':
    unittest.main()
