"""Regression coverage for sustained partial-mask tracking dropouts."""
import unittest

import numpy as np

from tools.gvhmr.person_tracking import gate_trailing_max, interpolate_centre_scale


class TrackingCropGateTests(unittest.TestCase):
    def test_long_collapse_stays_rejected_until_full_size_recovery(self):
        boxes = np.tile([0., 0., 100., 400.], (70, 1))
        boxes[10:55, 2:] = [10, 40]
        keep = gate_trailing_max(boxes, np.ones(70, bool))
        self.assertTrue(keep[:10].all())
        self.assertFalse(keep[10:55].any())
        self.assertTrue(keep[55:].all())
        filled = interpolate_centre_scale(boxes, keep).numpy()
        np.testing.assert_allclose(filled, np.tile([0, 0, 100, 400], (70, 1)))

    def test_sparse_mask_stays_rejected_despite_full_size_bounds(self):
        boxes = np.tile([0., 0., 100., 400.], (70, 1))
        area = np.full(70, 30000.)
        area[10:55] = 200
        keep = gate_trailing_max(boxes, np.ones(70, bool), area=area)
        self.assertTrue(keep[:10].all())
        self.assertFalse(keep[10:55].any())
        self.assertTrue(keep[55:].all())


if __name__ == '__main__':
    unittest.main()
