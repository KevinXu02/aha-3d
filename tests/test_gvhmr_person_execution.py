"""Exercise the source-tracking launcher's local device contract."""
import unittest

from tools.gvhmr.person_reconstruct import parser
from tools.gvhmr.run import configure_execution


class PersonExecutionTests(unittest.TestCase):
    def arguments(self, *extra):
        return parser().parse_args([
            '--video', 'source.mp4', '--output', 'new-run', '--actor-id', 'actor',
            '--pi3x-bundle', 'cameras', '--person-masks', 'masks.npz', *extra,
        ])

    def test_default_preserves_inherited_devices(self):
        args = self.arguments()
        env, report = configure_execution(args.gpu, environ={'INDOOR_RUN_ID': '123', 'CUDA_VISIBLE_DEVICES': '3,5'},
                                          hostnames=['workstation'])
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '3,5')
        self.assertEqual(report['run_id'], '123')

    def test_default_without_mask_uses_device_zero(self):
        args = self.arguments()
        env, _ = configure_execution(args.gpu, environ={}, hostnames=['workstation'])
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '0')

    def test_explicit_gpu(self):
        args = self.arguments('--gpu', '2')
        env, _ = configure_execution(args.gpu, environ={}, hostnames=['workstation'])
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '2')

    def test_explicit_gpu_must_stay_inside_inherited_mask(self):
        args = self.arguments('--gpu', '0')
        with self.assertRaises(ValueError):
            configure_execution(args.gpu, environ={'CUDA_VISIBLE_DEVICES': '3,5'}, hostnames=['workstation'])
