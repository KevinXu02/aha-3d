#!/usr/bin/env python3
"""Detect PMPose low-evidence intervals and replace them with constrained Kimodo.

Run in the shared Kimodo environment. Input GVHMR runs are immutable. Unresolved
intervals are explicitly invalid and block downstream optimization by default.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]


def run(args):
    import numpy as np
    import torch
    torch.set_num_threads(4)
    import cv2
    from scipy.spatial.transform import Rotation
    from aha3d.motion.endpoint_completion import propose_gaps, generate_gap
    from aha3d.motion.completion import CompletionRejected
    from tools.gvhmr.tracking_evidence import file_sha256

    manifest = json.loads((args.run / 'provenance.json').read_text())
    if manifest.get('pose_detector') != 'pmpose':
        raise ValueError('Automatic low-evidence completion requires the PMPose input evidence')
    if manifest.get('person_boxes', {}).get('mask_archive_sha256') != file_sha256(args.masks):
        raise ValueError('Completion must use the exact masks consumed by this actor inference')
    with np.load(args.run / 'motion_native.npz', allow_pickle=False) as z:
        native = {key: z[key].copy() for key in z.files}
    active = native['track_active']
    times = native['frame_times_seconds']
    if not np.allclose(np.diff(times), 1/30, atol=1e-6, rtol=0):
        raise ValueError('Native endpoint completion requires the source 30 Hz timeline')
    kp = torch.load(args.run / 'preprocess/vitpose.pt', map_location='cpu', weights_only=True).numpy()
    with np.load(args.masks, allow_pickle=False) as z:
        masks = z['masks']
        for key, expected in [('source_video_sha256', manifest['source_video_sha256']), ('actor_id', manifest['actor_id'])]:
            if key in z and str(z[key].item()) != expected:
                raise ValueError('Completion masks belong to a different video/actor')
    if len(masks) != len(kp):
        raise ValueError('Completion masks and PMPose timelines differ')
    height, width = masks.shape[1:]
    support = np.zeros(kp.shape[:2], bool)
    for frame in range(len(kp)):
        near = cv2.dilate(masks[frame].astype('uint8'), np.ones((11, 11), 'uint8'))
        xy = np.rint(np.nan_to_num(kp[frame, :, :2])).astype(int)
        support[frame] = near[np.clip(xy[:, 1], 0, height-1), np.clip(xy[:, 0], 0, width-1)] > 0
    report = propose_gaps(kp, active, confidence=args.confidence, min_visible=args.min_visible,
                          context=args.context, image_size=(width, height), mask_support=support)
    report.update(source_video_sha256=manifest['source_video_sha256'], actor_id=manifest['actor_id'],
                  source_native_sha256=file_sha256(args.run / 'motion_native.npz'),
                  source_keypoints_sha256=file_sha256(args.run / 'preprocess/vitpose.pt'),
                  masks_sha256=file_sha256(args.masks), generated_frames=[], motion_quality_validated=False)
    report['implementation_sha256'] = file_sha256(__file__)
    report['endpoint_adapter_sha256'] = file_sha256(ROOT/'src/aha3d/motion/endpoint_completion.py')
    args.out.mkdir(parents=True, exist_ok=False)
    def save_report():
        (args.out / 'completion.json').write_text(json.dumps(report, indent=2)+'\n')
    save_report()
    if args.detect_only:
        return report
    if args.checkpoint.name != 'Kimodo-SMPLX-RP-v1' or not (args.checkpoint/'config.yaml').is_file():
        raise ValueError('Supply the installed Kimodo-SMPLX-RP-v1 checkpoint directory')
    prefix = 'smpl_params_global__'
    n = len(active)
    local = Rotation.from_rotvec(np.concatenate([native[prefix+'global_orient'][:, None],
            native[prefix+'body_pose'].reshape(n, 21, 3)], axis=1).reshape(-1, 3)).as_matrix().reshape(n, 22, 3, 3)
    betas = native[prefix+'betas']
    if not np.allclose(betas, betas[0], atol=1e-5, rtol=0):
        raise ValueError('Completion requires one fixed source body shape')
    # Exact shaped rest joints need no extra SMPL-X package in the shared runtime.
    with np.load(args.smplx_model, allow_pickle=True) as geometry:
        regressor = geometry['J_regressor']
        if regressor.dtype == object:
            regressor = regressor.item()
        shaped = geometry['v_template'] + np.einsum('vck,k->vc', geometry['shapedirs'][..., :10], betas[0])
        rest = np.asarray(regressor @ shaped)[:22].astype(float)
    roots = native[prefix+'transl'].astype(float) + rest[0]
    model = None
    for gap in report['gaps']:
        if gap['status'] != 'ready':
            continue
        if model is None:
            print('Loading installed Kimodo and local text encoder for bounded gaps', flush=True)
            os.environ.update(CHECKPOINT_DIR=str(args.checkpoint.parent.absolute()), HF_HUB_OFFLINE='1',
                              TRANSFORMERS_OFFLINE='1', TEXT_ENCODER_MODE='local')
            from kimodo import load_model
            model = load_model('Kimodo-SMPLX-RP-v1', device='cuda')
        print(f"Completing [{gap['start']}, {gap['end']}) with constraints {gap['constraints']}", flush=True)
        try:
            local, roots, detail = generate_gap(model, local, roots, rest, gap, args.prompt,
                                                 seed=args.seed, steps=args.steps,
                                                 diagnostic_path=args.out/f"raw_gap_{gap['start']}_{gap['end']}.npz")
            gap.update(status='completed', validation=detail)
            report['generated_frames'].extend(range(gap['start'], gap['end']))
        except CompletionRejected as error:
            gap.update(status='unresolved_rejected_candidate', reason=str(error), validation=error.report)
        save_report()
    generated = np.zeros(n, bool); generated[report['generated_frames']] = True
    valid = active & (~np.asarray(report['low_evidence']) | generated)
    report.update(unresolved_frames=np.flatnonzero(active & ~valid).tolist(),
                  model_executed=model is not None, prompt=args.prompt, seed=args.seed,
                  source_motion_outside_completed_intervals_preserved=True)
    save_report()
    # Keep a diagnostic candidate even if another interval has no usable right end.
    np.savez_compressed(args.out / 'candidate.npz', local_rot_mats=local, root_positions=roots,
                        rest_joints=rest, betas=betas, time_seconds=times,
                        generated_frame_mask=generated, motion_valid=valid, track_active=active)
    if report['unresolved_frames']:
        raise RuntimeError('Unresolved low-evidence frames block optimization; see completion.json')
    output = args.out / 'gvhmr'
    shutil.copytree(args.run, output)
    # Never use the invalid world body to condition Kimodo. Use scene cameras to
    # bring generated poses back into camera coordinates for the following lift.
    if generated.any():
        if 'scene_ground' not in manifest:
            raise ValueError('Exporting completed motion requires scene-bound GVHMR cameras')
        from tools.gvhmr.scene_ground import ground_cameras
        c2w = ground_cameras(np.load(args.run/'camera_adapter/camera_tracks.npz')['c2w'],
                             manifest['scene_ground']['prior'])
        rwc, twc = c2w[:, :3, :3], c2w[:, :3, 3]
        native[prefix+'global_orient'][generated] = Rotation.from_matrix(local[generated, 0]).as_rotvec()
        native[prefix+'body_pose'][generated] = Rotation.from_matrix(local[generated, 1:].reshape(-1, 3, 3)).as_rotvec().reshape(-1, 63)
        native[prefix+'transl'][generated] = roots[generated]-rest[0]
        incam = 'smpl_params_incam__'
        native[incam+'body_pose'][generated] = native[prefix+'body_pose'][generated]
        native[incam+'global_orient'][generated] = Rotation.from_matrix(rwc[generated].transpose(0, 2, 1) @ local[generated, 0]).as_rotvec()
        native[incam+'transl'][generated] = np.einsum('tji,tj->ti', rwc[generated], roots[generated]-twc[generated])-rest[0]
    native.update(generated_frame_mask=generated, motion_valid=valid)
    np.savez_compressed(output/'motion_native.npz', **native)
    # Preserve lifecycle split, actual original network outputs, and both exports.
    for path in [output/'hmr4d_results.pt', output/'active_inference/hmr4d_results.pt']:
        if not path.exists():
            continue
        pred = torch.load(path, map_location='cpu', weights_only=False)
        length = len(pred['smpl_params_global']['transl'])
        for group in ('smpl_params_global', 'smpl_params_incam'):
            for key in ('body_pose', 'global_orient', 'transl', 'betas'):
                pred[group][key] = torch.from_numpy(native[group+'__'+key][:length])
        pred.update(generated_frame_mask=torch.from_numpy(generated[:length]), motion_valid=torch.from_numpy(valid[:length]))
        if 'net_outputs' in pred:
            # Invalid GVHMR contact logits must not constrain generated motion.
            pred['net_outputs']['static_conf_logits'][:, generated[:length]] = -100.
            for gap in report['gaps']:
                if gap['status'] == 'completed':
                    contacts = torch.as_tensor(gap['validation']['generated_foot_contacts'])
                    pred['net_outputs']['static_conf_logits'][:, gap['start']:gap['end'], :4] = torch.where(contacts, 10., -10.)
        torch.save(pred, path)
    hashes = dict(manifest.get('output_sha256', {}))
    for path in (output/'motion_native.npz', output/'hmr4d_results.pt', output/'active_inference/hmr4d_results.pt'):
        if path.exists():
            hashes[str(path.relative_to(output))] = file_sha256(path)
    manifest.update(completion=report, original_inference_run=str(args.run.absolute()), output_sha256=hashes)
    (output/'provenance.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'masks', 'out', 'smplx-model', 'checkpoint'):
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--prompt', required=True)
    p.add_argument('--context', type=int, choices=(2, 3), default=3)
    p.add_argument('--confidence', type=float, default=.5)
    p.add_argument('--min-visible', type=int, default=5)
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--detect-only', action='store_true')
    run(p.parse_args())
