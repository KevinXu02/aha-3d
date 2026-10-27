#!/usr/bin/env python3
"""Adapt reviewed same-shot Pi3X cameras to GVHMR's SLAM rotation input.

Pi3X and DPVO camera translations/world origins are not compared. GVHMR consumes
relative OpenCV camera rotations and retains its own estimated intrinsics here.
No camera inference, coordinate fitting, extrapolation or downloads occur.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

try:
    from .tracking_evidence import REVISION, file_sha256, validate_times, video_timeline
    from .motion_quality import stats
except ImportError:
    from tracking_evidence import REVISION, file_sha256, validate_times, video_timeline
    from motion_quality import stats

PI3X_CONVENTION = 'camera-to-world OpenCV camera axes (right,down,forward); aligned world Z up'
SLAM_CONVENTION = 'OpenCV c2w: tx,ty,tz,qx,qy,qz,qw'
SOURCE_FILES = ['tools/demo/demo.py', 'hmr4d/utils/preproc/slam.py', 'hmr4d/utils/geo_transform.py']
BUNDLE_FILES = ['inputs.json', 'cameras.json', 'camera_review.json']
OPTIONAL_BUNDLE_FILES = ['run_report.json', 'execution.json', 'bundle_validation.json']


def rigid(values, label='camera c2w'):
    poses = np.asarray(values, dtype=float)
    if poses.ndim not in (2, 3) or poses.shape[-2:] != (4, 4) or not np.isfinite(poses).all():
        raise ValueError(f'{label} must contain finite 4x4 poses')
    rot = poses[..., :3, :3]
    if not np.allclose(rot.swapaxes(-1, -2) @ rot, np.eye(3), atol=2e-4, rtol=0) or \
            not np.allclose(np.linalg.det(rot), 1., atol=2e-4, rtol=0) or \
            not np.allclose(poses[..., 3, :], [0, 0, 0, 1], atol=1e-7, rtol=0):
        raise ValueError(f'{label} must be proper rigid transforms; no reflection, scale or projective row')
    return poses


def raster(value, label):
    size = np.asarray(value)
    if size.shape != (2,) or not np.isfinite(size).all() or (size <= 0).any() or not np.equal(size, np.floor(size)).all():
        raise ValueError(f'{label} must be positive integer width,height')
    return size.astype(int)


def validate_bundle(cameras, inputs, review, target_times, image_size, source_sha256):
    times = validate_times(target_times)
    size = raster(image_size, 'Normalized video raster')
    if inputs.get('source_sha256') != source_sha256 or review.get('source_sha256') != source_sha256:
        raise ValueError('Pi3X bundle/review source hash differs from exact normalized input video')
    if inputs.get('source_frame_count') != len(times) or not np.array_equal(inputs['original_size_wh'], size):
        raise ValueError('Pi3X source raster/frame count differs from normalized video')
    processed = raster(inputs['processed_size_wh'], 'Pi3X processed raster')
    if not np.array_equal(raster(cameras['processed_size_wh'], 'Camera processed raster'), processed):
        raise ValueError('Pi3X camera/input processed rasters differ')
    if cameras.get('convention') != PI3X_CONVENTION:
        raise ValueError('Pi3X cameras must declare exact OpenCV camera-to-world convention')
    if rigid(cameras['world_transform'], 'Pi3X common world alignment').shape != (4, 4):
        raise ValueError('Pi3X world alignment must be one common rigid transform')
    if review.get('continuous_shot_reviewed') is not True or review.get('cut_frame_indices') != []:
        raise ValueError('Require explicit reviewed continuous shot without cuts; split cuts before interpolation')
    if not isinstance(review.get('reviewed_by'), str) or not review['reviewed_by'].strip() or \
            not isinstance(review.get('review_scope'), str) or not review['review_scope'].strip():
        raise ValueError('Camera review must retain reviewer and actual review scope')
    reviewed_ids = np.asarray(review['reviewed_frame_indices'])
    if reviewed_ids.ndim != 1 or not len(reviewed_ids) or reviewed_ids.dtype.kind not in 'iu' or \
            reviewed_ids.min() < 0 or reviewed_ids.max() >= len(times) or len(np.unique(reviewed_ids)) != len(reviewed_ids):
        raise ValueError('Camera review frame IDs must refer to actual source frames')
    ids = np.asarray(inputs['frame_indices'])
    if ids.ndim != 1 or len(ids) < 2 or ids.dtype.kind not in 'iu' or np.any(np.diff(ids) <= 0) or \
            ids[0] != 0 or ids[-1] != len(times) - 1:
        raise ValueError('Pi3X observation frame IDs must be increasing and include both source endpoints; no extrapolation')
    observed_times = np.asarray(inputs['timestamps_seconds'], dtype=float)
    if observed_times.shape != ids.shape or not np.isfinite(observed_times).all() or \
            not np.allclose(observed_times, times[ids], atol=1e-6, rtol=0):
        raise ValueError('Pi3X observation timestamps differ from decoded PTS at source frame IDs')
    frames = cameras['frames']
    if [row['source_frame'] for row in frames] != ids.tolist() or not np.allclose(
            [row['timestamp_seconds'] for row in frames], observed_times, atol=1e-6, rtol=0):
        raise ValueError('Pi3X camera/input frame IDs or timestamps differ')
    poses = rigid([row['c2w'] for row in frames])
    K = np.asarray([row['intrinsics'] for row in frames], dtype=float)
    if K.shape != (len(ids), 3, 3) or not np.isfinite(K).all() or (K[:, [0, 1], [0, 1]] <= 0).any() or \
            not np.allclose(K[:, 2], [0, 0, 1], atol=1e-7, rtol=0):
        raise ValueError('Invalid Pi3X intrinsics metadata, even though it is not used by GVHMR')
    # Use exact decoded source PTS for interpolation keys after validating metadata.
    return dict(time_seconds=times, image_size=size, observation_frame_indices=ids,
                observation_time_seconds=times[ids], observation_c2w=poses, observation_K=K,
                processed_size_wh=processed, world_transform=np.asarray(cameras['world_transform']))


def interpolate_c2w(poses, observed_times, target_times):
    from scipy.spatial.transform import Rotation, Slerp
    poses = rigid(poses)
    observed = np.asarray(observed_times, dtype=float)
    target = np.asarray(target_times, dtype=float)
    if observed.shape != (len(poses),) or len(observed) < 2 or not np.isfinite(observed).all() or np.any(np.diff(observed) <= 0):
        raise ValueError('Need increasing timestamped camera observations')
    if target.ndim != 1 or not len(target) or not np.isfinite(target).all() or np.any(np.diff(target) <= 0) or \
            target[0] < observed[0] or target[-1] > observed[-1]:
        raise ValueError('Camera interpolation forbids extrapolation or missing/nonmonotonic target timestamps')
    result = np.tile(np.eye(4), (len(target), 1, 1))
    result[:, :3, :3] = Slerp(observed, Rotation.from_matrix(poses[:, :3, :3]))(target).as_matrix()
    result[:, :3, 3] = np.stack([np.interp(target, observed, poses[:, axis, 3]) for axis in range(3)], axis=-1)
    return result


def fullimg_intrinsics(observation_K, observation_times, target_times, processed_size, image_size):
    """Pi3X processed-grid K -> full-image K at target times (linear, no extrapolation).

    Integer-index pixel centres are converted to boundary centres before the
    whole-image resize, matching aha3d.workflow.camera and GVHMR's estimate_K
    convention (principal point W/2, H/2 for a centred camera).
    """
    K = np.asarray(observation_K, dtype=float)
    observed = np.asarray(observation_times, dtype=float)
    target = np.asarray(target_times, dtype=float)
    if target.ndim != 1 or not len(target) or not np.isfinite(target).all() or np.any(np.diff(target) <= 0):
        raise ValueError('Target timestamps must be finite and strictly increasing')
    if K.shape != (len(observed), 3, 3) or len(observed) < 2 or np.any(np.diff(observed) <= 0):
        raise ValueError('Need increasing timestamped Pi3X intrinsics observations')
    if target[0] < observed[0] - 1e-8 or target[-1] > observed[-1] + 1e-8:
        raise ValueError('Intrinsics interpolation forbids extrapolation')
    out = np.stack([np.interp(target, observed, K[:, i, j]) for i in range(3) for j in range(3)], -1).reshape(-1, 3, 3)
    sx, sy = raster(image_size, 'Full image raster') / raster(processed_size, 'Pi3X processed raster')
    out[:, 0, 2] += .5; out[:, 1, 2] += .5
    out[:, 0, :] *= sx; out[:, 1, :] *= sy
    if not np.isfinite(out).all() or (out[:, [0, 1], [0, 1]] <= 0).any():
        raise ValueError('Invalid converted intrinsics')
    return out


def c2w_to_slam(poses):
    from scipy.spatial.transform import Rotation
    poses = rigid(poses)
    if poses.ndim != 3:
        raise ValueError('SLAM export requires a sequence of c2w poses')
    quaternions = Rotation.from_matrix(poses[:, :3, :3]).as_quat()  # SciPy defaults to XYZW.
    # Keep quaternion signs continuous; q and -q encode the same orientation.
    for index in range(1, len(quaternions)):
        if np.dot(quaternions[index - 1], quaternions[index]) < 0:
            quaternions[index] *= -1
    return np.concatenate([poses[:, :3, 3], quaternions], axis=-1)


def slam_to_c2w(track, expected_frames=None):
    from scipy.spatial.transform import Rotation
    track = np.asarray(track, dtype=float)
    if track.ndim != 2 or track.shape[1] != 7 or len(track) < 2 or not np.isfinite(track).all():
        raise ValueError('Expected finite upstream SLAM array [T,7] in tx,ty,tz,qx,qy,qz,qw order')
    if expected_frames is not None and len(track) != expected_frames:
        raise ValueError('SLAM track frame count differs from exact normalized video')
    if not np.allclose(np.linalg.norm(track[:, 3:], axis=-1), 1., atol=2e-4, rtol=0):
        raise ValueError('SLAM quaternions must have unit norm; do not silently normalize invalid data')
    result = np.tile(np.eye(4), (len(track), 1, 1))
    result[:, :3, 3] = track[:, :3]
    result[:, :3, :3] = Rotation.from_quat(track[:, 3:]).as_matrix()
    return result


def gvhmr_relative_rotations(c2w):
    poses = rigid(c2w)
    if poses.ndim != 3 or len(poses) < 2:
        raise ValueError('Need at least two full camera poses')
    w2c = poses[:, :3, :3].swapaxes(-1, -2)
    return w2c[1:] @ w2c[:-1].swapaxes(-1, -2)


def gvhmr_sixd(relative):
    """Exact upstream first-two-rows representation with final interval repeated."""
    sixd = np.asarray(relative)[:, :2].reshape(-1, 6)
    return np.concatenate([sixd, sixd[-1:]], axis=0).astype(np.float32)


def prepare_pi3x(bundle, times, image_size, source_sha256):
    bundle = Path(bundle).resolve(strict=True)
    paths = {name:bundle / name for name in BUNDLE_FILES}
    paths.update({name:bundle / name for name in OPTIONAL_BUNDLE_FILES if (bundle / name).is_file()})
    inputs, cameras, review = [json.loads(paths[name].read_text()) for name in BUNDLE_FILES]
    arrays = validate_bundle(cameras, inputs, review, times, image_size, source_sha256)
    poses = interpolate_c2w(arrays['observation_c2w'], arrays['observation_time_seconds'], arrays['time_seconds'])
    track = c2w_to_slam(poses)
    relative = gvhmr_relative_rotations(slam_to_c2w(track))
    arrays.update(c2w=poses, slam=track, relative_rotation=relative, cam_angvel_6d=gvhmr_sixd(relative),
                  K_fullimg_pi3x=fullimg_intrinsics(arrays['observation_K'], arrays['observation_time_seconds'],
                                                    arrays['time_seconds'], arrays['processed_size_wh'], arrays['image_size']))
    report = dict(schema_version=1, estimator='pi3x', source_sha256=source_sha256,
        frames=len(times), observation_frames=len(arrays['observation_frame_indices']),
        bundle=str(bundle), bundle_sha256={name:file_sha256(path) for name, path in paths.items()},
        source_convention=cameras['convention'], upstream_slam_convention=SLAM_CONVENTION,
        input_review=review, interpolation='rotation SLERP; translation linear; exact source PTS; no extrapolation',
        world_alignment='one original Pi3X rigid world transform retained; no additional axis conversion',
        translation_use='preserved for provenance; ignored by GVHMR load_data_dict',
        intrinsics_use=('Pi3X K is stored as full-image K_fullimg_pi3x but unused by default (upstream estimate_K); '
                        'person_reconstruct --intrinsics pi3x explicitly substitutes it'),
        units=cameras.get('units'), camera_accuracy_validated=False, physical_accuracy_validated=False,
        implementation_sha256=file_sha256(__file__),
        versions={name:importlib.metadata.version(name) for name in ['numpy', 'scipy']},
        limitations=['Same-shot declaration retains the actual review scope; no automatic cut detection is performed.',
                     'Sparse-camera interpolation is an estimate between observations, not a recovered dense path.',
                     'World-basis-invariant relative rotations do not validate metric scale or physical camera accuracy.',
                     'Static-landmark cross-view validation and full-rate visual review remain separate.'])
    return arrays, report


def verify_upstream(repo):
    repo = Path(repo).resolve(strict=True)
    git = ['git', '-c', f'safe.directory={repo}', '-C', str(repo)]
    revision = subprocess.run([*git, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    if revision != REVISION:
        raise ValueError('Camera adapter requires the audited GVHMR BEDLAM2 revision ' + REVISION)
    dirty = subprocess.run([*git, 'diff', 'HEAD', '--name-only', '--', *SOURCE_FILES],
                           capture_output=True, text=True, check=True).stdout.splitlines()
    if dirty:
        raise ValueError('Audited upstream camera consumer changed: ' + ', '.join(dirty))
    return dict(revision=revision, path=str(repo), sources={name:file_sha256(repo / name) for name in SOURCE_FILES})


def preload_pi3x_slam(*, repo, bundle, slam_path, output, times, image_size, source_sha256, torch_module=None):
    output, slam_path = Path(output).resolve(), Path(slam_path).resolve()
    if output.exists() or slam_path.exists():
        raise FileExistsError('Pi3X camera export and SLAM cache must be fresh')
    upstream = verify_upstream(repo)
    arrays, report = prepare_pi3x(bundle, times, image_size, source_sha256)
    if torch_module is None:
        import torch as torch_module
    output.mkdir(parents=True, exist_ok=False)
    slam_path.parent.mkdir(parents=True, exist_ok=True)
    torch_module.save(arrays['slam'], slam_path)  # Exact numpy format expected by upstream load_data_dict.
    np.savez_compressed(output / 'camera_tracks.npz', **arrays)
    for name in report['bundle_sha256']:
        shutil.copyfile(Path(bundle) / name, output / name)
    report.update(upstream=upstream, slam_path=str(slam_path), slam_sha256=file_sha256(slam_path),
                  output_sha256={'camera_tracks.npz':file_sha256(output / 'camera_tracks.npz')})
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def compare_relative_tracks(pi3x_track, dpvo_track, times):
    t = validate_times(times)
    pi3x = gvhmr_relative_rotations(slam_to_c2w(pi3x_track, len(t)))
    dpvo = gvhmr_relative_rotations(slam_to_c2w(dpvo_track, len(t)))
    difference = pi3x @ dpvo.swapaxes(-1, -2)
    angles = np.degrees(np.arccos(np.clip((np.trace(difference, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))
    arrays = dict(time_seconds=t, interval_start_seconds=t[:-1], interval_end_seconds=t[1:],
        pi3x_relative_rotation=pi3x, dpvo_relative_rotation=dpvo, relative_angle_discrepancy_degrees=angles)
    report = dict(scope='same-source relative-camera-rotation consistency', frames=len(t),
        relative_angle_discrepancy_degrees=stats(angles),
        intervals=[dict(first_frame=i, next_frame=i+1, start_seconds=float(t[i]), end_seconds=float(t[i+1]),
                        discrepancy_degrees=float(value)) for i, value in enumerate(angles)],
        absolute_world_poses_compared=False, translations_compared=False, physical_accuracy_validated=False,
        interpretation='Disagreement only; neither camera estimator is ground truth.')
    return arrays, report


def validate_dpvo_provenance(manifest, source_hash, track_hash):
    if manifest.get('camera_estimator') != 'dpvo' or manifest.get('camera_mode') != 'moving' or \
            manifest.get('normalized_input_sha256') != source_hash:
        raise ValueError('DPVO provenance must declare moving-camera DPVO on this exact normalized video')
    track_evidence = manifest.get('camera_track', {})
    if track_evidence.get('estimator') != 'dpvo' or track_evidence.get('sha256') != track_hash:
        raise ValueError('DPVO track hash must match completed launcher camera-track provenance')


def compare_dpvo(*, repo, bundle, dpvo_slam, video, output, dpvo_provenance=None):
    output, dpvo_slam, video = Path(output).resolve(), Path(dpvo_slam).resolve(strict=True), Path(video).resolve(strict=True)
    if output.exists():
        raise FileExistsError(output)
    upstream = verify_upstream(repo)
    timeline, source_hash = video_timeline(video), file_sha256(video)
    pi3x, provenance = prepare_pi3x(bundle, timeline['time_seconds'], timeline['image_size'], source_hash)
    manifest_path = Path(dpvo_provenance or dpvo_slam.parent.parent / 'provenance.json').resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    validate_dpvo_provenance(manifest, source_hash, file_sha256(dpvo_slam))
    import torch
    # Upstream saves a NumPy array. Only trusted project-generated SLAM caches are accepted here.
    dpvo = torch.load(dpvo_slam, map_location='cpu', weights_only=False)
    arrays, report = compare_relative_tracks(pi3x['slam'], dpvo, timeline['time_seconds'])
    output.mkdir(parents=True, exist_ok=False)
    arrays['pi3x_slam'] = pi3x['slam']
    arrays['dpvo_slam'] = np.asarray(dpvo)
    np.savez_compressed(output / 'relative_comparison.npz', **arrays)
    report.update(upstream=upstream, pi3x=provenance, source_sha256=source_hash,
        dpvo_slam=dict(path=str(dpvo_slam), sha256=file_sha256(dpvo_slam)),
        dpvo_provenance=dict(path=str(manifest_path), sha256=file_sha256(manifest_path)),
        output_sha256={'relative_comparison.npz':file_sha256(output / 'relative_comparison.npz')})
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['export', 'compare'])
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--pi3x-bundle', type=Path, required=True)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dpvo-slam', type=Path)
    parser.add_argument('--dpvo-provenance', type=Path)
    args = parser.parse_args()
    if args.action == 'compare':
        if args.dpvo_slam is None:
            parser.error('compare requires --dpvo-slam from a trusted project-generated DPVO run')
        report = compare_dpvo(repo=args.repo, bundle=args.pi3x_bundle, video=args.video, output=args.output,
                              dpvo_slam=args.dpvo_slam, dpvo_provenance=args.dpvo_provenance)
    else:
        if args.dpvo_slam or args.dpvo_provenance:
            parser.error('DPVO comparison arguments require compare action')
        timeline = video_timeline(args.video)
        report = preload_pi3x_slam(repo=args.repo, bundle=args.pi3x_bundle, slam_path=args.output / 'slam.pt',
            output=args.output, times=timeline['time_seconds'], image_size=timeline['image_size'],
            source_sha256=file_sha256(args.video))
    print(json.dumps(dict(output=str(args.output), frames=report['frames'], physical_accuracy_validated=False)))


if __name__ == '__main__':
    main()
