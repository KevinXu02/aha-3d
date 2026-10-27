#!/usr/bin/env python3
"""Dense per-frame metric cameras: one Pi3X forward pass over *all* frames.

Why not the existing bundle: the reviewed Pi3X bundles hold 32 observations,
which camera_tracks.py SLERP/linear-interpolates to full rate. The post-optimiser
observes the SLAM scale only through camera motion (contact velocity vs dT_wc),
and interpolation removes exactly the high-frequency part of that motion.

DPVO also works and is far cheaper (6s for 260 frames) -- see run_dpvo.py; it needs
`PYTHONPATH=$GVHMR_ROOT/third-party/DPVO` because the built wheel omits the
`dpvo.loop_closure` namespace, as docs/install/gvhmr.md states. Both cameras were
run through the full A/B on clip32 and give the same conclusions. Dense Pi3X is
preferred here only because it reproduces the reviewed observations to ~1mm/0.04
deg against DPVO's ~87mm/2.0deg, so it stays inside the project's reviewed world.

Why one pass rather than sliding windows: the project's own capacity benchmark
(external/Pi3/benchmark_pi3x.json, 80GB A800) reaches N=1024 frames at
336x504 / 864 patches per image. Our clips are 243-429 frames at 48x27=1296
patches, i.e. ~0.4-0.6M patches against the ~0.9M the benchmark sustained, so a
whole clip fits in a single globally consistent pass on the 97GB cards. Window
stitching would add per-window Sim(3) drift for no benefit.

Preprocessing mirrors the bundle's own executed_reconstruct_source.py exactly
(same pixel limit, same /14 patch sizing, same PIL LANCZOS resize, same
image-only Pi3X and dtype) so the dense run is comparable to the reviewed one.

The raw Pi3X world is arbitrary (first camera at identity, own scale), so the
dense cameras are mapped into the project's *reviewed* floor-aligned Z-up world
by a Umeyama fit against that bundle's observation poses. That inherits the
reviewed floor alignment instead of re-deriving one.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


def umeyama(A, B):
    A, B = np.asarray(A, float), np.asarray(B, float)
    mu_a, mu_b = A.mean(0), B.mean(0)
    Ac, Bc = A - mu_a, B - mu_b
    U, D, Vt = np.linalg.svd(Ac.T @ Bc / len(A))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = Vt.T @ S @ U.T
    s = float((D * np.diag(S)).sum() / ((Ac ** 2).sum() / len(A)))
    return s, R, mu_b - s * R @ mu_a


def processed_size(ow, oh, pixel_limit):
    """Identical sizing rule to the bundle's executed_reconstruct_source.py."""
    ratio = math.sqrt(pixel_limit / (ow * oh))
    tw, th = ow * ratio, oh * ratio
    kw, kh = max(1, round(tw / 14)), max(1, round(th / 14))
    while kw * kh * 196 > pixel_limit:
        if kw / kh > tw / th and kw > 1:
            kw -= 1
        elif kh > 1:
            kh -= 1
        else:
            kw -= 1
    return kw * 14, kh * 14


def load_reference(path):
    """Reviewed observations, from a camera_tracks.npz or a raw Pi3X cameras.json.

    Runs that fed Pi3X into GVHMR have a camera_adapter/camera_tracks.npz with the
    observations already interpolated to full rate. Runs that did not (clip31 used
    DPVO) still have the reviewed Pi3X bundle, which carries the same observation
    poses; only the interpolated `c2w` has to be stood in for.
    """
    path = Path(path)
    if path.suffix == '.npz':
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    cams = json.loads(path.read_text())
    frames = cams['frames']
    obs_c2w = np.array([f['c2w'] for f in frames], dtype=np.float64)
    obs_idx = np.array([f['source_frame'] for f in frames], dtype=np.int64)
    K = np.array([f['intrinsics'] for f in frames], dtype=np.float64)
    n = int(obs_idx.max()) + 1
    # placeholder full-rate track; replaced by the dense poses below
    c2w = np.tile(np.eye(4), (n, 1, 1))
    c2w[obs_idx] = obs_c2w
    w, h = cams['processed_size_wh']
    return {'c2w': c2w, 'observation_frame_indices': obs_idx, 'observation_c2w': obs_c2w,
            'observation_K': K, 'processed_size_wh': np.array([w, h], dtype=np.int64),
            'image_size': np.array(cams.get('original_size_wh', [w, h]), dtype=np.int64),
            'world_transform': np.array(cams['world_transform'], dtype=np.float64),
            'time_seconds': np.array([f['timestamp_seconds'] for f in frames], dtype=np.float64)}


def dense_track_payload(ref, c2w, intrinsics, processed_wh):
    """Update every camera representation together; never retain sparse SLAM."""
    from scipy.spatial.transform import Rotation
    n = len(c2w)
    times = np.asarray(ref['time_seconds'])
    if times.shape != (n,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Dense export requires exact full-source frame timestamps')
    if intrinsics.shape != (n, 3, 3):
        raise ValueError('Dense intrinsics must cover every camera frame')
    w2c = c2w[:, :3, :3].transpose(0, 2, 1)
    relative = w2c[1:] @ w2c[:-1].transpose(0, 2, 1)
    sixd = relative[:, :2].reshape(-1, 6)
    payload = dict(ref)
    payload.update(c2w=c2w,
        slam=np.concatenate([c2w[:, :3, 3], Rotation.from_matrix(c2w[:, :3, :3]).as_quat()], axis=1),
        relative_rotation=relative, cam_angvel_6d=np.concatenate([sixd, sixd[-1:]]).astype(np.float32),
        intrinsics=intrinsics, dense_processed_size_wh=np.asarray(processed_wh),
        source_frame_indices=np.arange(n),
        camera_source=np.asarray('pi3x_dense_single_pass + umeyama_to_reviewed_bundle'))
    return payload


def save_dense_predictions(folder, predictions, *, intrinsics, non_edge, c2w,
                           scale, rotation, translation, times, image_size):
    """Persist all raw model outputs plus matched aligned axial depth/cameras.

    NPY sidecars permit frame-wise mmap access for every actor without loading a
    whole clip. Raw model outputs are unchanged; depth alone receives the same
    similarity scale as aligned camera translations. Body scale is unrelated.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    for key, value in predictions.items():
        np.save(folder / (key + '.npy'), value, allow_pickle=False)
    n = len(c2w)
    lp = predictions['local_points']
    if lp.ndim != 4 or lp.shape[0] != n or lp.shape[-1] != 3:
        raise ValueError('Expected one local point map per source frame')
    fields = dict(depth_m=lp[..., 2] * scale, intrinsics=intrinsics,
                  non_edge=non_edge, c2w_aligned=c2w, time_seconds=times,
                  source_frame_indices=np.arange(n), image_size_wh=image_size,
                  processed_size_wh=[lp.shape[2], lp.shape[1]],
                  raw_to_aligned_scale=scale, raw_to_aligned_rotation=rotation,
                  raw_to_aligned_translation=translation)
    for key, value in fields.items():
        np.save(folder / (key + '.npy'), value, allow_pickle=False)
    report = dict(schema_version=1, frames=n, raw_model_outputs=list(predictions),
                  depth='depth_m.npy: axial camera Z in aligned scene-reference units',
                  raw_geometry='points/local_points/camera_poses retain raw Pi3X metric coordinates',
                  confidence='conf.npy contains logits; apply sigmoid once',
                  camera='c2w_aligned.npy: OpenCV camera-to-world in reviewed bundle basis',
                  intrinsics='intrinsics.npy: recovered from this same forward pass at processed raster',
                  scale=float(scale), complete=True)
    (folder / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--video', required=True, help='the exact normalized clip GVHMR consumed')
    ap.add_argument('--upstream', required=True, help='external/Pi3 checkout')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--reference-tracks', required=True,
                    help='camera_adapter/camera_tracks.npz, or a reviewed Pi3X cameras.json')
    ap.add_argument('--out', required=True)
    ap.add_argument('--pixel-limit', type=int, default=255000)
    ap.add_argument('--max-frames', type=int, default=0, help='0 = all frames')
    ap.add_argument('--masks', help='SAM3 masks.npz; also emit per-frame subject depth')
    a = ap.parse_args()
    out = Path(a.out)
    predictions_dir = out.with_name(out.stem + '-predictions')
    if out.exists() or predictions_dir.exists():
        raise FileExistsError('Use a fresh output; camera and depth caches are immutable')

    import cv2
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    sys.path.insert(0, str(Path(a.upstream).resolve()))
    from pi3.models.pi3x import Pi3X
    from pi3.utils.geometry import recover_intrinsic_from_rays_d, depth_normal_edge

    ref = load_reference(a.reference_tracks)
    n_expected = len(ref['c2w'])

    cap = cv2.VideoCapture(a.video)
    ow, oh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width, height = processed_size(ow, oh, a.pixel_limit)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).resize(
            (width, height), Image.Resampling.LANCZOS)
        frames.append(np.asarray(im))
    cap.release()
    if a.max_frames:
        frames = frames[:a.max_frames]
    rgb = np.stack(frames)
    n = len(rgb)
    print(f'decoded {n} frames at {width}x{height} '
          f'({width//14}x{height//14}={width*height//196} patches/img, '
          f'{n*width*height//196} total)', flush=True)
    if n != n_expected:
        raise ValueError(f'{n} decoded frames vs {n_expected} in the reference track')

    model = Pi3X(use_multimodal=False).eval()
    mism = model.load_state_dict(load_file(a.checkpoint), strict=False)
    allowed = ('depth_encoder.', 'depth_emb', 'ray_embed.', 'pose_inject_blk.')
    if mism.missing_keys or any(not k.startswith(allowed) for k in mism.unexpected_keys):
        raise ValueError('checkpoint does not match the image-only Pi3X model')
    model = model.cuda()

    imgs = torch.from_numpy(rgb.copy()).permute(0, 3, 1, 2).float().div_(255)[None].cuda()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.inference_mode(), torch.autocast('cuda', dtype=dtype):
        res = model(imgs)
    torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    # Per-frame subject depth, while the dense prediction is still on the GPU.
    # This is the one quantity that survives a collapsed mask: it depends on
    # *where* the mask samples, not on how much of the body it covers, so a
    # legs-only mask still reports the right distance. Saving z(t) rather than
    # the point maps keeps it to a few kB instead of ~1 GB per clip.
    subject_z = None
    if a.masks:
        import cv2
        mk = np.load(a.masks, allow_pickle=True)['masks']
        lp = res['local_points'].detach().float()[0]                 # (N,h,w,3)
        cf = torch.sigmoid(res['conf'].detach().float()[0, ..., 0])   # (N,h,w)
        hh, ww = lp.shape[1:3]
        subject_z = np.full(n, np.nan)
        for i in range(min(n, len(mk))):
            m = cv2.resize(mk[i].astype(np.uint8), (ww, hh),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
            sel = torch.from_numpy(m).to(lp.device) & (cf[i] > 0.5)
            if int(sel.sum()) >= 20:
                subject_z[i] = float(lp[i][sel][:, 2].median())
        got = int(np.isfinite(subject_z).sum())
        print(f'subject depth: {got}/{n} frames, '
              f'median {np.nanmedian(subject_z):.2f} m', flush=True)

    raw_c2w = res['camera_poses'].detach().float().cpu().numpy()[0]
    with torch.inference_mode():
        intrinsic = recover_intrinsic_from_rays_d(res['rays'], force_center_principal_point=True)
        non_edge = ~depth_normal_edge(res['local_points'], rtol=.03,
                                      mask=torch.sigmoid(res['conf'][..., 0]) > .1)
    intrinsic = intrinsic.detach().float().cpu().numpy()[0]
    non_edge = non_edge.cpu().numpy()[0]
    if not np.isfinite(raw_c2w).all():
        raise ValueError('nonfinite camera poses')
    print(f'Pi3X forward: {secs:.1f}s, peak {peak_gb:.1f} GB', flush=True)

    # map the dense raw cameras into the reviewed, floor-aligned world
    obs_idx = np.asarray(ref['observation_frame_indices'], dtype=int)
    obs_c2w = np.asarray(ref['observation_c2w'], dtype=np.float64)
    keep = obs_idx < n
    s, R, t = umeyama(raw_c2w[obs_idx[keep], :3, 3], obs_c2w[keep, :3, 3])
    dense_t = s * (R @ raw_c2w[:, :3, 3].T).T + t
    dense_R = R[None] @ raw_c2w[:, :3, :3]
    resid = np.linalg.norm(dense_t[obs_idx[keep]] - obs_c2w[keep, :3, 3], axis=1)
    rel = np.matmul(dense_R[obs_idx[keep]].transpose(0, 2, 1), obs_c2w[keep, :3, :3])
    rot_err = np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1)))

    c2w = np.tile(np.eye(4), (n, 1, 1))
    c2w[:, :3, :3], c2w[:, :3, 3] = dense_R, dense_t
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = {key: value.detach().float().cpu().numpy()[0] for key, value in res.items()}
    dense_report = save_dense_predictions(predictions_dir, raw, intrinsics=intrinsic,
        non_edge=non_edge, c2w=c2w, scale=s, rotation=R, translation=t,
        times=ref['time_seconds'], image_size=[ow, oh])
    payload = dense_track_payload(ref, c2w, intrinsic, [width, height])
    payload['dense_predictions'] = np.asarray(str(predictions_dir.resolve()))
    if subject_z is not None:
        payload['subject_depth_m'] = subject_z * s
        payload['focal_processed_px'] = intrinsic[:, 0, 0]
    np.savez(out, **payload)

    interp_t = np.asarray(ref['c2w'], dtype=np.float64)[:n, :3, 3]
    report = {
        'frames': int(n), 'processed_size_wh': [width, height],
        'forward_seconds': secs, 'peak_vram_gb': peak_gb,
        'alignment_scale_raw_to_reviewed': s,
        'observations_used': int(keep.sum()),
        'translation_residual_m': {'rms': float(np.sqrt((resid ** 2).mean())),
                                   'median': float(np.median(resid)), 'max': float(resid.max())},
        'rotation_disagreement_deg': {'median': float(np.median(rot_err)), 'max': float(rot_err.max())},
        'path_length_m': {'reviewed_interpolated': float(np.linalg.norm(np.diff(interp_t, axis=0), axis=1).sum()),
                          'pi3x_dense': float(np.linalg.norm(np.diff(dense_t, axis=0), axis=1).sum())},
        'written': str(out),
        'dense_predictions': str(predictions_dir),
        'depth_export': dense_report,
    }
    Path(str(out) + '.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
