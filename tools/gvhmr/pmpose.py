#!/usr/bin/env python3
"""Mask-conditioned COCO-17 extraction in the separate BBoxMaskPose runtime."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def add_arguments(parser):
    parser.add_argument('--pose-detector', choices=('pmpose', 'vitpose'), default='pmpose')
    parser.add_argument('--pmpose-python', type=Path, default=Path(os.environ.get('PMPOSE_PYTHON', ROOT / '.runtime/bmp/venv/bin/python')))
    parser.add_argument('--pmpose-root', type=Path, default=Path(os.environ.get('BMP_ROOT', ROOT / '.runtime/bmp/src')))
    parser.add_argument('--pmpose-checkpoint', type=Path, default=Path(os.environ.get('PMPOSE_CHECKPOINT', ROOT / '.runtime/bmp/checkpoints/PMPose-h-1.0.0.pth')))
    parser.add_argument('--pmpose-variant', choices=('PMPose-h', 'PMPose-b'), default='PMPose-h')
    parser.add_argument('--pmpose-ld-preload', default=os.environ.get('PMPOSE_LD_PRELOAD', ''))


def extract(a, video, boxes, output, frames):
    if a.person_masks is None:
        raise ValueError('PMPose requires source-person masks; use --pose-detector vitpose for the explicit YOLO control')
    env = os.environ.copy()
    if a.pmpose_ld_preload:
        env['LD_PRELOAD'] = a.pmpose_ld_preload
    argv = [str(a.pmpose_python.absolute()), str(Path(__file__).resolve()),
            '--video', str(video), '--boxes', str(boxes), '--masks', str(a.person_masks),
            '--output', str(output), '--frames', str(frames),
            '--root', str(a.pmpose_root.absolute()), '--checkpoint', str(a.pmpose_checkpoint.absolute()),
            '--variant', a.pmpose_variant]
    subprocess.run(argv, env=env, check=True)
    return json.loads(Path(str(output) + '.json').read_text())


def coco17(prediction):
    import numpy as np
    value = np.asarray(prediction, dtype=np.float32)
    if value.shape != (23, 3) or not np.isfinite(value).all():
        raise ValueError(f'Expected finite merged COCO/AIC/MPII (23,3), got {value.shape}')
    if (value[:, 2] < 0).any() or (value[:, 2] > 1).any():
        raise ValueError('PMPose confidence outside [0,1]')
    return value[:17].copy()


def worker(a):
    import cv2
    import numpy as np
    import torch
    for key in ('video', 'boxes', 'masks', 'output', 'root', 'checkpoint'):
        setattr(a, key, getattr(a, key).absolute())
    if a.output.exists():
        raise FileExistsError(a.output)
    boxes = torch.load(a.boxes, map_location='cpu', weights_only=True)['bbx_xyxy'].numpy()
    with np.load(a.masks, allow_pickle=False) as z:
        masks = z['masks']
    if boxes.shape != (a.frames, 4) or len(masks) < a.frames:
        raise ValueError('Boxes/masks do not cover the active source prefix')
    sys.path.insert(0, str(a.root)); os.chdir(a.root / 'mmpose')
    import mmpretrain  # Registers the PMPose backbone.
    original = torch.load
    try:
        torch.load = lambda *args, **kw: original(*args, **{**kw, 'weights_only': False})
        from pmpose import PMPose
        model = PMPose(device='cuda', variant=a.variant)
        model.load_from_file(str(a.checkpoint))
    finally:
        torch.load = original
    cap = cv2.VideoCapture(str(a.video))
    if not cap.isOpened():
        raise ValueError('Cannot decode source video')
    kp = np.zeros((a.frames, 17, 3), dtype=np.float32)
    empty = []; start = time.monotonic()
    try:
        for t in range(a.frames):
            ok, frame = cap.read()
            if not ok or masks[t].shape != frame.shape[:2]:
                raise ValueError(f'Frame/mask raster mismatch at {t}')
            box = boxes[t]
            if not np.isfinite(box).all() or (box[2:] <= box[:2]).any():
                raise ValueError(f'Invalid active bbox at {t}')
            if not masks[t].any():
                empty.append(t)
                continue  # No identity evidence: zero confidence, never invent a mask.
            result = model.predict(frame, bboxes=box[None], masks=[masks[t]], return_probmaps=False)
            kp[t] = coco17(result[0][0])
            if t % 50 == 0:
                print(f'PMPose {t}/{a.frames}', flush=True)
    finally:
        cap.release()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(kp), a.output)
    meta = dict(detector='pmpose', variant=a.variant, checkpoint=str(a.checkpoint),
                masks=str(a.masks), boxes=str(a.boxes), video=str(a.video),
                frames=a.frames, source_frame_indices=list(range(a.frames)),
                empty_mask_frames=empty, seconds=time.monotonic()-start,
                mean_confidence=float(kp[..., 2].mean()), python=sys.executable,
                torch=torch.__version__, numpy=np.__version__,
                layout='COCO-17 full-image pixels and confidence; first 17 of 23 merged joints',
                mask_policy='Actual source masks, including partial masks; empty masks produce zero confidence')
    Path(str(a.output) + '.json').write_text(json.dumps(meta, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('video', 'boxes', 'masks', 'output', 'root', 'checkpoint'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--frames', type=int, required=True)
    p.add_argument('--variant', choices=('PMPose-h', 'PMPose-b'), default='PMPose-h')
    worker(p.parse_args())
