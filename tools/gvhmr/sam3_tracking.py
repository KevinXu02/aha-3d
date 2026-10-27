#!/usr/bin/env python3
"""Prompted SAM3 instance tracking, preserving source timestamps and raw masks.

The graph supplies its existing source identity; this worker does not hash assets.
Standalone runs record source path/size/mtime and are not graph-bound archives.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


def normalized_box(box, width, height):
    box = np.asarray(box, dtype=np.float32)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError('Need finite XYXY box')
    if np.any(box[:2] < 0) or np.any(box[2:] > [width, height]) or np.any(box[2:] <= box[:2]):
        raise ValueError('Prompt box must have positive extent inside the source raster')
    return box / np.asarray([width, height, width, height], dtype=np.float32)


def tracking_orders(anchor, count):
    if not 0 <= anchor < count:
        raise ValueError('Prompt frame outside source timeline')
    return [list(range(anchor, count))] + ([list(range(anchor, -1, -1))] if anchor else [])


def tracker_name(archive):
    if 'tracker' not in archive:
        raise ValueError('Mask archive requires explicit SAM3 tracker provenance')
    name = str(np.asarray(archive['tracker']).item())
    if name != 'SAM3':
        raise ValueError('Unsupported mask tracker: ' + name)
    return name


def run(a):
    import av
    import torch
    from PIL import Image
    if not torch.cuda.is_available():
        raise RuntimeError('SAM3 motion tracking requires a visible CUDA GPU')
    root = a.root.resolve(strict=True)
    sys.path.insert(0, str(root))
    from sam3.model_builder import build_sam3_video_model
    import sam3.model.sam3_tracking_predictor as predictor_module
    if not Path(predictor_module.__file__).resolve().is_relative_to(root):
        raise ValueError('SAM3 imported from a different runtime')
    source = a.video.resolve(strict=True)
    times, images = [], []
    with av.open(str(source)) as container:
        for frame in container.decode(video=0):
            if frame.pts is None:
                raise ValueError('Source frame has no presentation timestamp')
            times.append(float(frame.pts * frame.time_base))
            rgb = frame.to_ndarray(format='rgb24')
            h, w = rgb.shape[:2]
            pixels = np.array(Image.fromarray(rgb).resize((1008, 1008)))
            images.append(torch.from_numpy(pixels).permute(2, 0, 1).float().div_(255).sub_(.5).div_(.5))
    times = np.asarray(times)
    if len(times) < 2 or not np.allclose(np.diff(times), 1/30, atol=1e-5):
        raise ValueError('Motion frontend requires exact normalized 30 Hz video')
    if (w, h) != (1280, 720):
        raise ValueError('Motion frontend requires 1280x720 video')
    box = normalized_box(a.box, w, h)
    orders = tracking_orders(a.prompt_frame, len(times))
    out = a.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    if (out/'masks.npz').exists() or (out/'tracking.json').exists():
        raise FileExistsError('Tracking outputs must be fresh')
    started = time.monotonic()
    images = torch.stack(images)
    masks = np.zeros((len(times), h, w), dtype=bool)
    seen = np.zeros(len(times), dtype=bool)
    original_loader = predictor_module.load_video_frames
    try:
        for order in orders:
            # Fresh model/state in each direction: no leaked temporal memory.
            model = build_sam3_video_model(checkpoint_path=str(a.checkpoint), load_from_HF=False, compile=False)
            tracker = model.tracker
            tracker.backbone = model.detector.backbone
            tracker.eval()
            def loader(**kw):
                if Path(kw['video_path']).resolve() != source or kw['image_size'] != 1008:
                    raise ValueError('Unexpected SAM3 video request')
                return images[order], h, w
            predictor_module.load_video_frames = loader
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                state = tracker.init_state(video_path=str(source), offload_video_to_cpu=True, offload_state_to_cpu=True)
                tracker.add_new_points_or_box(state, frame_idx=0, obj_id=1, box=box, rel_coordinates=True)
                for local, ids, low, full, scores in tracker.propagate_in_video(
                        state, start_frame_idx=0, max_frame_num_to_track=len(order), reverse=False,
                        propagate_preflight=True):
                    idx = order[local]
                    if not seen[idx]:
                        masks[idx] = (full[0, 0] > 0).cpu().numpy()
                        seen[idx] = True
            del state, tracker, model
            torch.cuda.empty_cache()
    finally:
        predictor_module.load_video_frames = original_loader
    if not seen.all():
        raise ValueError('Tracker did not cover the full source timeline')
    from tools.gvhmr.person_tracking import mask_boxes
    boxes, valid, area = mask_boxes(masks)
    fields = dict(masks=masks, time_seconds=times, image_size=np.asarray([w, h]),
                  source_frame_indices=np.arange(len(times)), tracker='SAM3', actor_id=a.actor_id,
                  mask_area_pixels=area, bbox_xyxy_boundary=boxes)
    if a.source_video_sha256:
        fields['source_video_sha256'] = np.asarray(a.source_video_sha256)
    np.savez_compressed(out/'masks.npz', **fields)
    report = dict(tracker='SAM3', actor_id=a.actor_id, frames=len(times), initial_frame=a.prompt_frame,
                  initial_box_xyxy=list(map(float, a.box)), empty_frames=int((~valid).sum()),
                  source=dict(path=str(source), bytes=source.stat().st_size, mtime_ns=source.stat().st_mtime_ns),
                  checkpoint=str(a.checkpoint.resolve()), elapsed_seconds=time.monotonic()-started,
                  graph_source_identity=a.source_video_sha256,
                  model_algorithm='SAM3 prompted instance tracker; fresh forward/reversed-prefix states',
                  limitations='Raw masks only; apply world bbox policy and source-reviewed lifecycle. Masks and smooth crops do not prevent PMPose distractor contamination.')
    (out/'tracking.json').write_text(json.dumps(report, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('video', 'root', 'checkpoint', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--actor-id', required=True)
    p.add_argument('--box', nargs=4, type=float, required=True)
    p.add_argument('--prompt-frame', type=int, default=0)
    p.add_argument('--source-video-sha256', help='Existing graph identity, forwarded unchanged; no hashing here')
    run(p.parse_args())


if __name__ == '__main__':
    main()
