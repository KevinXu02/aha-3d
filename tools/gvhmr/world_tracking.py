"""World-postopt bbox policy for the shared source-person mask adapter."""
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np


def gate_trailing_max(boxes, valid, *, area, fps=30, window_s=1., min_frac=.5,
                      min_fill_frac=.25, min_hist=5):
    boxes = np.asarray(boxes, dtype=float)
    keep = np.asarray(valid, dtype=bool).copy()
    extent = np.maximum(boxes[:, 2:] - boxes[:, :2], 1.)
    scale = extent.max(1)
    fill = np.asarray(area, dtype=float) / extent.prod(1)
    history = []
    window = max(1, round(fps * window_s))
    for frame in range(len(keep)):
        if not keep[frame]:
            continue
        recent = [(s, f) for t, s, f in history if t >= frame - window]
        # Do not reopen warmup when older accepted frames leave the time window.
        if len(recent) < min_hist and len(history) >= min_hist:
            recent = [(s, f) for _, s, f in history[-min_hist:]]
        if len(recent) >= min_hist and (scale[frame] < min_frac * max(s for s, _ in recent)
                                       or fill[frame] < min_fill_frac * max(f for _, f in recent)):
            keep[frame] = False
        else:
            history.append((frame, scale[frame], fill[frame]))
    return keep


def track_end(keep):
    supported = np.flatnonzero(keep)
    if not len(supported):
        return 0
    end = int(supported[-1]) + 1
    return end if end < len(keep) else None


def dense_boxes(boxes, valid, helpers, area):
    import torch
    boxes = np.asarray(boxes, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if not np.isfinite(boxes[valid]).all() or np.any(boxes[valid, 2:] <= boxes[valid, :2]):
        raise ValueError('Invalid supported box')
    keep = gate_trailing_max(boxes, valid, area=area)
    if not keep.any():
        raise ValueError('No supported boxes after quality gate')
    centre = (boxes[:, :2] + boxes[:, 2:]) / 2
    size = boxes[:, 2:] - boxes[:, :2]
    values = np.concatenate([centre, size], axis=1)
    ids = np.flatnonzero(keep)
    filled = np.stack([np.interp(np.arange(len(boxes)), ids, values[ids, axis]) for axis in range(4)], axis=1)
    dense = torch.from_numpy(np.concatenate([filled[:, :2] - filled[:, 2:] / 2,
                                            filled[:, :2] + filled[:, 2:] / 2], axis=1).astype(np.float32))
    interpolated = dense.clone()
    _, net, camera = helpers
    for _ in range(2):
        dense = net.moving_average_smooth(dense, window_size=5, dim=0)
    xys = camera.get_bbx_xys_from_xyxy(dense.float(), base_enlarge=1.2).float()
    if not torch.isfinite(xys).all() or not (xys[:, 2] > 0).all():
        raise ValueError('Invalid crop conversion')
    return dense.float(), xys, interpolated


@contextmanager
def install_policy():
    """Scope injection to this fresh inference process, with truthful provenance.

    Keep the stable adapter's source, raster, lifecycle and upstream validation.
    No installed or shared source file is modified. This works on clean mainline
    as well as a checkout with independently edited default tracking code.
    """
    from tools.gvhmr import person_tracking as base
    original_prepare, original_dense = base.prepare_gvhmr_boxes, base.dense_upstream_boxes

    def prepare(mask_archive, expected_video, actor_id, repo, output_dir, **kwargs):
        with np.load(mask_archive, allow_pickle=False) as z:
            area = np.asarray(z['masks']).reshape(len(z['masks']), -1).sum(1)

        def convert(boxes, valid, helpers, **unused):
            return dense_boxes(boxes, valid, helpers, area[:len(boxes)])

        base.dense_upstream_boxes = convert
        try:
            report = original_prepare(mask_archive, expected_video, actor_id, repo, output_dir, **kwargs)
        finally:
            base.dense_upstream_boxes = original_dense
        output = Path(output_dir)
        raw_path = output / 'raw_tracking.json'
        raw = json.loads(raw_path.read_text())
        raw['interpolation_policy'] = ('World-postopt: trailing accepted maximum over 1 s, scale >= 0.5 and '
                                       'mask fill >= 0.25; centre/extent interpolation, two 5-frame averages; crop 1.2x')
        raw['world_bbox_implementation_sha256'] = base.file_sha256(__file__)
        raw_path.write_text(json.dumps(raw, indent=2, allow_nan=False) + '\n')
        with np.load(output / 'person_boxes.npz') as z:
            keep = gate_trailing_max(z['raw_bbox_xyxy'], z['mask_nonempty'], area=z['mask_area_pixels'])
        report.update(world_bbox_implementation_sha256=base.file_sha256(__file__),
                      bbox_accepted_frames=np.flatnonzero(keep).tolist(),
                      bbox_unsupported_frames=np.flatnonzero(~keep).tolist(),
                      interpolation_policy=raw['interpolation_policy'])
        report['outputs']['raw_tracking.json'] = base.file_sha256(raw_path)
        (output / 'person_boxes.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        return report

    base.prepare_gvhmr_boxes = prepare
    try:
        yield
    finally:
        base.prepare_gvhmr_boxes = original_prepare
        base.dense_upstream_boxes = original_dense
