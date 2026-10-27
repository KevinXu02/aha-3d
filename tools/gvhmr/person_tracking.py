"""Source-bound person mask boxes for fresh GVHMR preprocessing.

This adapter never instantiates YOLO. It retains empty-mask support separately.
Size outliers are dropped, gaps filled in centre-scale, then two five-frame smooths.
Mask presence is a crop observation, not full-body visibility or identity truth.
"""
from pathlib import Path
from contextlib import contextmanager
import importlib
import json
import sys

import numpy as np
from .tracking_evidence import (REVISION, file_sha256, array_sha256, video_timeline,
                               validate_times, build_tracking_evidence)


def mask_boxes(masks):
    masks = np.asarray(masks)
    if masks.dtype != np.bool_ or masks.ndim != 3 or min(masks.shape) < 1:
        raise ValueError('Need boolean masks[T,H,W]')
    boxes = np.full((len(masks), 4), np.nan, dtype=np.float32)
    area = masks.reshape(len(masks), -1).sum(1)
    for i in np.flatnonzero(area):
        y, x = np.nonzero(masks[i])
        boxes[i] = [x.min(), y.min(), x.max() + 1, y.max() + 1]
    return boxes, area > 0, area


def validate_archive(data, times, image_size, source_sha256, actor_id):
    """Validate actual masks, not only a caller-supplied bounding-box sidecar."""
    t = validate_times(times)
    if not actor_id or str(np.asarray(data['actor_id']).item()) != actor_id:
        raise ValueError('Source-person actor binding differs')
    if str(np.asarray(data['source_video_sha256']).item()) != source_sha256:
        raise ValueError('Source-person source hash differs')
    if not np.array_equal(data['time_seconds'], t):
        raise ValueError('Source-person exact full-frame timestamps differ')
    size = np.asarray(image_size)
    if not np.array_equal(data['image_size'], size) or size.shape != (2,):
        raise ValueError('Source-person raster differs')
    masks = np.asarray(data['masks'])
    if masks.shape != (len(t), int(size[1]), int(size[0])):
        raise ValueError('Source-person full-frame mask shape differs')
    boxes, valid, area = mask_boxes(masks)
    if not valid.any():
        raise ValueError('Source-person has no nonempty mask; cannot construct crops')
    if 'bbox_xyxy_boundary' in data and not np.array_equal(data['bbox_xyxy_boundary'], boxes, equal_nan=True):
        raise ValueError('Claimed mask boxes disagree with actual masks')
    if 'mask_area_pixels' in data and not np.array_equal(data['mask_area_pixels'], area):
        raise ValueError('Claimed mask area disagrees with actual masks')
    return boxes, valid, area


def upstream_helpers(repo):
    from .camera_tracks import verify_upstream
    repo = Path(repo).resolve(strict=True)
    verify_upstream(repo)
    sys.path.insert(0, str(repo))
    try:
        seq = importlib.import_module('hmr4d.utils.seq_utils')
        net = importlib.import_module('hmr4d.utils.net_utils')
        cam = importlib.import_module('hmr4d.utils.geo.hmr_cam')
    finally:
        sys.path.pop(0)
    for module in (seq, net, cam):
        if not Path(module.__file__).resolve().is_relative_to(repo):
            raise ValueError('Imported GVHMR helper belongs to another checkout')
    return seq, net, cam


def scale_of(boxes):
    """The quantity the crop is actually sized by: max(w, h)."""
    w = np.clip(boxes[:, 2] - boxes[:, 0], 0, None)
    h = np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    return np.maximum(w, h)


def gate_trailing_max(boxes, valid, area=None, fps=30, window_s=1.0,
                      min_frac=0.5, min_fill_frac=0.25, min_hist=5):
    """Reject a box far smaller than the best of the trailing window.

    The reference is a running MAX over recently accepted frames, which is what
    makes this survive a long collapse: inside clip32's 14-frame dropout a centred
    median is itself collapsed, but the trailing window still holds the pre-collapse
    value. Only accepted frames feed the reference, so a collapse cannot lower it.

    Two quantities are gated, and a frame needs both:
      * box scale, max(w, h), against `min_frac`;
      * mask fill, area / (w*h), against `min_fill_frac` -- because a box can keep
        a plausible extent while the mask inside it is a scrap. clip32 frame 245 is
        a 95x347 box holding 220 mask pixels: it passes on scale, and if trusted it
        becomes the right-hand anchor of the interpolation and drags the bridged
        segment down. Fill and not raw area: area falls as 1/z^2 when someone walks
        away, so an absolute-area gate condemns a perfectly good receding track
        (measured on clip42: 26 good frames rejected, heights 405 down to 372).
    """
    n = len(valid)
    s = scale_of(boxes)
    if area is None:
        area = np.where(valid, np.maximum(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1.0), 0.0)
    area = np.asarray(area, dtype=float)
    w = np.clip(boxes[:, 2] - boxes[:, 0], 1, None)
    h = np.clip(boxes[:, 3] - boxes[:, 1], 1, None)
    fill = area / (w * h)
    win = max(1, int(round(window_s * fps)))
    keep = np.asarray(valid).copy()
    acc = []                                   # (frame, scale, fill) of accepted frames
    for t in range(n):
        if not keep[t]:
            continue
        hist = [(v, f_) for f, v, f_ in acc if f >= t - win]
        if len(hist) < min_hist:
            # Too few accepted frames remain in the window. Use the most recent
            # accepted frames rather than starting fresh -- resetting here would
            # wave through the tail of a collapse, which is exactly where the worst
            # boxes are (clip42 let a 44px scrap past that way).
            hist = [(v, f_) for _, v, f_ in acc[-min_hist:]]
        if len(hist) < min_hist:
            acc.append((t, s[t], fill[t])); continue
        if (s[t] < min_frac * max(v for v, _ in hist)
                or fill[t] < min_fill_frac * max(f_ for _, f_ in hist)):
            keep[t] = False                    # rejected, and not fed to the reference
        else:
            acc.append((t, s[t], fill[t]))
    return keep


def track_end(keep, fps=30, min_gap_s=1.0):
    """Where tracking stops: the start of the final run of unusable frames.

    Rejections that are followed by a real recovery are an interior dropout and
    get interpolated. Only the trailing unsupported suffix determines this bound;
    fps and min_gap_s are retained for compatibility and do not affect it.
    """
    n = len(keep)
    good = np.flatnonzero(keep)
    if not len(good):
        return 0
    last = int(good[-1])
    if last + 1 >= n:
        return None
    return last + 1


def interpolate_centre_scale(boxes, valid):
    """Fill unsupported frames by interpolating centre and size, not corners.

    Linear interpolation of centre and extent is equivalent to interpolating
    corners with the same support. Endpoints are held rather than extrapolated.
    """
    import torch
    n = len(valid)
    idx = np.flatnonzero(valid)
    cx = (boxes[idx, 0] + boxes[idx, 2]) / 2
    cy = (boxes[idx, 1] + boxes[idx, 3]) / 2
    w = boxes[idx, 2] - boxes[idx, 0]
    h = boxes[idx, 3] - boxes[idx, 1]
    t = np.arange(n, dtype=np.float64)
    f = lambda v: np.interp(t, idx, v)          # np.interp holds the endpoints
    cx, cy, w, h = f(cx), f(cy), f(w), f(h)
    dense = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return torch.from_numpy(dense.astype(np.float32))


def dense_upstream_boxes(boxes, valid, helpers, *, area=None, base_enlarge=1.2,
                         fps=30, window_s=1.0, min_frac=0.5, min_fill_frac=0.25):
    """Dense crop boxes for GVHMR.

    Upstream interpolated the corners and enlarged by 1.2. Here, a box
    smaller than half the best box of the previous second is treated as
    unsupported, and gaps are filled in centre-scale space. Note the crop is
    square -- `get_bbx_xys` fits the box to a 192:256 ratio
    and then takes max(w, h) * base_enlarge -- so for an upright person the side
    is essentially the box height times this factor.
    """
    import torch
    seq, net, cam = helpers
    boxes, valid = np.asarray(boxes, np.float32), np.asarray(valid)
    if valid.dtype != np.bool_ or boxes.shape != (len(valid), 4) or not valid.any():
        raise ValueError('Need valid source mask-box support')
    if not np.isfinite(boxes[valid]).all() or np.any(boxes[valid, 2:] <= boxes[valid, :2]):
        raise ValueError('Invalid supported box')
    kept = gate_trailing_max(boxes, valid, area=area, fps=fps, window_s=window_s,
                             min_frac=min_frac, min_fill_frac=min_fill_frac)
    if not kept.any():
        raise ValueError('Trailing-max gate removed every supported box')
    dense = interpolate_centre_scale(boxes, kept)
    interpolated = dense.clone()
    for _ in range(2):
        dense = net.moving_average_smooth(dense, window_size=5, dim=0)
    dense = dense.float()
    xys = cam.get_bbx_xys_from_xyxy(dense, base_enlarge=base_enlarge).float()
    if not torch.isfinite(xys).all() or not (xys[:, 2] > 0).all():
        raise ValueError('Upstream crop conversion returned invalid boxes')
    return dense, xys, interpolated


def prepare_gvhmr_boxes(mask_archive, expected_video, actor_id, repo, output_dir, *, track_active=None, lifecycle=None):
    """Create fresh bbx.pt and raw_tracking.json; caller owns a fresh run."""
    import torch
    output = Path(output_dir).resolve()
    targets = [output/'bbx.pt', output/'raw_tracking.json', output/'person_boxes.npz', output/'person_boxes.json']
    if any(p.exists() for p in targets):
        raise FileExistsError('Source-person injected box outputs must be fresh')
    video, archive = Path(expected_video).resolve(strict=True), Path(mask_archive).resolve(strict=True)
    timeline = video_timeline(video)
    digest = file_sha256(video)
    with np.load(archive, allow_pickle=False) as data:
        from .sam3_tracking import tracker_name
        tracker = tracker_name(data)
        boxes, valid, area = validate_archive(data, timeline['time_seconds'], timeline['image_size'], digest, actor_id)
        archive_active = data['track_active'] if 'track_active' in data else None
        archive_mapping = data['source_frame_indices'] if 'source_frame_indices' in data else None
    from .track_lifecycle import track_active_mask, pad_inactive, source_frame_mapping
    active = track_active_mask(track_active, len(valid))
    source_frame_mapping(archive_mapping, len(valid))
    if archive_active is not None and not np.array_equal(track_active_mask(archive_active,len(valid)),active):
        raise ValueError('Source-person archive lifecycle differs from explicit reviewed lifecycle')
    end = int(active.sum())
    original_valid = valid.copy()
    valid &= active
    helpers = upstream_helpers(repo)
    dense, xys, interpolated = dense_upstream_boxes(boxes[:end], valid[:end], helpers, area=area[:end])
    full_dense = pad_inactive(dense.numpy(), active, fill='nan')
    history = [[dict(id=1, bbx_xyxy=box.tolist())] if ok else [] for box, ok in zip(boxes, valid)]
    raw = build_tracking_evidence(history, timeline['time_seconds'], timeline['image_size'],
        source_path=video, source_sha256=digest, selected_track_id=1, actor_id=actor_id)
    raw.update(selection_policy=tracker.lower()+'_mask_actor', selected_track_id=1,
        tracker=tracker, prompt_object_id=1, identity_review='explicit_reviewed_actor_initialization',
        observation_origin='actual full-rate '+tracker+' masks; no YOLO tracking or prior actor feature reuse',
        raw_box_confidence='not a detector confidence; nonempty predicted '+tracker+' mask support only',
        detected_semantics='nonempty predicted mask crop; not whole-body visibility or physical occlusion',
        mask_archive=dict(path=str(archive), sha256=file_sha256(archive)),
        status='dense_boxes_returned', dense_bbx_xyxy_sha256=array_sha256(full_dense),
        track_active=active.tolist(), source_frame_indices=list(range(len(active))), lifecycle=lifecycle,
        interpolation_policy='boxes below 50% of the trailing 1s maximum scale, or 25% of its maximum mask fill ratio, dropped as unsupported, then centre-scale linear interpolation with endpoint hold within the active prefix, then two 5-frame moving averages; no boxes after terminal exit',
        crop_policy='get_bbx_xys_from_xyxy(base_enlarge=1.2); square crop, side = max(w,h) after the 192:256 fit',
        instrumentation_sha256=file_sha256(__file__),
        limitations=['Predicted segmentation and bounds may truncate occluded/out-of-frame body parts.',
                    'Nonempty masks do not establish physical visibility or correct identity.',
                    'Empty-mask frames retain explicit unsupported status; filled crops are not observations.'])
    output.mkdir(parents=True, exist_ok=True)
    torch.save(dict(bbx_xyxy=dense, bbx_xys=xys), targets[0])
    targets[1].write_text(json.dumps(raw, indent=2, allow_nan=False)+'\n')
    np.savez_compressed(targets[2], time_seconds=timeline['time_seconds'], image_size=timeline['image_size'],
        source_video_sha256=np.asarray(digest), actor_id=np.asarray(actor_id), raw_bbox_xyxy=boxes,
        raw_mask_nonempty=original_valid, mask_nonempty=valid, mask_area_pixels=area,
        track_active=active, source_frame_indices=np.arange(len(active)),
        interpolated_bbox_xyxy=pad_inactive(interpolated.numpy(), active, fill='nan'),
        smoothed_bbox_xyxy=full_dense, bbx_xys=pad_inactive(xys.numpy(), active, fill='nan'))
    report = dict(schema_version=1, tracker=tracker, actor_id=actor_id, source_video_sha256=digest,
        frames=len(valid), nonempty_frames=int(valid.sum()), unsupported_frames=np.flatnonzero(~valid).tolist(),
        active_frames=end, terminal_exit_frame=end if end<len(active) else None,
        inactive_boxes_policy='No inferred/extrapolated boxes; full-timeline sidecar rows NaN, actual bbx.pt contains active prefix only',
        default_tracker_invoked=False, mask_archive_sha256=file_sha256(archive),
        injected_bbx_file_sha256=file_sha256(targets[0]), dense_bbx_xyxy_sha256=array_sha256(full_dense),
        implementation_sha256=file_sha256(__file__), upstream_revision=REVISION,
        helper_sources={str(Path(m.__file__).relative_to(Path(repo).resolve())):file_sha256(m.__file__) for m in helpers},
        observation_semantics=raw['detected_semantics'], outputs={p.name:file_sha256(p) for p in targets[:3]})
    targets[3].write_text(json.dumps(report, indent=2)+'\n')
    return report


@contextmanager
def forbid_default_tracker(tracker_class):
    """Fail closed if the purported injected path attempts default tracking."""
    original = tracker_class.__init__
    def forbidden(self, *args, **kwargs):
        raise RuntimeError('Default Tracker invoked despite source-person box injection')
    tracker_class.__init__ = forbidden
    try:
        yield
    finally:
        tracker_class.__init__ = original
