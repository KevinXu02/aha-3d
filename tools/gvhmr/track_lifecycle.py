"""Reviewed terminal-exit binding, bounded tracking and inactive storage padding."""
from pathlib import Path
import json
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from aha3d.motion.lifecycle import track_active_mask, source_frame_mapping
from .tracking_evidence import file_sha256, validate_times


def normalized_source_times(times):
    times = validate_times(times)
    if abs(times[0]) > 1e-8:
        raise ValueError('Lifecycle inference requires zero-origin normalized 30 Hz source PTS')
    return times


def load_lifecycle(path, *, times, image_size, source_sha256, actor_id):
    times = normalized_source_times(times)
    if path is None:
        return np.ones(len(times), bool), dict(policy='legacy_all_active', terminal_exit_frame=None)
    path = Path(path).resolve(strict=True)
    data = json.loads(path.read_text())
    if data.get('schema_version') != 1 or data.get('source_video_sha256') != source_sha256 or data.get('actor_id') != actor_id:
        raise ValueError('Lifecycle review source/actor/schema differs')
    if not np.array_equal(data.get('time_seconds'), times) or not np.array_equal(data.get('image_size'), image_size):
        raise ValueError('Lifecycle exact timestamps/raster differ')
    if not str(data.get('reviewer', '')).strip() or not str(data.get('reason', '')).strip():
        raise ValueError('Lifecycle requires reviewer and evidence reason')
    frames = np.asarray(data.get('evidence_frames', []))
    if frames.ndim != 1 or frames.dtype.kind not in 'iu' or not len(frames) or np.any((frames < 0) | (frames >= len(times))):
        raise ValueError('Lifecycle needs valid supporting source frames')
    exit_frame = data.get('terminal_exit_frame')
    active = np.ones(len(times), bool)
    if exit_frame is not None:
        if type(exit_frame) is not int or not 1 <= exit_frame < len(times):
            raise ValueError('Terminal exit must be an interior integer first-absent frame')
        if not np.any(frames >= exit_frame) or not np.any(frames < exit_frame):
            raise ValueError('Terminal exit review needs evidence before and after exit')
        active[exit_frame:] = False
    return active, dict(policy='source_reviewed_terminal_exit', path=str(path), sha256=file_sha256(path),
        terminal_exit_frame=exit_frame, reviewed_by=data['reviewer'], reason=data['reason'],
        scope='Missing observations never terminate a track automatically; inactive tail is not estimated motion.')


def edge_exit_proposal(masks, *, min_absent_frames=5, edge_margin_px=2):
    """Advisory proposal only: edge contact followed by a terminal absent suffix.

    An occluder at an image boundary can produce the same evidence. A source
    review must confirm exit before this proposal may drive track_active.
    """
    from .person_tracking import mask_boxes
    boxes, present, _ = mask_boxes(masks)
    if type(min_absent_frames) is not int or min_absent_frames < 2 or edge_margin_px < 0:
        raise ValueError('Invalid exit proposal evidence bounds')
    ids = np.flatnonzero(present)
    if not len(ids):
        return dict(candidate=None, requires_review=True, reason='No established visible mask')
    last = int(ids[-1]); first_absent = last + 1
    box=boxes[last]; h,w=np.asarray(masks).shape[1:]
    edge = bool(box[0] <= edge_margin_px or box[1] <= edge_margin_px or box[2] >= w-edge_margin_px or box[3] >= h-edge_margin_px)
    candidate = first_absent if edge and len(present)-first_absent >= min_absent_frames else None
    return dict(candidate=candidate, requires_review=True, last_mask_frame=last, edge_contact=edge,
        absent_suffix_frames=len(present)-first_absent,
        reason='Edge mask followed by terminal absence; cannot distinguish boundary occlusion from exit automatically.')


def pad_inactive(array, active, *, fill='last'):
    """Return full timeline; inactive padding is never a model prediction."""
    active=track_active_mask(active, len(active)); array=np.asarray(array); end=int(active.sum())
    if array.ndim < 1 or len(array) != end:
        raise ValueError('Array must contain exactly the inferred active prefix')
    result=np.empty((len(active),)+array.shape[1:],dtype=array.dtype); result[:end]=array
    if fill == 'last': result[end:]=array[-1]
    elif fill == 'nan':
        if result.dtype.kind != 'f': raise ValueError('NaN padding requires floating point')
        result[end:]=np.nan
    elif fill == 'zero': result[end:]=0
    else: raise ValueError('Unknown inactive padding policy')
    return result


def write_lossless_prefix(video, output, active):
    """Exact decoded RGB prefix in FFV1 AVI, with explicit original-frame mapping."""
    import av
    from fractions import Fraction
    active=track_active_mask(active,len(active)); end=int(active.sum())
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    digest=__import__('hashlib').sha256(); count=0
    with av.open(str(video)) as src, av.open(str(output),'w') as dst:
        stream=dst.add_stream('ffv1',rate=30); stream.width=src.streams.video[0].width; stream.height=src.streams.video[0].height; stream.pix_fmt='bgr0'
        for frame in src.decode(video=0):
            if count == end: break
            if frame.pts is None or abs(float(frame.pts*frame.time_base)-count/30)>1e-6:
                raise ValueError('Lossless prefix requires zero-origin normalized 30 Hz source PTS')
            rgb=frame.to_ndarray(format='rgb24'); digest.update(rgb.tobytes())
            row=av.VideoFrame.from_ndarray(rgb,format='rgb24'); row.pts=count;row.time_base=Fraction(1,30)
            for packet in stream.encode(row): dst.mux(packet)
            count+=1
        for packet in stream.encode(): dst.mux(packet)
    if count != end: raise ValueError('Source ended before active prefix')
    check=__import__('hashlib').sha256(); decoded=0
    with av.open(str(output)) as src:
        for frame in src.decode(video=0): check.update(frame.to_ndarray(format='rgb24').tobytes());decoded+=1
    if decoded != end or check.digest() != digest.digest():
        raise ValueError('Lossless active prefix changed decoded source pixels')
    return dict(path=str(output),sha256=file_sha256(output),frames=end,
        source_frame_indices=list(range(end)),decoded_rgb_sha256=digest.hexdigest(),lossless_rgb_verified=True)
