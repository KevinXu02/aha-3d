#!/usr/bin/env python3
"""Optional, hash-bound PromptHMR world-postopt experiments in the human runner.

The default backend is packaged here; prepare snapshots its exact source files.
The default scene pipeline and its selected deliveries are never changed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BACKEND = Path(__file__).resolve().parent / 'world_backend'
UPSTREAM_FILES = (
    'experiments/postopt_ab/build_results.py',
    'experiments/postopt_ab/build_gvhmr_global.py',
    'experiments/postopt_ab/run_ab.py',
    'experiments/postopt_ab/metrics.py',
    'experiments/postopt_ab/export_viewer_data.py',
    'pipeline/postprocessing_v2.py',
    'prompt_hmr/utils/rotation_conversions.py',
)
WEIGHTS = dict(postopt_acc=0.1, postopt_cont_vel=1000.0, postopt_cont_height=10.0,
               postopt_vreg=0.25, postopt_rot_reg=1.0, postopt_transl_reg=0.5)


def optimizer_config():
    return dict(WEIGHTS, postopt_lr_v2=0.01, postopt_orient_source='global',
                postopt_iters=1000, postopt_ground='flat', postopt_optimize_scale=False)


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def fingerprint(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return dict(path=str(path), sha256=h.hexdigest())


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def prepare(a):
    """Emit an executable graph; GPU work happens only in execute."""
    dest = a.out.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    upstream = dest / 'upstream'
    original = {}
    for rel in UPSTREAM_FILES:
        source, target = a.prompt_hmr / rel, upstream / rel
        before = fingerprint(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if fingerprint(target)['sha256'] != before['sha256']:
            raise ValueError(f'Upstream changed while copying: {source}')
        original[rel] = before
    entry = dest / 'world_postopt.py'
    shutil.copyfile(__file__, entry)
    # Copied entry is standalone: no project imports or relative runtime paths.
    code = [fingerprint(entry)] + [fingerprint(upstream / p) for p in UPSTREAM_FILES]
    inputs = {'model': fingerprint(a.smplx_model),
              'camera': fingerprint(a.camera_tracks),
              'camera_report': fingerprint(Path(str(a.camera_tracks) + '.json')),
              'native': fingerprint(a.gvhmr_run / 'motion_native.npz'),
              'provenance': fingerprint(a.gvhmr_run / 'provenance.json'),
              'hmr': fingerprint(a.gvhmr_run / 'hmr4d_results.pt'),
              'boxes': fingerprint(a.gvhmr_run / 'preprocess/bbx.pt'),
              'keypoints': fingerprint(a.gvhmr_run / 'preprocess/vitpose.pt')}
    active = a.gvhmr_run / 'active_inference/hmr4d_results.pt'
    if active.exists():
        inputs['active_hmr'] = fingerprint(active)
    # Preserve a venv's executable path: resolving its symlink selects base Python.
    base = [str(a.python.absolute()), str(entry), 'stage', '--upstream', str(upstream),
            '--params', '{params_file}', '--out', '{stage_dir}/result']
    env = {'CUDA_VISIBLE_DEVICES': str(a.gpu), 'OMP_NUM_THREADS': '4',
           'OPENBLAS_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1'}
    def stage(sid, needs, ins, outputs, params):
        return dict(id=sid, needs=needs, argv=base + ['--kind', sid],
                    inputs=ins, outputs={k: 'result/' + v for k, v in outputs.items()},
                    code=code, params=params, env=env)
    common = dict(gvhmr_run=str(a.gvhmr_run.resolve()), actor_id=a.actor_id,
                  camera_tracks=str(a.camera_tracks.resolve()),
                  smplx_model=str(a.smplx_model.resolve()))
    stages = [stage('adapt', [], inputs,
                    dict(init='results_init.pkl', native='results_native.pkl', gvhmr='results_gvhmr.pkl', meta='adapter_meta.json'), common)]
    def ref(s, o):
        return dict(stage=s, output=o)
    variants = ('v2',)
    for variant in variants:
        cfg = optimizer_config()
        s = stage(variant, ['adapt'], dict(init=ref('adapt', 'init'), meta=ref('adapt', 'meta'), model=inputs['model']),
                  dict(result='results.pkl', config='config.json'), cfg)
        s['argv'] += ['--init', '{input:init}', '--meta', '{input:meta}', '--model', '{input:model}']
        stages.append(s)
    s = stage('measure', ['adapt', *variants],
              dict(init=ref('adapt', 'init'), gvhmr=ref('adapt', 'gvhmr'), meta=ref('adapt', 'meta'),
                   model=inputs['model'], **{v: ref(v, 'result') for v in variants}),
              dict(metrics='metrics.json', series='series.npz', decision='decision.json',
                   report='comparison.md', plot='comparison.png'), {})
    s['argv'] += ['--meta', '{input:meta}', '--model', '{input:model}', '--results',
                  'gvhmr_global={input:gvhmr}', 'lifted_init={input:init}',
                  *[v + '={input:' + v + '}' for v in variants]]
    stages.append(s)
    config = dict(schema_version=1, id='world_postopt_' + a.actor_id, root=str(ROOT), stages=stages)
    write_json(dest / 'upstream_provenance.json', original)
    write_json(dest / 'experiment.json', config)
    print(dest / 'experiment.json')


def validate_inputs(native, cameras, provenance, actor):
    import numpy as np
    if str(native['source_actor_id']) != actor or provenance['actor_id'] != actor:
        raise ValueError('Actor identity mismatch')
    if provenance.get('tracker') != 'sam3' or provenance.get('camera_estimator') != 'pi3x':
        raise ValueError('Requires SAM3 tracking and Pi3X camera provenance')
    fps = float(native['fps'])
    times = np.asarray(native['frame_times_seconds'])
    if not math.isfinite(fps) or fps <= 0 or len(times) < 3:
        raise ValueError('Invalid timing')
    if not np.allclose(np.diff(times), 1 / fps, atol=1e-5):
        raise ValueError('Nonuniform timestamps require rotation resampling first')
    if len(cameras['c2w']) != len(times) or not np.allclose(cameras['time_seconds'], times, atol=1e-5):
        raise ValueError('Dense camera/body timestamps differ')
    source = str(cameras['camera_source']).lower()
    if 'dense' not in source or 'pi3' not in source:
        raise ValueError('Requires dense Pi3X cameras, not interpolated sparse observations')
    if not np.asarray(native['track_active']).any():
        raise ValueError('No active actor frames')
    return fps


def adapt(a, params):
    import joblib
    import numpy as np
    import torch
    import smplx
    from scipy.spatial.transform import Rotation
    source = Path(params['gvhmr_run'])
    native = np.load(source / 'motion_native.npz', allow_pickle=False)
    cams = np.load(params['camera_tracks'], allow_pickle=False)
    provenance = json.loads((source / 'provenance.json').read_text())
    fps = validate_inputs(native, cams, provenance, params['actor_id'])
    adapter = module(a.upstream / 'experiments/postopt_ab/build_results.py', 'postopt_adapter')
    scene_prior = None
    if params.get('scene_ground'):
        from tools.gvhmr.scene_ground import load_prior
        scene_prior = load_prior(params['scene_ground'], provenance['source_video_sha256'])
        if provenance.get('scene_ground', {}).get('prior', {}).get('prior_sha256') != scene_prior['prior_sha256']:
            raise ValueError('Rerun GVHMR with this exact scene prior before scene-ground optimization')
        transform = np.asarray(scene_prior['world_to_ground'])
        adapter.Z_UP_TO_Y_UP = transform[:3, :3]
        adapter.ransac_floor_height = lambda *args, **kwargs: -float(transform[1, 3])
    adapter.build(source, params['camera_tracks'], params['smplx_model'], params['actor_id'], a.out)
    initial = joblib.load(a.out / 'results_init.pkl')
    person = initial['people'][1]
    frames = person['frames']
    if not np.array_equal(frames, np.flatnonzero(native['track_active'])):
        raise ValueError('Adapter dropped active frames; review boxes before comparing')
    # Correct upstream floor sampling: inactive padded bodies must not vote.
    metrics = module(a.upstream / 'experiments/postopt_ab/metrics.py', 'postopt_metrics')
    body = smplx.SMPLX(params['smplx_model'], use_pca=False, flat_hand_mean=True, num_betas=10).cuda()
    sw = person['smplx_world']
    _, verts = metrics.forward(body, sw['pose'], sw['shape'], sw['trans'], 'cuda')
    floor_delta = 0. if scene_prior else adapter.ransac_floor_height(verts[..., 1].min(1).values.cpu().numpy())
    sw['trans'][:, 1] -= floor_delta
    cw = initial['camera_world']
    cw['Twc'][:, 1] -= floor_delta
    cw['Tcw'] = -(cw['Rcw'] @ cw['Twc'][..., None])[..., 0]
    cw['pred_cam_T'] = cw['Twc'].copy()
    joblib.dump(initial, a.out / 'results_init.pkl')
    # Additional comparator: preserve native global articulation and distances.
    # One whole-clip rotation from root orientation correspondences, scale exactly 1.
    baseline = copy.deepcopy(initial)
    target = baseline['people'][1]['smplx_world']
    prefix = 'smpl_params_global__'
    native_rot = Rotation.from_rotvec(native[prefix + 'global_orient'][frames]).as_matrix()
    lifted_rot = Rotation.from_rotvec(sw['pose'][:, :3]).as_matrix()
    rotation = Rotation.from_matrix(lifted_rot @ native_rot.transpose(0, 2, 1)).mean().as_matrix()
    betas = native[prefix + 'betas'][frames].astype(np.float32)
    # Rest pelvis is invariant to root rotation; needed for exact SMPL translation.
    rest, _ = metrics.forward(body, np.zeros((len(frames), 165)), betas,
                              np.zeros((len(frames), 3)), 'cuda')
    pelvis = rest[:, 0].cpu().numpy()
    translation = np.median(sw['trans'] + pelvis - (native[prefix + 'transl'][frames] + pelvis) @ rotation.T, axis=0)
    target['pose'][:, :3] = Rotation.from_matrix(rotation @ native_rot).as_rotvec()
    target['pose'][:, 3:66] = native[prefix + 'body_pose'][frames]
    target['shape'] = betas
    target['trans'] = ((native[prefix + 'transl'][frames] + pelvis) @ rotation.T + translation - pelvis).astype(np.float32)
    joblib.dump(baseline, a.out / 'results_native.pkl')
    # Fair postprocessing comparator: GVHMR's own global branch and its exactly
    # recovered camera, not a scene-aligned body projected through Pi3X.
    baseline_argv = [sys.executable, str(a.upstream / 'experiments/postopt_ab/build_gvhmr_global.py'),
                    '--gvhmr-run', str(source), '--init', str(a.out / 'results_init.pkl'),
                    '--smplx-model', params['smplx_model'], '--out', str(a.out / 'results_gvhmr.pkl')]
    if scene_prior:
        baseline_module = module(a.upstream / 'experiments/postopt_ab/build_gvhmr_global.py', 'scene_gvhmr_baseline')
        baseline_module.BR.ransac_floor_height = lambda *args, **kwargs: 0.
        old_argv = sys.argv
        try:
            sys.argv = baseline_argv[1:]
            baseline_module.main()
        finally:
            sys.argv = old_argv
    else:
        subprocess.run(baseline_argv, check=True)
    global_result = joblib.load(a.out / 'results_gvhmr.pkl')
    gsw = global_result['people'][1]['smplx_world']
    _, gv = metrics.forward(body, gsw['pose'], gsw['shape'], gsw['trans'], 'cuda')
    global_floor_delta = 0. if scene_prior else adapter.ransac_floor_height(gv[..., 1].min(1).values.cpu().numpy())
    gsw['trans'][:, 1] -= global_floor_delta
    gcw = global_result['camera_world']
    gcw['Twc'][:, 1] -= global_floor_delta
    gcw['Tcw'] = -(gcw['Rcw'] @ gcw['Twc'][..., None])[..., 0]
    gcw['pred_cam_T'] = gcw['Twc'].copy()
    joblib.dump(global_result, a.out / 'results_gvhmr.pkl')
    meta = json.loads((a.out / 'adapter_meta.json').read_text())
    meta.update(fps=fps, active_floor_delta_m=float(floor_delta),
                gvhmr_active_floor_delta_m=float(global_floor_delta),
                primary_comparator='gvhmr_global: native global branch in its own gravity frame; Kabsch camera',
                native_alignment=dict(rotation=rotation.tolist(), translation=translation.tolist(), scale=1.0),
                baseline_scope='Fresh whole-clip rigid native comparator; not an existing selected room delivery',
                source_video_sha256=provenance['source_video_sha256'])
    if scene_prior:
        meta.update(scene_ground=scene_prior, body_floor_fitting=False,
                    primary_comparator='GVHMR scene-guided global branch with the same Kimodo completion, if present',
                    completion=provenance.get('completion'))
        initial['scene_ground'] = scene_prior
        joblib.dump(initial, a.out / 'results_init.pkl')
    write_json(a.out / 'adapter_meta.json', meta)


def optimize(a, params):
    import joblib
    import torch
    import smplx
    from omegaconf import OmegaConf
    harness = module(a.upstream / 'experiments/postopt_ab/run_ab.py', 'postopt_harness')
    harness._stub_prompt_hmr()
    optimizer = module(a.upstream / 'pipeline/postprocessing_v2.py', 'postopt_v2')
    initial = joblib.load(a.init)
    if initial.get('scene_ground'):
        if params.get('postopt_orient_source', 'global') != 'global' or params.get('postopt_optimize_scale', False):
            raise ValueError('Scene-guided v2 requires the scene global orientation and fixed trajectory scale')
        from tools.gvhmr.scene_ground import yaw_rotation_6d, scene_orientation
        optimizer.rotation_6d_to_matrix = yaw_rotation_6d
        optimizer._orientation_from_global_branch = scene_orientation(optimizer)
        params = dict(params, placement_rotation='yaw_only_scene_upright', scene_ground=initial['scene_ground'])
    require_global_orientation(initial, params)
    params['fps'] = json.loads(a.meta.read_text())['fps']
    params['seed'] = 0
    torch.manual_seed(0)
    body = smplx.SMPLX(str(a.model), use_pca=False, flat_hand_mean=True, num_betas=10)
    result = optimizer.post_optimization_v2(OmegaConf.create(params), initial,
                                            [None] * initial['n_frames'], body, flat_ground=True)
    a.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, a.out / 'results.pkl')
    write_json(a.out / 'config.json', params)


def require_global_orientation(result, params):
    import numpy as np
    if params.get('postopt_orient_source', 'global') != 'global':
        return
    for pid, person in result['people'].items():
        orient = np.asarray(person['smplx_cam'].get('global_orient_world'))
        if orient.shape != (len(person['frames']), 3) or not np.isfinite(orient).all():
            raise ValueError(f'Person {pid} requires finite global-branch orientation; rebuild adaptation')


def person_camera(result, pid):
    """Per-person extrinsics inherit shared intrinsics when upstream omits them."""
    return dict(result['camera_world'], **result['people'][pid].get('camera_world', {}))


def with_camera(result, camera):
    """Explicit counterfactual; override person cameras as well as the top level."""
    fixed = copy.deepcopy(result)
    fixed['camera_world'] = camera
    for person in fixed['people'].values():
        person['camera_world'] = camera
    return fixed


def assess(table):
    """Compare motion metrics; per-person placement paths are not camera errors."""
    base = table['gvhmr_global']['people']
    metrics = ('contact_slip_mps_mean', 'root_accel_rms', 'penetration_cm_mean',
               'contact_height_cm_median')
    diagnostics = {}
    for name, row in table.items():
        diagnostics[name] = {
            str(pid): {key: (person[key] - base[pid][key]
                            if person.get(key) is not None and base[pid].get(key) is not None else None)
                       for key in metrics}
            for pid, person in row['people'].items()}
    return dict(allow_scene_import=False, selection='unchanged', baseline='gvhmr_global',
                reason='Motion comparison is separate from acceptance in the authored scene.',
                camera_path_is_quality_criterion=False, metric_deltas_vs_gvhmr=diagnostics)


def measure(a):
    import joblib
    import numpy as np
    import smplx
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics = module(a.upstream / 'experiments/postopt_ab/metrics.py', 'postopt_metrics')
    body = smplx.SMPLX(str(a.model), use_pca=False, flat_hand_mean=True, num_betas=10).cuda()
    meta = json.loads(a.meta.read_text())
    paths = dict(spec.split('=', 1) for spec in a.results)
    initial = joblib.load(paths['lifted_init'])
    def camera_joints(result):
        values = {}
        for pid, person in result['people'].items():
            sw = person['smplx_world']; fr = np.asarray(person['frames'])
            joints, _ = metrics.forward(body, sw['pose'], sw['shape'], sw['trans'], 'cuda')
            cw = person.get('camera_world', result['camera_world'])
            values[pid] = ((np.asarray(cw['Rcw'])[fr, None] @ joints[:, :55].cpu().numpy()[..., None])[..., 0]
                           + np.asarray(cw['Tcw'])[fr, None])
        return values
    reference_joints = camera_joints(initial)
    table, series = {}, {}
    for name, path in paths.items():
        result = joblib.load(path)
        if result['n_frames'] != initial['n_frames'] or result['people'].keys() != initial['people'].keys():
            raise ValueError('Optimizer changed timing or actor set')
        for pid, person in result['people'].items():
            if not np.array_equal(person['frames'], initial['people'][pid]['frames']):
                raise ValueError('Optimizer changed lifecycle')
            for key in ('pose', 'shape', 'trans'):
                if not np.isfinite(person['smplx_world'][key]).all():
                    raise ValueError('Nonfinite body result')
            if not np.array_equal(person['smplx_cam']['static_conf_logits'], initial['people'][pid]['smplx_cam']['static_conf_logits']):
                raise ValueError('Comparators use different contact evidence')
            if name not in ('native_rigid', 'gvhmr_global'):
                for key, value in [('shape', slice(None)), ('pose', slice(3, None))]:
                    if not np.array_equal(person['smplx_world'][key][:, value], initial['people'][pid]['smplx_world'][key][:, value]):
                        raise ValueError('Postopt changed body dimensions or local articulation')
        pp, ser, cam = metrics.evaluate(result, body, meta['fps'])
        table[name] = dict(people=pp, camera=dict(path_m=cam['camera_path_m']))
        if name in ('gvhmr_global', 'v2'):
            cj = camera_joints(result)
            residual = max(float(np.linalg.norm(cj[pid] - reference_joints[pid], axis=-1).max()) for pid in cj)
            # Saved axis-angle/float32 SMPL round trips are not bit-exact.
            # This is a numerical consistency bound, not a pose-accuracy claim.
            tolerance_m = 0.002
            if residual > tolerance_m:
                raise ValueError(f'{name} changed camera-frame body: {residual} m')
            table[name]['camera_space_invariance_max_m'] = residual
            table[name]['camera_space_tolerance_m'] = tolerance_m
        if name == 'v2' and meta.get('completion'):
            # Placement optimization may change root/orientation after completion;
            # check the FINAL source-shaped FK joins, not just the sampler splice.
            joins = []
            for person in result['people'].values():
                sw = person['smplx_world']; fr = np.asarray(person['frames'])
                joints, _ = metrics.forward(body, sw['pose'], sw['shape'], sw['trans'], 'cuda')
                velocity = np.diff(joints[:, :22].cpu().numpy(), axis=0)*meta['fps']
                for gap in meta['completion']['gaps']:
                    if gap['status'] != 'completed':
                        continue
                    start, end = [int(np.flatnonzero(fr == gap[key])[0]) for key in ('start', 'end')]
                    jump = float(max(np.linalg.norm(velocity[start-1]-velocity[start-2], axis=-1).max(),
                                     np.linalg.norm(velocity[end-1]-velocity[end], axis=-1).max()))
                    joins.append(dict(start=gap['start'], end=gap['end'], velocity_jump_m_s=jump,
                                      passed=jump <= 1.0, limit_m_s=1.0))
            table[name]['completion_joins'] = joins
            if not all(row['passed'] for row in joins):
                raise ValueError('V2 placement regressed a completed interval join: ' + str(joins))

        series.update({name + '__' + k: v for k, v in ser.items()})
        series[name + '__camera_Twc'] = cam['camera_Twc']
        # Reprojection using unchanged Pi3X cameras exposes moving-camera compensation.
        fixed = with_camera(result, initial['camera_world'])
        fixed_pp, _, _ = metrics.evaluate(fixed, body, meta['fps'])
        for pid in pp:
            pp[pid]['fixed_camera_reproj_px_median'] = fixed_pp[pid]['reproj_px_median']
    a.out.mkdir(parents=True, exist_ok=True)
    write_json(a.out / 'metrics.json', table)
    np.savez_compressed(a.out / 'series.npz', **series)
    decision = assess(table)
    write_json(a.out / 'decision.json', decision)
    lines = ['# World postopt comparison', '',
             f"Actor: `{meta['actor_id']}`; {meta['frames_active']}/{meta['frames_total']} active frames; {meta['fps']:g} fps.", '',
             'The primary comparator is GVHMR global output with an exactly recovered camera. Lifted init is optimizer input. Frames live in their own gravity/world bases; no trajectory scaling is fitted.', '',
             '| Variant | Foot slip mean (m/s) | Root accel RMS (m/s²) | Contact height median (cm) | Mean penetration (cm) | Max penetration (cm) | Reprojection (px) |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    if meta.get('scene_ground'):
        completion_note = ' with the same accepted Kimodo completion' if meta.get('completion') else ''
        lines[4] = ('The baseline is scene-guided GVHMR' + completion_note + '. '
                    'All arms use the fixed reconstructed scene floor/upright at scale 1. '
                    'The original raw GVHMR run is archived separately; this is not a comparison against unmodified GVHMR.')
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for name, row in table.items():
        p = next(iter(row['people'].values()))
        height = p['contact_height_cm_median']
        height_text = f'{height:.2f}' if height is not None else 'unknown'
        reproj = p['reproj_px_median']
        reproj_text = f'{reproj:.2f}' if reproj is not None else 'unknown'
        lines.append(f"| {name} | {p['contact_slip_mps_mean']:.4f} | {p['root_accel_rms']:.3f} | {height_text} | {p['penetration_cm_mean']:.3f} | {p['penetration_cm_max']:.2f} | {reproj_text} |")
        pid = next(iter(row['people']))
        fr = series[f'{name}__p{pid}_frames']
        root = series[f'{name}__p{pid}_root']
        axes[0].plot(root[:, 0], root[:, 2], label=name)
        acc = np.linalg.norm(np.diff(root, n=2, axis=0) * meta['fps'] ** 2, axis=1)
        axes[1].plot(fr[1:-1] / meta['fps'], acc, label=name)
        axes[2].plot(fr / meta['fps'], series[f'{name}__p{pid}_lowest_vertex_y'], label=name)
    for ax, title in zip(axes, ['Root paths in separate world gauges (m)', 'Root acceleration (m/s²) vs seconds', 'Lowest mesh height (m) vs seconds']):
        ax.set_title(title); ax.grid(alpha=0.3)
    for ax in axes[:1]:
        ax.set_aspect('equal', adjustable='datalim')
    axes[2].axhline(0, color='black', linestyle=':')
    axes[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(a.out / 'comparison.png', dpi=140); plt.close(fig)
    contact_note = ('Contact labels come from the GVHMR static head outside completed intervals and conservatively gated Kimodo labels inside them. '
                    if meta.get('completion') else 'Contact labels come from the GVHMR static head. ')
    floor_note = ('The support plane is the fixed reviewed scene-reconstruction estimate, not surveyed ground truth. '
                  if meta.get('scene_ground') else 'The flat support plane is body-derived, not measured authored-room geometry. ')
    lines += ['', contact_note + 'These labels are not reviewed contact ground truth. '
              'The selected 2D detector is a GVHMR input; the upstream heldout-named confidence-band metric is not independent of reconstruction. '
              + floor_note +
              'Floor metrics do not validate hand, seat or furniture contact.', '',
              'V2 preserves reprojection through each person’s own placement camera. Camera path and original-camera projections are coordinate diagnostics, not motion-quality criteria. '
              'Assess slip, contact height/penetration and root acceleration against GVHMR. Scene import requires separate common-scene validation.', '', '![World comparison](comparison.png)', '']
    (a.out / 'comparison.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--prompt-hmr', type=Path, default=BACKEND,
                   help='Optional legacy backend override; defaults to the packaged motion helpers')
    for name in ('gvhmr-run', 'camera-tracks', 'smplx-model', 'python', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--actor-id', required=True)
    p.add_argument('--gpu', type=int, required=True, help='One explicitly coordinated visible GPU')
    p = sub.add_parser('stage')
    p.add_argument('--kind', choices=('adapt', 'v2', 'measure'), required=True)
    for name in ('upstream', 'params', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    for name in ('init', 'meta', 'model'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--results', nargs='+')
    a = parser.parse_args()
    if a.command == 'prepare':
        prepare(a)
    elif a.kind == 'adapt':
        adapt(a, json.loads(a.params.read_text()))
    elif a.kind == 'measure':
        measure(a)
    else:
        optimize(a, json.loads(a.params.read_text()))


if __name__ == '__main__':
    main()
