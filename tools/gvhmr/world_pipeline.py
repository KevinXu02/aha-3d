#!/usr/bin/env python3
"""Prepare the full source-video SAM3 → PMPose → GVHMR → dense Pi3X → v2 workflow."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import world_postopt as post
from pmpose import add_arguments

ROOT = Path(__file__).resolve().parents[2]
# The public script also runs from an uninstalled checkout or a source snapshot.
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
FRONTEND_FILES = ('run_pi3x_dense.py',)


def source_files(folder):
    """Pin installed source/config files individually; upstream configs have links."""
    return [post.fingerprint(p) for p in sorted(Path(folder).rglob('*'))
            if p.is_file() and p.suffix in ('.py', '.yaml', '.yml')]


def prepare(a):
    gpu = a.gpu
    if gpu is None:
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0').strip()
        if not visible.isdecimal():
            raise ValueError('Pass --gpu: inherited CUDA_VISIBLE_DEVICES is not one numeric device')
        gpu = int(visible)
    kimodo_completion = getattr(a, 'kimodo_completion', False)
    if kimodo_completion:
        if not getattr(a, 'scene_ground', None) or a.pose_detector != 'pmpose':
            raise ValueError('Kimodo completion requires a scene-ground prior and PMPose evidence')
        if not (getattr(a, 'completion_prompt', None) or '').strip():
            raise ValueError('--kimodo-completion requires --completion-prompt with the observed action')
    dest = a.out.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    snapshot = dest / 'snapshot'
    upstream = dest / 'upstream'
    provenance = {}
    for rel in (*post.UPSTREAM_FILES, *('experiments/postopt_ab/' + f for f in FRONTEND_FILES)):
        target = upstream / rel; target.parent.mkdir(parents=True, exist_ok=True)
        before = post.fingerprint(a.prompt_hmr / rel)
        shutil.copyfile(a.prompt_hmr / rel, target)
        if post.fingerprint(target)['sha256'] != before['sha256']:
            raise ValueError('Upstream changed during snapshot: ' + rel)
        provenance[rel] = before
    # Snapshot local imports, including the current bbox policy, without editing
    # another task's working files. Installed models/environments stay external.
    for folder in ('tools/gvhmr', 'src/aha3d/motion'):
        shutil.copytree(ROOT / folder, snapshot / folder, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for rel in ('src/aha3d/__init__.py', 'src/aha3d/io.py', 'src/aha3d/runtime.py'):
        target = snapshot / rel; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, target)
    entry = snapshot / 'tools/gvhmr/world_pipeline.py'
    python = str(a.python.absolute())
    code = [dict(path=str(snapshot)), dict(path=str(upstream))]
    env = dict(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1',
               PYTHONPATH=str(snapshot) + os.pathsep + str(snapshot / 'src'))
    params = dict(project_root=str(ROOT), snapshot=str(snapshot), upstream=str(upstream),
                  actor_id=a.actor_id, prompt_frame=a.prompt_frame, box=a.box,
                  tracker='sam3', sam3=str(a.sam3.resolve()),
                  sam3_python=str(a.sam3_python.absolute()),
                  gvhmr_repo=str(a.gvhmr_repo.resolve()), pi3=str(a.pi3.resolve()),
                  gpu=gpu, pose_detector=a.pose_detector, intrinsics=getattr(a, 'intrinsics', 'upstream'),
                  pmpose_python=str(a.pmpose_python.absolute()), pmpose_root=str(a.pmpose_root.absolute()),
                  pmpose_checkpoint=str(a.pmpose_checkpoint.absolute()), pmpose_variant=a.pmpose_variant,
                  pmpose_ld_preload=a.pmpose_ld_preload)
    scene_ground = getattr(a, 'scene_ground', None)
    params['kimodo_completion'] = kimodo_completion
    if kimodo_completion:
        params.update(kimodo_python=str(a.kimodo_python.absolute()),
                      completion_prompt=a.completion_prompt, completion_context=a.completion_context)
    def stage(sid, needs, inputs, outputs, extra=None):
        argv = [python, str(entry), 'stage', '--kind', sid, '--params', '{params_file}', '--out', '{stage_dir}/result']
        for name in inputs:
            argv += ['--input', name + '={input:' + name + '}']
        return dict(id=sid, needs=needs, inputs=inputs,
                    outputs={k: 'result/' + v for k, v in dict(outputs, runtime='runtime.json').items()},
                    argv=argv, code=code, params=dict(params, **(extra or {})), env=env)
    def ref(stage, output): return dict(stage=stage, output=output)
    video, model = post.fingerprint(a.video), post.fingerprint(a.smplx_model)
    stages = [stage('track', [], dict(video=video, checkpoint=post.fingerprint(a.sam3_checkpoint)),
                    dict(masks='masks.npz', report='tracking.json'))]
    stages[0]['params']['source_video_sha256'] = video['sha256']
    stages[0]['code'] = code + source_files(a.sam3 / 'sam3')
    stages += [stage('lifecycle', ['track'], dict(video=video, masks=ref('track', 'masks'),
                    review=post.fingerprint(a.lifecycle_review)), dict(lifecycle='lifecycle.json', gate='bbox_gate.json'))]
    # Pin only relevant bundle files, not reference meshes/images.
    bundle_inputs = {name.replace('.', '_'): post.fingerprint(a.pi3x_bundle / name)
                     for name in ('inputs.json', 'cameras.json', 'camera_review.json')}
    if scene_ground:
        from tools.gvhmr.scene_ground import load_prior
        load_prior(scene_ground, video['sha256'])
        bundle_inputs['scene_ground'] = post.fingerprint(scene_ground)
    gvhmr_code = source_files(a.gvhmr_repo / 'hmr4d') + [post.fingerprint(a.gvhmr_repo / 'tools/demo/demo.py')]
    s = stage('bodies', ['track', 'lifecycle'],
              dict(video=video, masks=ref('track', 'masks'), lifecycle=ref('lifecycle', 'lifecycle'),
                   checkpoint=post.fingerprint(a.gvhmr_repo / 'inputs/checkpoints/gvhmr/gvhmr_b1b2.ckpt'),
                   **bundle_inputs), dict(run='gvhmr', native='native_marker.json'),
              dict(pi3x_bundle=str(a.pi3x_bundle.resolve())))
    for key, rel in {
        'smplx': 'body_models/smplx/SMPLX_NEUTRAL.npz',
        'smpl': 'body_models/smpl/SMPL_NEUTRAL.pkl',
        'image_features': 'hmr2/epoch=10-step=25000.ckpt',
        'pose_model': 'vitpose/vitpose-h-multi-coco.pth',
    }.items():
        asset = a.gvhmr_repo / 'inputs/checkpoints' / rel
        if asset.exists():
            s['inputs'][key] = post.fingerprint(asset)
            s['argv'] += ['--input', key + '={input:' + key + '}']
    s['code'] = code + gvhmr_code
    if a.pose_detector == 'pmpose':
        # Rebuild input flags after changing the detector dependency.
        s['argv'] = s['argv'][:s['argv'].index('--input')]
        s['inputs']['pose_model'] = post.fingerprint(a.pmpose_checkpoint)
        for name in s['inputs']:
            s['argv'] += ['--input', name + '={input:' + name + '}']
        s['code'] += source_files(a.pmpose_root)
    stages.append(s)
    body_stage = 'bodies'
    if kimodo_completion:
        s = stage('complete', ['bodies', 'track'],
                  dict(bodies=ref('bodies', 'run'), masks=ref('track', 'masks'), model=model,
                       checkpoint_config=post.fingerprint(a.kimodo_checkpoint/'config.yaml'),
                       checkpoint=post.fingerprint(a.kimodo_checkpoint/'model.safetensors')),
                  dict(run='gvhmr', report='completion.json', candidate='candidate.npz'))
        s['code'] += source_files(a.kimodo_upstream/'kimodo')
        s['params']['kimodo_upstream'] = str(a.kimodo_upstream.absolute())
        stages.append(s)
        body_stage = 'complete'
    s = stage('camera', [body_stage],
              dict(bodies=ref(body_stage, 'run'), video=video,
                   checkpoint=post.fingerprint(a.pi3_checkpoint)),
              dict(camera='camera_tracks_dense.npz', report='camera_tracks_dense.npz.json',
                   dense_predictions='camera_tracks_dense-predictions'))
    s['code'] = code + source_files(a.pi3 / 'pi3')
    stages.append(s)
    stages += [stage('adapt', [body_stage, 'camera'], dict(bodies=ref(body_stage, 'run'),
                     camera=ref('camera', 'camera'), model=model,
                     **({'scene_ground': post.fingerprint(scene_ground)} if scene_ground else {})),
                     dict(init='results_init.pkl', native='results_native.pkl', gvhmr='results_gvhmr.pkl', meta='adapter_meta.json'))]
    cfg = post.optimizer_config()
    stages += [stage('v2', ['adapt'], dict(init=ref('adapt', 'init'), meta=ref('adapt', 'meta'), model=model),
                     dict(result='results.pkl', config='config.json'), dict(optimizer=cfg))]
    stages += [stage('measure', ['adapt', 'v2'],
                     dict(init=ref('adapt', 'init'), gvhmr=ref('adapt', 'gvhmr'), meta=ref('adapt', 'meta'),
                          v2=ref('v2', 'result'), model=model),
                     dict(metrics='metrics.json', series='series.npz', decision='decision.json',
                          report='comparison.md', plot='comparison.png'))]
    stages += [stage('review', ['track', body_stage, 'adapt', 'v2'],
                     dict(video=video, masks=ref('track', 'masks'), bodies=ref(body_stage, 'run'),
                          init=ref('adapt', 'init'), native=ref('adapt', 'native'), gvhmr=ref('adapt', 'gvhmr'), meta=ref('adapt', 'meta'),
                          v2=ref('v2', 'result'), model=model),
                     dict(video='comparison.mp4', frames='representative_frames.jpg', validation='video_validation.json',
                          world_video='world_comparison.mp4', world_frames='world_frames.jpg', world_validation='world_validation.json'))]
    post.write_json(dest / 'upstream_provenance.json', provenance)
    source_state = {}
    for name, repo in [('indoor', ROOT), ('motion_backend', a.prompt_hmr)]:
        def git(*args):
            result = subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True)
            return result.stdout.strip() if result.returncode == 0 else None
        source_state[name] = dict(commit=git('rev-parse', 'HEAD'), branch=git('branch', '--show-current'),
                                  local_changes=git('status', '--short'))
    post.write_json(dest / 'source_state.json', source_state)
    post.write_json(dest / 'experiment.json', dict(schema_version=1, id='world_pipeline_' + a.actor_id,
                                                  root=str(ROOT), stages=stages))
    print(dest / 'experiment.json')


def stage(a):
    p = json.loads(a.params.read_text())
    ins = dict(x.split('=', 1) for x in a.input)
    a.out.mkdir(parents=True, exist_ok=False)
    versions = {}
    for package in ('torch', 'numpy', 'scipy', 'smplx', 'av', 'opencv-python', 'safetensors', 'omegaconf'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    post.write_json(a.out / 'runtime.json', dict(python=sys.version, executable=sys.executable,
                    packages=versions, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    run_id=os.environ.get('INDOOR_RUN_ID')))
    sys.path[:0] = [p['snapshot'], str(Path(p['snapshot']) / 'src')]
    upstream = Path(p['upstream'])
    if a.kind == 'track':
        import cv2
        cap = cv2.VideoCapture(ins['video'])
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if size != (1280, 720):
            raise ValueError('Motion tracking requires 1280x720 source raster')
        if abs(fps - 30) > 1e-5:
            raise ValueError('GVHMR frontend requires a normalized 30 Hz source')
        if p.get('tracker') != 'sam3':
            raise ValueError('Tracking stage requires explicit SAM3 provenance')
        cmd = [p['sam3_python'], str(Path(p['snapshot'])/'tools/gvhmr/sam3_tracking.py'),
               '--video', ins['video'], '--root', p['sam3'], '--checkpoint', ins['checkpoint'],
               '--actor-id', p['actor_id'], '--prompt-frame', str(p['prompt_frame']),
               '--box', *map(str, p['box']), '--out', str(a.out),
               '--source-video-sha256', p['source_video_sha256']]
        subprocess.run(cmd, check=True)
    elif a.kind == 'lifecycle':
        import numpy as np
        from tools.gvhmr.tracking_evidence import video_timeline, file_sha256
        from tools.gvhmr.world_tracking import gate_trailing_max, track_end
        from tools.gvhmr.track_lifecycle import load_lifecycle
        tl = video_timeline(Path(ins['video']))
        load_lifecycle(ins['review'], times=tl['time_seconds'], image_size=tl['image_size'],
                       source_sha256=file_sha256(Path(ins['video'])), actor_id=p['actor_id'])
        z = np.load(ins['masks'], allow_pickle=False)
        area = z['mask_area_pixels']; boxes = np.nan_to_num(z['bbox_xyxy_boundary'])
        keep = gate_trailing_max(boxes, area > 0, area=area, fps=30)
        lifecycle_review = json.loads(Path(ins['review']).read_text())
        if lifecycle_review['reviewer'].startswith('derived:'):
            raise ValueError('Mask-derived lifecycle is only a proposal; source review is required')
        post.write_json(a.out / 'lifecycle.json', lifecycle_review)
        post.write_json(a.out / 'bbox_gate.json', dict(proposed_support_end=track_end(keep),
                        reviewed_exit=lifecycle_review['terminal_exit_frame'], accepted_frames=np.flatnonzero(keep).tolist(),
                        rejected_frames=np.flatnonzero(~keep).tolist(),
                        scope='Scale and mask-fill trailing-max gate; unsupported crops do not establish physical exit'))
    elif a.kind == 'bodies':
        cmd = [sys.executable, '-m', 'tools.gvhmr.person_reconstruct', '--video', ins['video'],
               '--actor-id', p['actor_id'], '--person-masks', ins['masks'], '--track-lifecycle', ins['lifecycle'],
               '--pi3x-bundle', p['pi3x_bundle'], '--repo', p['gvhmr_repo'],
               '--gpu', str(p['gpu']), '--output', str(a.out / 'gvhmr')]
        cmd += ['--pose-detector', p.get('pose_detector', 'vitpose'), '--intrinsics', p.get('intrinsics', 'upstream')]
        if 'scene_ground' in ins:
            cmd += ['--scene-ground', ins['scene_ground']]
        if p.get('pose_detector') == 'pmpose':
            for key in ('pmpose_python', 'pmpose_root', 'pmpose_variant', 'pmpose_ld_preload'):
                cmd += ['--' + key.replace('_', '-'), p[key]]
            cmd += ['--pmpose-checkpoint', ins['pose_model']]
        from tools.gvhmr.world_tracking import install_policy
        from tools.gvhmr import person_reconstruct
        sys.argv = ['person_reconstruct', *cmd[3:]]
        with install_policy():
            person_reconstruct.main()
        post.write_json(a.out / 'native_marker.json', dict(fresh_inference=True, path='gvhmr/motion_native.npz'))
    elif a.kind == 'complete':
        # The worker creates its own output root. Keep stage runtime evidence
        # written above by using a child and moving only this task's new results.
        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join([p['kimodo_upstream'], p['snapshot'], str(Path(p['snapshot'])/'src')])
        subprocess.run([p['kimodo_python'], str(Path(p['snapshot'])/'tools/gvhmr/occlusion_completion.py'),
                        '--run', ins['bodies'], '--masks', ins['masks'], '--smplx-model', ins['model'],
                        '--checkpoint', str(Path(ins['checkpoint']).parent),
                        '--prompt', p['completion_prompt'], '--context', str(p['completion_context']),
                        '--out', str(a.out/'work')], env=env, check=True)
        for child in (a.out/'work').iterdir():
            shutil.move(str(child), str(a.out/child.name))
        (a.out/'work').rmdir()
    elif a.kind == 'camera':
        subprocess.run([sys.executable, str(upstream / 'experiments/postopt_ab/run_pi3x_dense.py'),
                        '--video', ins['video'], '--upstream', p['pi3'], '--checkpoint', ins['checkpoint'],
                        '--reference-tracks', str(Path(ins['bodies']) / 'camera_adapter/camera_tracks.npz'),
                        '--out', str(a.out / 'camera_tracks_dense.npz')], check=True)
    elif a.kind == 'adapt':
        post.adapt(argparse.Namespace(upstream=upstream, out=a.out),
                   dict(gvhmr_run=ins['bodies'], camera_tracks=ins['camera'], smplx_model=ins['model'], actor_id=p['actor_id'],
                        scene_ground=ins.get('scene_ground')))
    elif a.kind == 'v2':
        post.optimize(argparse.Namespace(upstream=upstream, out=a.out, init=ins['init'],
                      model=ins['model'], meta=Path(ins['meta']), kind='v2'), p['optimizer'])
    elif a.kind == 'measure':
        post.measure(argparse.Namespace(upstream=upstream, out=a.out, model=ins['model'], meta=Path(ins['meta']),
                     results=['gvhmr_global=' + ins['gvhmr'], 'lifted_init=' + ins['init'], 'v2=' + ins['v2']]))
    elif a.kind == 'review':
        review(a.out, upstream, ins)


def review(out, upstream, ins):
    """Full-duration tracking/crop and body skeleton comparison with decode evidence."""
    import cv2
    import imageio
    import joblib
    import numpy as np
    import smplx
    import torch
    from PIL import Image
    metrics = post.module(upstream / 'experiments/postopt_ab/metrics.py', 'review_metrics')
    initial, native, candidate = [joblib.load(ins[k]) for k in ('init', 'gvhmr', 'v2')]
    meta = json.loads(Path(ins['meta']).read_text())
    masks = np.load(ins['masks'], allow_pickle=False)['masks']
    boxes = torch.load(Path(ins['bodies']) / 'preprocess/bbx.pt', map_location='cpu', weights_only=False)
    boxes = boxes['bbx_xyxy'].numpy()
    from tools.gvhmr.world_review import render as render_world
    render_world(out, upstream, ins['model'], ins['gvhmr'], ins['v2'], meta['fps'])
    body = smplx.SMPLX(ins['model'], use_pca=False, flat_hand_mean=True, num_betas=10).cuda()
    coords = []
    for result, cam in ((native, post.person_camera(native, 1)),
                        (candidate, post.person_camera(candidate, 1)),
                        (candidate, initial['camera_world'])):
        person = result['people'][1]; sw = person['smplx_world']; fr = person['frames']
        joints, _ = metrics.forward(body, sw['pose'], sw['shape'], sw['trans'], 'cuda')
        jc = (cam['Rcw'][fr, None] @ joints[:, :22].cpu().numpy()[..., None])[..., 0] + cam['Tcw'][fr, None]
        uv = jc[..., :2] / np.maximum(jc[..., 2:], .001) * cam['img_focal'] + cam['img_center']
        coords.append({int(f): (uv[i], jc[i, :, 2]) for i, f in enumerate(fr)})
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    cap = cv2.VideoCapture(ins['video']); fps = cap.get(cv2.CAP_PROP_FPS); count = 0
    writer = imageio.get_writer(out / 'comparison.mp4', fps=fps, codec='libx264', quality=7, macro_block_size=1)
    n = initial['n_frames']; end = int(initial['people'][1]['frames'][-1]) + 1
    selected = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1] + [f for f in (end - 1, end, end + 1) if 0 <= f < n]))
    images = []
    while True:
        ok, source = cap.read()
        if not ok: break
        panels = []
        for column, title in enumerate(('Source masks and actual bbox', 'GVHMR / recovered camera', 'V2 / own placement camera', 'V2 / original camera (diagnostic)')):
            img = cv2.resize(source, (426, 240)); sx, sy = 426 / source.shape[1], 240 / source.shape[0]
            if column == 0:
                mask = cv2.resize(masks[count].astype('uint8'), (426, 240), interpolation=cv2.INTER_NEAREST).astype(bool)
                img[mask] = (img[mask] * .6 + np.array([50, 180, 255]) * .4).astype('uint8')
                if np.isfinite(boxes[count]).all():
                    x0, y0, x1, y1 = (boxes[count] * [sx, sy, sx, sy]).astype(int)
                    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 1)
            elif count in coords[column - 1]:
                uv, z = coords[column - 1][count]; uv = uv * [sx, sy]
                color = [(80, 245, 110), (20, 230, 255), (70, 90, 255)][column - 1]
                for child, parent in enumerate(parents):
                    if parent >= 0 and z[child] > 0 and z[parent] > 0 and np.isfinite(uv[[child, parent]]).all():
                        start, stop = np.clip(np.rint(uv[[child, parent]]), -10000, 10000).astype(int)
                        cv2.line(img, tuple(start), tuple(stop), color, 2, cv2.LINE_AA)
            panel = np.zeros((280, 426, 3), dtype='uint8'); panel[40:] = img
            cv2.putText(panel, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, .46, (255, 255, 255), 1)
            cv2.putText(panel, f'frame {count}/{n-1} | ' + ('ACTIVE' if count in coords[0] else 'INACTIVE'),
                        (6, 33), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1)
            panels.append(panel)
        frame = cv2.cvtColor(np.concatenate(panels, axis=1), cv2.COLOR_BGR2RGB)
        writer.append_data(frame)
        if count in selected: images.append(Image.fromarray(frame))
        count += 1
    cap.release(); writer.close()
    sheet = Image.new('RGB', (1704, 280 * len(images)))
    for index, image in enumerate(images): sheet.paste(image, (0, 280 * index))
    sheet.save(out / 'representative_frames.jpg', quality=90)
    cap = cv2.VideoCapture(str(out / 'comparison.mp4')); decoded_fps = cap.get(cv2.CAP_PROP_FPS); decoded = 0
    while cap.read()[0]: decoded += 1
    cap.release()
    if count != n or decoded != n or abs(decoded_fps - fps) > 1e-5 or abs(fps - meta['fps']) > 1e-5:
        raise ValueError('Comparison video decode/timing mismatch')
    post.write_json(out / 'video_validation.json', dict(source=post.fingerprint(ins['video']),
                    video=post.fingerprint(out / 'comparison.mp4'), frames=decoded, fps=fps,
                    duration_seconds=n / fps, representative_frames=selected, active_frames=len(coords[0]),
                    scope='Full decode, source masks, actual inference boxes and skeleton diagnostics; no authored-room mesh validation'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--prompt-hmr', type=Path, default=post.BACKEND,
                   help='Optional legacy backend override; defaults to the packaged motion helpers')
    for name in ('video', 'pi3x-bundle', 'lifecycle-review',
                 'gvhmr-repo', 'pi3', 'pi3-checkpoint', 'smplx-model', 'python', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--sam3', type=Path, default=Path(os.environ.get('SAM3_ROOT', ROOT/'external/sam3')))
    p.add_argument('--sam3-python', type=Path, default=Path(os.environ.get('SAM3_PYTHON', ROOT/'.runtime/sam3-segmentation/venv/bin/python')))
    p.add_argument('--sam3-checkpoint', type=Path, default=Path(os.environ.get('SAM3_CHECKPOINT', ROOT/'.runtime/sam3-segmentation/checkpoints/sam3.pt')))
    p.add_argument('--actor-id', required=True)
    p.add_argument('--box', type=float, nargs=4, required=True)
    p.add_argument('--prompt-frame', type=int, default=0)
    p.add_argument('--gpu', type=int, help='Physical GPU index; default: inherited CUDA_VISIBLE_DEVICES, else device 0')
    p.add_argument('--scene-ground', type=Path, required=True,
                   help='Source-bound accepted floor/upright exported by tools.gvhmr.scene_ground')
    p.add_argument('--intrinsics', choices=('upstream', 'pi3x'), default='upstream',
                   help='GVHMR K_fullimg source: upstream estimate_K (default) or reviewed same-shot Pi3X intrinsics')
    p.add_argument('--kimodo-completion', action='store_true',
                   help='Opt in to experimental Kimodo occlusion completion (default: skipped)')
    p.add_argument('--completion-prompt', help='Observed action; required only with --kimodo-completion')
    p.add_argument('--completion-context', type=int, choices=(2, 3), default=3)
    p.add_argument('--kimodo-python', type=Path, default=ROOT/'.venv/bin/python')
    p.add_argument('--kimodo-checkpoint', type=Path, default=ROOT/'kimodo_blender/checkpoints/Kimodo-SMPLX-RP-v1')
    p.add_argument('--kimodo-upstream', type=Path, default=ROOT/'kimodo_blender/upstream')
    add_arguments(p)
    p = sub.add_parser('stage')
    p.add_argument('--kind', choices=('track', 'lifecycle', 'bodies', 'complete', 'camera', 'adapt', 'v2', 'measure', 'review'), required=True)
    p.add_argument('--params', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--input', action='append', default=[])
    a = parser.parse_args()
    prepare(a) if a.command == 'prepare' else stage(a)


if __name__ == '__main__':
    main()
