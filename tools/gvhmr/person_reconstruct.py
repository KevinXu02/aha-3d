#!/usr/bin/env python3
"""Fresh GVHMR inference with full-rate source-person boxes or explicit YOLO control.

Run with the dedicated GVHMR runtime, an exact normalized30Hz video, reviewed
actor, and hash-bound Pi3X bundle. Never consumes prior pose/feature/HMR caches.
"""
from pathlib import Path
import argparse,contextlib,importlib.util,json,os,random,shutil,sys,time

ROOT=Path(__file__).resolve().parents[2]

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--actor-id',required=True);p.add_argument('--pi3x-bundle',type=Path,required=True)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--person-masks',type=Path);group.add_argument('--baseline-track-id',type=int)
    p.add_argument('--gpu',type=int,help='Physical GPU index; default: inherited CUDA_VISIBLE_DEVICES, else device 0');p.add_argument('--seed',type=int,default=7)
    p.add_argument('--track-lifecycle',type=Path,help='Source-bound reviewed terminal-exit JSON; missing masks alone never stop tracking')
    p.add_argument('--repo',type=Path,default=ROOT/'.runtime/gvhmr-bedlam2')
    p.add_argument('--scene-ground',type=Path,help='Accepted scene floor/upright in the Pi3X camera world')
    p.add_argument('--intrinsics',choices=('upstream','pi3x'),default='upstream',help='GVHMR K_fullimg: upstream estimate_K (default) or the reviewed same-shot Pi3X intrinsics')
    from tools.gvhmr.pmpose import add_arguments
    add_arguments(p)
    return p

def main():
    a=parser().parse_args()
    if a.pose_detector == "pmpose" and a.person_masks is None:
        raise ValueError("PMPose requires source-person masks; choose --pose-detector vitpose for explicit YOLO control")
    for key in ("pmpose_python", "pmpose_root", "pmpose_checkpoint"):
        setattr(a, key, getattr(a, key).absolute())
    from tools.gvhmr.run import configure_execution
    env,execution=configure_execution(a.gpu)
    for k in ('CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER'):
        if k in env:os.environ[k]=env[k]
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
    import numpy as np,torch
    from tools.gvhmr.demo_entry import legacy_compatibility
    from tools.gvhmr.tracking_evidence import file_sha256,video_timeline,record_tracking_evidence
    from tools.gvhmr.camera_tracks import preload_pi3x_slam,verify_upstream
    from tools.gvhmr.person_tracking import prepare_gvhmr_boxes,forbid_default_tracker
    from tools.gvhmr.track_lifecycle import load_lifecycle,write_lossless_prefix,pad_inactive
    legacy_compatibility();repo=a.repo.resolve(strict=True);upstream=verify_upstream(repo)
    output=a.output.resolve();video=a.video.resolve(strict=True);bundle=a.pi3x_bundle.resolve(strict=True)
    if a.person_masks is not None:a.person_masks=a.person_masks.resolve(strict=True)
    if a.track_lifecycle is not None:a.track_lifecycle=a.track_lifecycle.resolve(strict=True)
    if a.scene_ground is not None:a.scene_ground=a.scene_ground.resolve(strict=True)
    output.mkdir(parents=True,exist_ok=False);timeline=video_timeline(video);source_hash=file_sha256(video)
    copied=output/'0_input_video.mp4';shutil.copyfile(video,copied);assert file_sha256(copied)==source_hash
    active,lifecycle=load_lifecycle(a.track_lifecycle,times=timeline['time_seconds'],image_size=timeline['image_size'],source_sha256=source_hash,actor_id=a.actor_id)
    end=int(active.sum());inference_video=copied;prefix=None
    if end<2:raise ValueError('GVHMR requires at least two active source frames')
    if end<len(active):
        inference_video=output/'active_input.avi';prefix=write_lossless_prefix(copied,inference_video,active)
    np.savez_compressed(output/'track_lifecycle.npz',track_active=active,source_frame_indices=np.arange(len(active)),time_seconds=timeline['time_seconds'],source_video_sha256=np.asarray(source_hash),source_actor_id=np.asarray(a.actor_id))
    sys.path.insert(0,str(repo));os.chdir(repo);torch.set_num_threads(4)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    from omegaconf import OmegaConf
    cfg=OmegaConf.load(repo/'hmr4d/configs/gvhmr_b1_demo.yaml');cfg.video_name=str(inference_video);cfg.output_dir=str(output);cfg.output_root=str(output);cfg.video_path=str(inference_video);cfg.static_cam=False;cfg.verbose=False;cfg.ckpt_path=str(repo/'inputs/checkpoints/gvhmr/gvhmr_b1b2.ckpt');OmegaConf.save(cfg,output/'demo.yaml');paths=cfg.paths;Path(cfg.preprocess_dir).mkdir()
    mode='default_yolo_explicit_actor_control'
    if a.person_masks:
        from tools.gvhmr.sam3_tracking import tracker_name
        with np.load(a.person_masks,allow_pickle=False) as archive:mode=tracker_name(archive).lower()
    report=dict(schema_version=1,status='running',tracker=mode,actor_id=a.actor_id,seed=a.seed,
        source_video_sha256=source_hash,normalized_input_sha256=source_hash,frames=timeline['frames'],source=str(video),camera_estimator='pi3x',
        intrinsics_estimator='unchanged upstream estimate_K' if a.intrinsics=='upstream' else 'reviewed Pi3X intrinsics (camera_adapter/camera_tracks.npz K_fullimg_pi3x)',upstream=upstream,execution=execution,
        implementation_sha256=file_sha256(__file__),preexisting_pose_feature_hmr_caches=False,
        source_actor_changed=True,body_scale=1,trajectory_scale_applied=1,postprocessing=True,upstream_postprocessing=True,timings_seconds={})
    report.update(lifecycle=lifecycle,active_inference_frames=end,source_frame_indices=list(range(len(active))),
        track_active=active.tolist(),active_source_prefix=prefix,
        inactive_padding='Repeat last inferred pose only for full-timeline storage; no tracking, feature extraction, inference or visible body after terminal exit.')
    (output/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    if a.person_masks:
        report['person_boxes']=prepare_gvhmr_boxes(a.person_masks,copied,a.actor_id,repo,Path(cfg.preprocess_dir),track_active=active,lifecycle=lifecycle)
        shutil.copyfile(Path(cfg.preprocess_dir)/'raw_tracking.json',output/'raw_tracking.json')
    report['camera']=preload_pi3x_slam(repo=repo,bundle=bundle,slam_path=paths.slam,output=output/'camera_adapter',times=timeline['time_seconds'],image_size=timeline['image_size'],source_sha256=source_hash)
    if end<len(active):
        shutil.copyfile(paths.slam,output/'camera_adapter/full_source_slam.pt')
        torch.save(torch.load(paths.slam,weights_only=False)[:end],paths.slam)
    for path in (paths.vitpose,paths.vit_features,paths.hmr4d_results):
        if Path(path).exists():raise FileExistsError('Fresh per-actor caches required: '+str(path))
    spec=importlib.util.spec_from_file_location('person_fresh_pinned_demo',repo/'tools/demo/demo.py');demo=importlib.util.module_from_spec(spec);spec.loader.exec_module(demo)
    calls={'vitpose_extract':0,'image_feature_extract':0,'pmpose_extract':0}
    report['pose_detector']=a.pose_detector
    if a.pose_detector == 'pmpose':
        from tools.gvhmr.pmpose import extract
        report['pmpose']=extract(a,copied,paths.bbx,paths.vitpose,end)
        calls['pmpose_extract']=1
        report['timings_seconds']['pmpose_extract']=report['pmpose']['seconds']
    originals=[]
    for cls,name,key in [(demo.VitPoseExtractor,'extract','vitpose_extract'),(demo.Extractor,'extract_video_features','image_feature_extract')]:
        original=getattr(cls,name);originals.append((cls,name,original))
        def make_wrapper(fn,key):
            def wrapped(self,*args,**kwargs):
                calls[key]+=1;t=time.monotonic();out=fn(self,*args,**kwargs);torch.cuda.synchronize();report['timings_seconds'][key]=time.monotonic()-t;return out
            return wrapped
        setattr(cls,name,make_wrapper(original,key))
    guard=forbid_default_tracker(demo.Tracker) if a.person_masks else record_tracking_evidence(output/'raw_tracking.json',selected_track_id=a.baseline_track_id,actor_id=a.actor_id,expected_video=inference_video,time_seconds=timeline['time_seconds'][:end])
    try:
        t=time.monotonic()
        with guard:demo.run_preprocess(cfg)
        torch.cuda.synchronize();report['timings_seconds']['preprocess']=time.monotonic()-t
    finally:
        for cls,name,original in originals:setattr(cls,name,original)
    expected={'vitpose_extract':int(a.pose_detector=='vitpose'),'image_feature_extract':1,'pmpose_extract':int(a.pose_detector=='pmpose')}
    if calls!=expected:raise RuntimeError('Fresh extraction did not execute exactly once')
    data=demo.load_data_dict(cfg);report['fresh_extraction_calls']=calls
    if a.intrinsics=='pi3x':
        K=np.load(output/'camera_adapter/camera_tracks.npz',allow_pickle=False)['K_fullimg_pi3x'][:end]
        report['intrinsics']=dict(source='pi3x',upstream_estimate_K=data['K_fullimg'][0].tolist(),fx_range=[float(K[:,0,0].min()),float(K[:,0,0].max())],principal_point_first=K[0,:2,2].tolist())
        data['K_fullimg']=torch.as_tensor(K,dtype=data['K_fullimg'].dtype)
    if not torch.equal(data['kp2d'].cpu(),torch.load(paths.vitpose,map_location='cpu',weights_only=True)):
        raise RuntimeError('GVHMR did not consume the selected detector keypoints')
    if int(data['length'])!=end or any(len(data[k])!=end for k in ('bbx_xys','kp2d','K_fullimg','cam_angvel','f_imgseq')):raise RuntimeError('Model inputs cross terminal lifecycle boundary')
    boxes=torch.load(paths.bbx,map_location='cpu',weights_only=True)
    assert torch.equal(data['bbx_xys'],boxes['bbx_xys'])
    np.savez_compressed(output/'actual_model_inputs.npz',**{k:v.detach().cpu().numpy() for k,v in data.items() if torch.is_tensor(v)},time_seconds=timeline['time_seconds'][:end],source_frame_indices=np.arange(end),track_active=active[:end],source_video_sha256=np.asarray(source_hash),actor_id=np.asarray(a.actor_id))
    t=time.monotonic();model=demo.DemoPL(pipeline=cfg.model.pipeline);model.load_pretrained_model(cfg.ckpt_path);model=model.eval().cuda();torch.cuda.synchronize();report['timings_seconds']['model_load']=time.monotonic()-t
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    t=time.monotonic()
    ground_context=contextlib.nullcontext(None)
    if a.scene_ground:
        from tools.gvhmr.scene_ground import load_prior,gvhmr_scene_context
        prior=load_prior(a.scene_ground,source_hash)
        if prior['evidence'].get('camera_sha256') != file_sha256(bundle/'cameras.json'):
            raise ValueError('Scene prior uses a different camera bundle/world basis')
        cameras=np.load(output/'camera_adapter/camera_tracks.npz',allow_pickle=False)['c2w'][:end]
        ground_context=gvhmr_scene_context(cameras,prior)
    with ground_context as scene_report, torch.inference_mode():
        pred=demo.detach_to_cpu(model.predict(data,static_cam=False))
    if scene_report is not None:
        if scene_report['rollout_calls']!=1 or scene_report['postprocess_calls']!=1:
            raise RuntimeError('Scene prior did not replace the expected GVHMR world/ground steps')
        report['scene_ground']=scene_report
        report['scene_ground']['implementation_sha256']=file_sha256(Path(__file__).with_name('scene_ground.py'))
    torch.cuda.synchronize();report['timings_seconds']['inference']=time.monotonic()-t
    if end<len(active):
        # Preserve actual short model outputs/caches separately. Full timeline files
        # below are explicit masked exports, never caches consumed by the model.
        actual=output/'active_inference';actual.mkdir();torch.save(pred,actual/'hmr4d_results.pt')
        shutil.move(str(cfg.preprocess_dir),str(actual/'preprocess'));Path(cfg.preprocess_dir).mkdir()
        full_boxes={k:torch.from_numpy(pad_inactive(v.numpy(),active,fill='nan')) for k,v in boxes.items()}
        torch.save(full_boxes,paths.bbx)
        for name,target in [('vitpose',paths.vitpose),('vit_features',paths.vit_features)]:
            old=torch.load(actual/'preprocess'/Path(target).name,map_location='cpu',weights_only=True)
            torch.save(torch.from_numpy(pad_inactive(old.numpy(),active,fill='zero')),target)
        shutil.copyfile(output/'camera_adapter/full_source_slam.pt',paths.slam)
        if not a.person_masks:
            from tools.gvhmr.tracking_evidence import build_tracking_evidence,array_sha256
            raw=json.loads((output/'raw_tracking.json').read_text());(actual/'raw_tracking.json').write_text(json.dumps(raw,indent=2)+'\n')
            raw=build_tracking_evidence(raw['raw_history']+[[]]*(len(active)-end),timeline['time_seconds'],timeline['image_size'],source_path=copied,source_sha256=source_hash,selected_track_id=a.baseline_track_id,actor_id=a.actor_id)
            raw.update(status='dense_boxes_returned',dense_bbx_xyxy_sha256=array_sha256(full_boxes['bbx_xyxy'].numpy()),track_active=active.tolist(),source_frame_indices=list(range(len(active))),lifecycle=lifecycle)
            (output/'raw_tracking.json').write_text(json.dumps(raw,indent=2)+'\n')
        pred={group:{key:torch.from_numpy(pad_inactive(val.numpy(),active)) for key,val in pred[group].items()} for group in ('smpl_params_global','smpl_params_incam')}|{'K_fullimg':torch.from_numpy(pad_inactive(data['K_fullimg'].cpu().numpy(),active))}
    pred.update(track_active=torch.from_numpy(active),source_frame_indices=torch.arange(len(active)))
    torch.save(pred,paths.hmr4d_results)
    arrays=dict(K_fullimg=pred['K_fullimg'].numpy(),frame_times_seconds=timeline['time_seconds'],fps=np.asarray(30.),source_video_sha256=np.asarray(source_hash),source_actor_id=np.asarray(a.actor_id),track_active=active,source_frame_indices=np.arange(len(active)),inactive_padding=np.asarray(report['inactive_padding']))
    for group in ('smpl_params_global','smpl_params_incam'):
        for key,val in pred[group].items():arrays[group+'__'+key]=val.numpy()
    np.savez_compressed(output/'motion_native.npz',**arrays)
    report.update(status='completed',default_tracker_forbidden=bool(a.person_masks),quality_accepted=False,
        output_sha256={str(Path(x).relative_to(output)):file_sha256(x) for x in [paths.bbx,paths.vitpose,paths.vit_features,paths.slam,paths.hmr4d_results,output/'motion_native.npz',output/'actual_model_inputs.npz',output/'raw_tracking.json']},checkpoint_sha256=file_sha256(cfg.ckpt_path))
    assert file_sha256(copied)==source_hash
    (output/'provenance.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(status='completed',output=str(output),tracker=mode,calls=calls,timings=report['timings_seconds'])),flush=True)

if __name__=='__main__':main()
