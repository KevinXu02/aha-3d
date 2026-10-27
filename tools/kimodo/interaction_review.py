"""Mesh intersection audit using Blender BVHs; diagnostics only, never a solver.

Run with Blender --background --python this_file -- collision --scene ...
--mesh ... --out ... . Closed cabinet poses and support transforms are respected.
Surface triangle intersections are not penetration depths or containment tests.
"""
import argparse
import gzip
import json
from pathlib import Path
import sys
import numpy as np


def collision(scene_path, mesh_path, out, stride=1):
    from mathutils.bvhtree import BVHTree
    from mathutils import Matrix, Quaternion, Vector
    raw=scene_path.read_bytes()
    scene=json.loads(gzip.decompress(raw) if scene_path.suffix=='.gz' else raw)
    cache=np.load(mesh_path);faces=cache['faces'].tolist()
    actor_ids={a['owner'] for a in scene.get('animation',{}).get('actors',[])}
    tracks={t['owner']:np.asarray(t['translations']) for t in scene.get('interaction',{}).get('object_tracks',[])}
    supports={o['instance_id']:o.get('support_id') for o in scene['objects']}
    def placement(owner,frame):
        shift=np.zeros(3);seen=set()
        while owner is not None:
            if owner in seen:raise ValueError('Cyclic support')
            seen.add(owner)
            if owner in tracks:shift+=tracks[owner][frame]
            owner=supports.get(owner)
        return shift
    joints={j['id']:j for j in scene['joints']}
    parts=[]
    for m in scene['meshes']:
        owner=m['owner']
        if owner in actor_ids:continue
        M=np.asarray(m['matrix']).reshape(4,4).T
        if m.get('joint'):
            j=joints[m['joint']]['closed'];q=j['quaternion']
            pose=Matrix.LocRotScale(Vector(j['position']),Quaternion((q[3],*q[:3])),Vector(j['scale']))
            M=np.asarray(pose)@M
        v=np.asarray(m['positions']).reshape(-1,3)@M[:3,:3].T+M[:3,3]
        f=np.asarray([i for g in m['groups'] for i in g['indices']]).reshape(-1,3)
        parts.append((owner,m['name'],v,f,BVHTree.FromPolygons(v.tolist(),f.tolist(),all_triangles=True)))
    reports={}
    for version,key in [('before','baseline_vertices'),('after','vertices')]:
        if key not in cache:continue
        vertices=cache[key];records=[]
        for frame in range(0,len(vertices),stride):
            v=vertices[frame];lo=v.min(0);hi=v.max(0)
            human=BVHTree.FromPolygons(v.tolist(),faces,all_triangles=True)
            hits={};locations={}
            for owner,name,p,tri,bvh in parts:
                shift=placement(owner,frame)
                if np.any(p.max(0)+shift<lo) or np.any(p.min(0)+shift>hi):continue
                body=human if not shift.any() else BVHTree.FromPolygons((v-shift).tolist(),faces,all_triangles=True)
                overlap=body.overlap(bvh)
                if overlap:
                    ids=sorted({a for a,b in overlap})
                    hits.setdefault(str(owner),set()).update(ids)
                    locations.setdefault(str(owner),[]).append(v[np.asarray(faces)[ids]].mean((0,1)).tolist())
            records.append(dict(frame=frame,time=frame/30,objects={o:dict(human_triangles=len(ids),location=np.mean(locations[o],axis=0).tolist()) for o,ids in hits.items()}))
            if frame%60==0:print(version,frame,flush=True)
        owners=sorted({o for r in records for o in r['objects']})
        summary={o:dict(intersection_frames=sum(o in r['objects'] for r in records),max_human_triangles=max(r['objects'].get(o,{}).get('human_triangles',0) for r in records)) for o in owners}
        reports[version]=dict(summary=summary,frames=records)
    result=dict(method='Blender BVHTree triangle surface intersections; no collision solver',stride=stride,
        scope='All exported static furniture with closed joints, animated target chair and full human triangles. Surface intersections include intended contact; no signed penetration depth or containment guarantee.',reports=reports)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v['summary'] for k,v in reports.items()},indent=2))


def rotation_report(stages, out):
    def angles(a,b):
        relative=a.astype(np.float64)@b.astype(np.float64).swapaxes(-1,-2)
        return np.where(np.all(a==b,axis=(-2,-1)),0.,np.rad2deg(np.arccos(np.clip((np.trace(relative,axis1=-2,axis2=-1)-1)/2,-1,1))))
    motions={name:dict(np.load(stages/file)) for name,file in [('raw','raw_segments.npz'),('joined','generated.npz'),('optimized','refined.npz')]}
    result={}
    for name,m in motions.items():
        r=m['local_rot_mats'];entry={}
        for joint,index in [('neck',12),('head',15)]:
            step=angles(r[1:,index],r[:-1,index]);change=angles(r[:,index],motions['joined']['local_rot_mats'][:,index])
            entry[joint]=dict(max_step_deg=float(step.max()),p95_step_deg=float(np.percentile(step,95)),max_change_from_joined_deg=float(change.max()),
                exact_equal_to_joined=bool(np.array_equal(r[:,index],motions['joined']['local_rot_mats'][:,index])))
        def global_rotation(local, chain):
            result=local[:,chain[0]]
            for joint in chain[1:]:result=result@local[:,joint]
            return result
        for joint,chain in [('world_neck',[0,3,6,9,12]),('world_head',[0,3,6,9,12,15])]:
            value=global_rotation(r,chain);reference=global_rotation(motions['joined']['local_rot_mats'],chain)
            step=angles(value[1:],value[:-1]);change=angles(value,reference)
            entry[joint]=dict(max_step_deg=float(step.max()),p95_step_deg=float(np.percentile(step,95)),max_change_from_joined_deg=float(change.max()),exact_equal_to_joined=bool(np.array_equal(value,reference)))
        result[name]=entry
    out.write_text(json.dumps(result,indent=2)+'\n')
    return result


def portal(run, stages='stages', viewer='viewer/demo-fast.html', collision_path='collision.json'):
    plan=json.loads((run/'snapshot/plan.json').read_text());segments=json.loads((run/stages/'segments.json').read_text())
    generation=json.loads((run/stages/'generation.json').read_text())
    # Historical runs used explicit height keys in both passes. Never relabel them.
    baseline_constraints=generation.get('pass_constraints',{}).get('baseline',
        ['Root2DConstraintSet']+(['Hips'] if generation['plan'].get('root_height_keys') else []))
    historical_hips=any('Hips' in item for item in baseline_constraints)
    baseline_description=('First-pass output using text, a 2D route and heading'
        + (' with historical full Hips constraints (position and rotation).' if historical_hips
           else ', without explicit pelvis-height constraints.'))
    rotations=rotation_report(run/stages,run/'rotation-report.json')
    motion=np.load(run/stages/'refined.npz');head=motion['posed_joints'][:,15]@np.array([[-1.,0,0],[0,0,1],[0,1,0]]).T
    hand=motion['posed_joints'][:,21]@np.array([[-1.,0,0],[0,0,1],[0,1,0]]).T
    data=dict(plan=plan,segments=segments,head=head.tolist(),hand=hand.tolist(),rotations=rotations,collision_path=collision_path)
    template=Path(__file__).with_name('interaction_review.html').read_text().replace('data-run-href=', 'href=').replace('data-run-src=', 'src=').replace('stages/',stages+'/').replace('viewer/demo-fast.html',viewer)
    receipt_path=run/stages/'constraint_receipt.json'
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text())['conditioned']
        heights=generation.get('pelvis_height_frames',[])
        native_hand='RightHandConstraintSet' in generation.get('constraints',[])
        hand_note=('Native hand conditioning includes root height and heading fitted from first-pass poses at contact frames; walking has no height targets.' if native_hand else 'Hand conditioning includes only hand position and rotation; walking has no height targets.')
        if not plan.get('hand_contact_seconds'):
            hand_note='This run has no hand-contact constraints.'
        if historical_hips:
            hand_note='The historical control retains sparse Hips position and rotation constraints throughout the clip; in the second pass, native hand context replaces duplicate Hips constraints at hand keyframes.'
        height_times=', '.join(f'{f/30:.2f}s' for f in heights) or 'none'
        route_count=len(set(receipt.get('smooth_root_2d',{}).get('indices',[])))
        heading_count=len(set(receipt.get('global_root_heading',{}).get('indices',[])))
        note=(f'<section><p>Inputs: {route_count} route waypoints and {heading_count} heading waypoints; '
              f'{"historical Hips sample times" if historical_hips else "independent seated-height targets"}: {height_times}. {hand_note} '
              'Segments continue to use 12 frames of context from the preceding generated motion.</p>'
              f'<a href="{stages}/constraint_receipt.json">View actual constraint frames and target values</a></section>')
        template=template.replace('<details><summary>Full process and files</summary>',note+'<details><summary>Full process and files</summary>')
    if not (run/stages/'hand_constraints.npz').exists():
        template=template.replace(f'<p><a href="{stages}/hand_constraints.npz">hand_constraints.npz</a>: Second-pass hand keyframe constraints.</p>','')
    if not (run/'old-collision.json').exists():
        template=template.replace(' · <a href="old-collision.json">Previous collision report</a>','')
    (run/'index.html').write_text(template.replace('/*BASELINE_DESCRIPTION*/',baseline_description).replace('/*REVIEW_DATA*/','window.REVIEW='+json.dumps(data).replace('<',r'\u003c')+';'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['collision','portal']);p.add_argument('--scene',type=Path);p.add_argument('--mesh',type=Path);p.add_argument('--out',type=Path);p.add_argument('--stride',type=int,default=1);p.add_argument('--run',type=Path)
    p.add_argument('--stages',default='stages');p.add_argument('--viewer',default='viewer/demo-fast.html');p.add_argument('--collision-report',default='collision.json')
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else None)
    if args.stage=='collision':
        if not all([args.scene,args.mesh,args.out]) or args.stride<1:p.error('collision requires --scene, --mesh, --out and a positive --stride')
        collision(args.scene,args.mesh,args.out,args.stride)
    else:
        if args.run is None:p.error('portal requires --run')
        portal(args.run,args.stages,args.viewer,args.collision_report)
