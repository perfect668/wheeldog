"""Minimal H1/cuff/FEM/passive-arm integration attempt. Never labels axial advance as dressing success."""
import argparse,json,sys,time,os,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,'/workspace/PhyRC_Sim')
p=argparse.ArgumentParser();p.add_argument('--gui',action='store_true');p.add_argument('--disable-human-collision',action='store_true');p.add_argument('--open-cuff',action='store_true');p.add_argument('--diagnostic-unattached',action='store_true');p.add_argument('--diagnostic-filter-robot',action='store_true');p.add_argument('--steps',type=int,default=2880);p.add_argument('--stiffness',type=float,default=4.);a,_=p.parse_known_args()
if not 1<=a.steps<=2880:raise ValueError('Diagnostic step budget must be in [1,2880]')
from isaacsim import SimulationApp
app=SimulationApp({'headless':not a.gui})
try:
    import torch
    from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema,UsdLux,Gf
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage
    from dynadress.human.passive_arm import build,Impedance,JOINT_NAMES
    from dynadress.robots.h1 import H1Kinematics
    from dynadress.garments.cuff import boundary_loops,progress
    from dynadress.recording.episode import Episode
    from Env_Config.Garment.Particle_Garment import Particle_Garment
    from Policy.sensors import RGBDSensors
    from scipy.spatial.transform import Rotation
    from PIL import Image
    w=World(physics_dt=1/240,rendering_dt=1/60,backend='torch',device='cuda:0');pc=w.get_physics_context();pc.enable_gpu_dynamics(True);pc.set_broadphase_type('GPU');pc.set_solver_type('TGS')
    stage=w.stage;UsdLux.DomeLight.Define(stage,'/World/Light').CreateIntensityAttr(1000.)
    add_reference_to_stage((ROOT/'assets/h1/generated/usd_path.txt').read_text().strip(),'/World/H1')
    for _ in range(4):
        paths=[str(v.GetPath()) for v in Usd.PrimRange(stage.GetPrimAtPath('/World/H1'),Usd.TraverseInstanceProxies()) if v.IsInstance()]
        for path in paths:
            v=stage.GetPrimAtPath(path)
            if v and not v.IsInstanceProxy():v.SetInstanceable(False)
    for v in Usd.PrimRange(stage.GetPrimAtPath('/World/H1')):
        if v.HasAPI(UsdPhysics.CollisionAPI) and 'tool_' in str(v.GetPath()):
            collision=PhysxSchema.PhysxCollisionAPI.Apply(v);collision.CreateContactOffsetAttr(.002);collision.CreateRestOffsetAttr(.0002)
        if v.IsA(UsdGeom.Mesh) and v.HasAPI(UsdPhysics.CollisionAPI):UsdPhysics.MeshCollisionAPI.Apply(v).CreateApproximationAttr('convexDecomposition')
        if v.HasAPI(UsdPhysics.ArticulationRootAPI):
            api=PhysxSchema.PhysxArticulationAPI.Apply(v);api.CreateEnabledSelfCollisionsAttr(True);api.CreateSolverPositionIterationCountAttr(64);api.CreateSolverVelocityIterationCountAttr(16)
    build(stage,shoulder=(1.3,0.,1.45));imp=Impedance(stiffness=(a.stiffness,)*4,damping=(1.,)*4,reference=(0.,1.2,0.,.3))
    if a.disable_human_collision:
        for v in Usd.PrimRange(stage.GetPrimAtPath('/World/PassiveArm')):
            if v.HasAPI(UsdPhysics.CollisionAPI):UsdPhysics.CollisionAPI(v).GetCollisionEnabledAttr().Set(False)
    human=w.scene.add(SingleArticulation('/World/PassiveArm',name='human'));robot=w.scene.add(SingleArticulation('/World/H1',name='h1'))
    cloth=Particle_Garment(w,usd_path='/workspace/PhyRC_Sim/Assets/Garment/Tops/Modelink/t_shirt_short.usd',pos=np.zeros(3),ori=np.array([0.,180.,180.]),scale=np.ones(3)*1.08,solver_position_iteration_count=64,friction=.3)
    mesh=UsdGeom.Mesh(stage.GetPrimAtPath(cloth.garment_mesh_prim_path));tri=np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
    if a.diagnostic_filter_robot:
        if not a.diagnostic_unattached:raise ValueError('Robot collision filtering is only allowed for unattached diagnostic')
        bodies=[v.GetPath() for v in Usd.PrimRange(stage.GetPrimAtPath('/World/H1')) if v.HasAPI(UsdPhysics.RigidBodyAPI)]
        UsdPhysics.FilteredPairsAPI.Apply(mesh.GetPrim()).CreateFilteredPairsRel().SetTargets(bodies)
    local=np.array(mesh.GetPointsAttr().Get());tf=np.array(UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(Usd.TimeCode.Default()));points=(np.c_[local,np.ones(len(local))]@tf)[:,:3]
    loops=boundary_loops(tri);cuff=min(loops,key=lambda ids:points[ids].mean(axis=0)[0]);offset=np.array([.55,0.,1.31])-points[cuff].mean(axis=0)
    cloth.set_pose(offset,None);points+=offset
    anchors={s:cuff[np.argsort(np.linalg.norm(points[cuff]-points[cuff[np.argmax(sign*points[cuff,1])]],axis=1))[:12]] for s,sign in [('L',1),('R',-1)]}
    # The rest garment is flattened: Z extrema can select almost the same point.
    # Opposite Y extrema define disjoint physical sides before opening in Z.
    if set(anchors['L']) & set(anchors['R']):raise RuntimeError('Cuff grasp patches overlap')
    if np.linalg.norm(points[anchors['L']].mean(axis=0)-points[anchors['R']].mean(axis=0))<.06:raise RuntimeError('Cuff grasp patches separated by less than tool clearance')
    model=H1Kinematics(ROOT/'assets/h1/generated/h1_task.urdf');q0={n:(.5 if n.endswith('_4') else 0.) for n in model.names};goals={}
    # Tool Z approaches along world X; jaw Y straddles the flattened shirt in Z.
    rotation=Rotation.from_euler('y',np.pi/2).as_matrix()@Rotation.from_euler('z',np.pi/2).as_matrix()
    for side in 'LR':
        goal=np.eye(4);goal[:3,:3]=rotation;goal[:3,3]=points[anchors[side]].mean(axis=0)-rotation@np.array([0,0,.07]);goals[side]=goal
        solution,error=model.solve(side,goal,q0,300)
        if error['position_error_m']>.003 or error['rotation_error_rad']>.03:raise RuntimeError(f'Initial cuff pose IK failed: {side} {error}')
        q0.update(solution)
    from pxr import PhysicsSchemaTools
    import omni.physx
    contacts=set()
    for v in Usd.PrimRange(stage.GetPrimAtPath('/World/H1')):
        if v.HasAPI(UsdPhysics.RigidBodyAPI):PhysxSchema.PhysxContactReportAPI.Apply(v).CreateThresholdAttr(0.)
        if v.IsA(UsdPhysics.RevoluteJoint):PhysxSchema.PhysxJointAPI.Apply(v).CreateMaxJointVelocityAttr(float(np.degrees(.5)))
        elif v.IsA(UsdPhysics.PrismaticJoint):PhysxSchema.PhysxJointAPI.Apply(v).CreateMaxJointVelocityAttr(.05)
    def contact_report(headers,data):
        for h in headers:contacts.add((str(PhysicsSchemaTools.intToSdfPath(h.actor0)),str(PhysicsSchemaTools.intToSdfPath(h.actor1))))
    subscription=omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(contact_report)
    w.reset();human.initialize();robot.initialize()
    dev=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda:0')
    cpu=lambda x:x.detach().cpu().numpy().copy()
    human_ids=np.array([human.get_dof_index(n) for n in JOINT_NAMES]);hq=np.zeros(4);hq[human_ids]=imp.reference;human.set_joint_positions(dev(hq));human.set_joint_velocities(dev(np.zeros(4)));human._articulation_view.set_gains(kps=dev(np.zeros((1,4))),kds=dev(np.ones((1,4))))
    mapping={}
    for n in model.names:
        side='L' if '08L' in n else 'R';i=n.rsplit('_',1)[-1];mapping[n]=next(v for v in robot.dof_names if '08'+side in v and '_joint_'+i+'_' in v)
    rq=cpu(robot.get_joint_positions())
    for n in model.names:rq[robot.get_dof_index(mapping[n])]=q0[n]
    for n in robot.dof_names:
        if '_jaw_' in n:rq[robot.get_dof_index(n)]=.006
    robot.set_joint_positions(dev(rq));robot.set_joint_velocities(dev(np.zeros(robot.num_dof)));robot._articulation_view.set_gains(kps=dev(np.full((1,robot.num_dof),200.)),kds=dev(np.full((1,robot.num_dof),15.)));robot._articulation_view.set_joint_position_targets(dev(rq[None,:]))
    human.set_joint_efforts(human._articulation_view.get_generalized_gravity_forces()[0]);robot.set_joint_efforts(robot._articulation_view.get_generalized_gravity_forces()[0]);w.step(render=False)
    out=ROOT/'artifacts/m1/coupled_pull'/time.strftime('%Y%m%dT%H%M%S');out.mkdir(parents=True)
    link_names=robot._articulation_view.body_names;link_indices={s:next(i for i,n in enumerate(link_names) if '08'+s in n and '_link7_' in n) for s in 'LR'}
    initial=cloth.get_vertice_positions();link_tfs=cpu(robot._articulation_view._physics_view.get_link_transforms())[0]
    for side in 'LR':
        idx=link_indices[side];pose=link_tfs[idx];rot=Rotation.from_quat(pose[3:]).as_matrix();ids=anchors[side];local_anchors=(initial[ids]-pose[:3])@rot
        actual_q={n:float(cpu(robot.get_joint_positions())[robot.get_dof_index(mapping[n])]) for n in model.names}
        relative=np.linalg.inv(model.fk(actual_q,f'AR5-5_08{side}-W4C1C9-ZY2_link7'))@model.fk(actual_q,'tool_'+side)
        live_tf=np.eye(4);live_tf[:3,:3]=rot;live_tf[:3,3]=pose[:3]
        grip_center=(live_tf@relative@np.array([0,0,.07,1]))[:3]
        if float(np.linalg.norm(initial[ids].mean(axis=0)-grip_center))>.025:
            raise RuntimeError(f'Initial physical grasp proximity rejected: {side}; cuff too far from tool')
        if not a.diagnostic_unattached:
            prim=stage.DefinePrim('/World/Grasp'+side,'OmniPhysicsVtxXformAttachment');prim.GetRelationship('omniphysics:src0').SetTargets([cloth.garment_mesh_prim_path]);prim.GetRelationship('omniphysics:src1').SetTargets([robot._articulation_view._physics_view.link_paths[0][idx]]);prim.GetAttribute('omniphysics:vtxIndicesSrc0').Set(ids.tolist());prim.GetAttribute('omniphysics:localPositionsSrc1').Set([Gf.Vec3f(*map(float,v)) for v in local_anchors])
    meta={'scope':'minimal H1/cuff/passive-arm integration attempt; no seated-manikin integration','seed':0,'physics_hz':240,'control_hz':20,'sample_hz':10,'stiffness':list(imp.stiffness),'damping':list(imp.damping),'q_ref':list(imp.reference),'human_collision_enabled':not a.disable_human_collision,'original_state_writes':'initialization only','source_commit':os.environ.get('PHYRC_SOURCE_COMMIT'),'grasp_indices':{s:v.tolist() for s,v in anchors.items()},'sim_only_gripper':True,'high_dim_sampling':'10 Hz cloth NPZ and RGB; fixed diagnostic budget <=120 frames','isaac_version':Path('/root/isaacsim/VERSION').read_text().strip()}
    meta['diagnostic_unattached']=a.diagnostic_unattached
    meta['diagnostic_filter_robot']=a.diagnostic_filter_robot
    episode=Episode(out,meta);sensors=RGBDSensors(stage,w,[],{'default_dimensions':{'width':800,'height':600},'cameras':[{'id':'integration','mount':'world','eye_world_m':[1.8,-2.,2.1],'target_world_m':[.65,0,1.2],'horizontal_fov_deg':60,'clip_m':[.05,20]}]})
    desired=dict(q0);commanded_goals={s:v.copy() for s,v in goals.items()};result={'success':False,'termination_reason':None,'scope':meta['scope']};start=float(w.current_time)
    for step in range(a.steps):
        qh=cpu(human.get_joint_positions())[human_ids];dqh=cpu(human.get_joint_velocities())[human_ids];gravity=cpu(human._articulation_view.get_generalized_gravity_forces())[0,human_ids];hold,_=imp.effort(qh,np.zeros(4),gravity);eff=np.zeros(4);eff[human_ids]=hold;human.set_joint_efforts(dev(eff))
        if step%12==0:
            advance=min(max(step/240-(3. if a.open_cuff else 1.),0.)*.02,.18)
            for side in 'LR':
                target=goals[side].copy();target[0,3]+=advance
                if a.open_cuff:target[2,3]+=(1 if side=='L' else -1)*min(step/240/7.,1.)*.07
                commanded_goals[side]=target.copy();solution,error=model.solve(side,target,desired,30)
                if error['position_error_m']>.005:result['termination_reason']='ik_infeasible';break
                desired.update(solution)
            if result['termination_reason']:break
            current=cpu(robot.get_joint_positions());current_dict={n:float(current[robot.get_dof_index(mapping[n])]) for n in model.names};cmd=model.bounded_step(current_dict,desired,1/20,speed=.25)
            for n,value in cmd.items():rq[robot.get_dof_index(mapping[n])]=value
            robot._articulation_view.set_joint_position_targets(dev(rq[None,:]))
        robot.set_joint_efforts(robot._articulation_view.get_generalized_gravity_forces()[0])
        if step%24==0:
            pos=cloth.get_vertice_positions();qrobot=cpu(robot.get_joint_positions());finite=np.isfinite(pos).all() and np.isfinite(qrobot).all() and np.isfinite(qh).all()
            if not finite:result['termination_reason']='nonfinite';break
            joint_viol=[n for n in model.names if not model.limits[n][0]-.001<=qrobot[robot.get_dof_index(mapping[n])]<=model.limits[n][1]+.001]
            live=cpu(robot._articulation_view._physics_view.get_link_transforms())[0];grasp_error={}
            for side in 'LR':
                if a.diagnostic_unattached:continue
                pose=live[link_indices[side]];idx=anchors[side];attachment=stage.GetPrimAtPath('/World/Grasp'+side);local=np.array(attachment.GetAttribute('omniphysics:localPositionsSrc1').Get());expected=local@Rotation.from_quat(pose[3:]).as_matrix().T+pose[:3];grasp_error[side]=float(np.max(np.linalg.norm(pos[idx]-expected,axis=1)))
            tracking={}
            for side in 'LR':
                actual={n:float(qrobot[robot.get_dof_index(mapping[n])]) for n in model.names}
                tracking[side]=float(np.linalg.norm(model.fk(actual,'tool_'+side)[:3,3]-commanded_goals[side][:3,3]))
            human_tf=cpu(human._articulation_view._physics_view.get_link_transforms())[0,human._articulation_view.body_names.index('forearm')]
            human_rotation=Rotation.from_quat(human_tf[3:]).as_matrix();hand_tip=human_tf[:3]+human_rotation@np.array([0,0,-.385]);elbow=human_tf[:3]
            progress_proxy=progress(pos[cuff],hand_tip,elbow)
            if max(tracking.values())>.03:result['termination_reason']='end_effector_tracking_error'
            elif joint_viol:result['termination_reason']='robot_joint_limit'
            elif max(grasp_error.values(),default=0.)>.03:result['termination_reason']='grasp_lost'
            episode.write(int(w.current_time_step_index),step//12,float(w.current_time),action={'end_effector_targets':{s:v.tolist() for s,v in commanded_goals.items()},'advance_m':advance,'gripper_command':'hold'},solver_state={'human_q_rad':qh,'human_dq_rad_s':dqh,'robot_q':qrobot,'robot_dq':cpu(robot.get_joint_velocities()),'cuff_keypoints_m':pos[cuff],'grasp_error_m':grasp_error,'end_effector_tracking_error_m':tracking,'human_hand_tip_m':hand_tip,'human_elbow_m':elbow,'cuff_progress':progress_proxy},commanded_wrench={'human_explicit_Nm':hold,'gravity_compensation_Nm':gravity},privileged={'K':imp.stiffness,'D':imp.damping,'q_ref':imp.reference})
            np.savez_compressed(out/f'cloth_{step:06d}.npz',positions=pos);imgs,_=sensors.capture();Image.fromarray(imgs['rgb'][0]).save(out/f'frame_{step//24:04d}.png')
            if result['termination_reason']:break
            if step%240==0:print('PULL_STEP',step,'GRASP_ERROR',grasp_error,'HUMAN_Q',qh.tolist(),flush=True)
        w.step(render=False)
    if result['termination_reason'] is None:result['termination_reason']='trajectory_ended_without_validated_sleeve_containment'
    result['rigid_contact_pairs']=sorted(contacts)
    result['simulation_duration_s']=float(w.current_time)-start;(out/'result.json').write_text(json.dumps(result,indent=2));episode.close('failure',result['termination_reason']);sensors.close();print('M1-COUPLED-PULL-RESULT',out,json.dumps(result),flush=True);w.stop()
except BaseException:
    traceback.print_exc();app.close(exit_code=1)
finally:app.close()
