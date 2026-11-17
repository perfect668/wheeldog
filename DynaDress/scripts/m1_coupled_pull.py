"""M1.1 deterministic H1 sleeve task. Success requires measured cuff containment.

Only initialization writes cloth nodes/human joint positions. Native FEM, rigid
articulation dynamics and native vertex attachments handle every subsequent step.
"""
import argparse,json,sys,time,os,traceback,hashlib,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),'/workspace/PhyRC_Sim']
p=argparse.ArgumentParser()
p.add_argument('--config',type=Path,default=ROOT/'dynadress/configs/m11_sleeve.json')
p.add_argument('--gui',action='store_true');p.add_argument('--seconds',type=float)
p.add_argument('--stiffness',type=float);p.add_argument('--offset-y',type=float,default=0.)
p.add_argument('--tag',default='nominal');p.add_argument('--no-rgb',action='store_true')
a,_=p.parse_known_args();cfg=json.loads(a.config.read_text())
if a.seconds is not None:cfg['max_seconds']=a.seconds
if a.stiffness is not None:cfg['human']['stiffness']=[a.stiffness]*4
from dynadress.robots.h1 import H1Kinematics
from dynadress.garments.cuff import boundary_loops,opened_preset,grasp_patches,opening,contains_section
from dynadress.human.passive_arm import Impedance,JOINT_NAMES
from dynadress.recording.episode import Episode,serial
from scipy.spatial.transform import Rotation
# Geometry preflight runs before starting Kit. Exported from the same pinned asset
# and exact refinement level using m11_prepare_geometry.py; checked again live.
geo=np.load(ROOT/'assets/h1/generated/m11_garment_geometry.npz');rest=geo['points'];tri=geo['triangles']
cuff=min(boundary_loops(tri),key=lambda ids:rest[ids,0].mean())
ref=np.array(cfg['human']['reference']);shoulder=np.array(cfg['human']['shoulder'])
upper_rot=Rotation.from_euler('XYZ',ref[:3]).as_matrix()
arm_rot=upper_rot@Rotation.from_euler('y',ref[3]).as_matrix()
axis=arm_rot@np.array([0.,0.,1.]);elbow=shoulder+upper_rot@np.array([0,0,-.30])
tip=elbow-.385*axis;center=tip-cfg['initial_hand_gap_m']*axis+np.array([0,a.offset_y,0])
preset,preset_info=opened_preset(rest,cuff,center,axis)
anchors=grasp_patches(preset,cuff,cfg['grasp_patch_count'],triangles=tri)
model=H1Kinematics(ROOT/'assets/h1/generated/h1_task.urdf')
q0={n:(.5 if n.endswith('_4') else 0.) for n in model.names};goals={};ik_info={}
nominal_distance=cfg['initial_hand_gap_m']+.125+.26*cfg['target_forearm_fraction']
for side in 'LR':
    goal=np.eye(4);goal[:3,:3]=arm_rot@Rotation.from_euler('z',cfg['wrist_roll_deg'][side],degrees=True).as_matrix()
    goal[:3,3]=preset[anchors[side]].mean(0)-goal[:3,:3]@np.array([0,0,.07])
    path=[]
    for distance in np.linspace(0,nominal_distance,20):
        waypoint=goal.copy();waypoint[:3,3]+=distance*axis;path.append(waypoint)
    sol,info=model.solve_candidates(side,goal,q0,count=48,seed=cfg['seed'],margin=.1,waypoints=path,min_elbow_outward=.25)
    q0.update(sol);goals[side]=goal;ik_info[side]=info
out=ROOT/'artifacts/m11'/f'{time.strftime("%Y%m%dT%H%M%S")}_{a.tag}'
out.mkdir(parents=True);(out/'config.json').write_text(json.dumps(cfg,indent=2))
print('M11_PREFLIGHT',json.dumps(ik_info),'OUTPUT',out,flush=True)
from isaacsim import SimulationApp
app=SimulationApp({'headless':not a.gui})
result={'success':False,'termination_reason':None,'transitions':[],'scope':'real ROKAE H1, simulation grippers; one-time pregrasp preset; original PhyRC FEM'}
episode=None;sensors=None
try:
    import torch
    from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema,UsdLux,Gf,PhysicsSchemaTools
    import omni.physx
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage
    from dynadress.human.passive_arm import build
    from Env_Config.Garment.Particle_Garment import Particle_Garment
    from Policy.sensors import RGBDSensors
    from PIL import Image
    w=World(physics_dt=1/240,rendering_dt=1/60,backend='torch',device='cuda:0')
    pc=w.get_physics_context();pc.enable_gpu_dynamics(True);pc.set_broadphase_type('GPU');pc.set_solver_type('TGS')
    stage=w.stage;UsdLux.DomeLight.Define(stage,'/World/Light').CreateIntensityAttr(700.)
    add_reference_to_stage((ROOT/'assets/h1/generated/usd_path.txt').read_text().strip(),'/World/H1')
    for _ in range(4):
        for path in [str(v.GetPath()) for v in Usd.PrimRange(stage.GetPrimAtPath('/World/H1'),Usd.TraverseInstanceProxies()) if v.IsInstance()]:
            v=stage.GetPrimAtPath(path)
            if v and not v.IsInstanceProxy():v.SetInstanceable(False)
    contacts=set()
    for v in Usd.PrimRange(stage.GetPrimAtPath('/World/H1')):
        if v.HasAPI(UsdPhysics.CollisionAPI) and 'tool_' in str(v.GetPath()):
            col=PhysxSchema.PhysxCollisionAPI.Apply(v);col.CreateContactOffsetAttr(.002);col.CreateRestOffsetAttr(.0002)
        if v.IsA(UsdGeom.Mesh) and v.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.MeshCollisionAPI.Apply(v).CreateApproximationAttr('convexDecomposition')
        if v.HasAPI(UsdPhysics.ArticulationRootAPI):
            api=PhysxSchema.PhysxArticulationAPI.Apply(v);api.CreateEnabledSelfCollisionsAttr(True)
            api.CreateSolverPositionIterationCountAttr(64);api.CreateSolverVelocityIterationCountAttr(16)
        if v.HasAPI(UsdPhysics.RigidBodyAPI):PhysxSchema.PhysxContactReportAPI.Apply(v).CreateThresholdAttr(0.)
        if v.IsA(UsdPhysics.RevoluteJoint):PhysxSchema.PhysxJointAPI.Apply(v).CreateMaxJointVelocityAttr(float(np.degrees(cfg['joint_speed_rad_s'])))
        elif v.IsA(UsdPhysics.PrismaticJoint):PhysxSchema.PhysxJointAPI.Apply(v).CreateMaxJointVelocityAttr(.05)
    # Links 5/6/7 share the same wrist pivot in this CAD URDF. Non-adjacent
    # 5/7 mesh shells overlap inside that mechanism; filter just that pair.
    wrist_filters=[]
    for s in 'LR':
        bodies={i:next(v for v in Usd.PrimRange(stage.GetPrimAtPath('/World/H1')) if v.HasAPI(UsdPhysics.RigidBodyAPI) and '08'+s in str(v.GetPath()) and '_link'+str(i)+'_' in str(v.GetPath())) for i in (5,7)}
        UsdPhysics.FilteredPairsAPI.Apply(bodies[5]).CreateFilteredPairsRel().SetTargets([bodies[7].GetPath()])
        wrist_filters.append([str(bodies[i].GetPath()) for i in (5,7)])
    def contact_report(headers,data):
        for h in headers:contacts.add((str(PhysicsSchemaTools.intToSdfPath(h.actor0)),str(PhysicsSchemaTools.intToSdfPath(h.actor1))))
    subscription=omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(contact_report)
    build(stage,shoulder=shoulder)
    imp=Impedance(stiffness=tuple(cfg['human']['stiffness']),damping=tuple(cfg['human']['damping']),reference=tuple(ref))
    human=w.scene.add(SingleArticulation('/World/PassiveArm',name='human'))
    robot=w.scene.add(SingleArticulation('/World/H1',name='h1'))
    w.reset();human.initialize();robot.initialize()
    dev=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda:0')
    cpu=lambda x:x.detach().cpu().numpy().copy()
    hids=np.array([human.get_dof_index(n) for n in JOINT_NAMES]);hq=np.zeros(4);hq[hids]=ref
    human.set_joint_positions(dev(hq));human.set_joint_velocities(dev(np.zeros(4)))
    kd=np.zeros(4);kd[hids]=imp.damping
    human._articulation_view.set_gains(kps=dev(np.zeros((1,4))),kds=dev(kd[None,:]))
    mapping={}
    for n in model.names:
        side='L' if '08L' in n else 'R';i=n.rsplit('_',1)[-1]
        mapping[n]=robot.get_dof_index(next(v for v in robot.dof_names if '08'+side in v and '_joint_'+i+'_' in v))
    rq=cpu(robot.get_joint_positions())
    for n in model.names:rq[mapping[n]]=q0[n]
    for n in robot.dof_names:
        if '_jaw_' in n:rq[robot.get_dof_index(n)]=0. # fingers straddle the attached patch, minimal inward intrusion
    robot.set_joint_positions(dev(rq));robot.set_joint_velocities(dev(np.zeros(robot.num_dof)))
    robot._articulation_view.set_gains(kps=dev(np.full((1,robot.num_dof),cfg['robot_gains']['stiffness'])),kds=dev(np.full((1,robot.num_dof),cfg['robot_gains']['damping'])))
    robot._articulation_view.set_joint_position_targets(dev(rq[None,:]))
    link_indices={s:next(i for i,n in enumerate(robot._articulation_view.body_names) if '08'+s in n and '_link7_' in n) for s in 'LR'}
    forearm_index=human._articulation_view.body_names.index('forearm')
    meta={'config':cfg,'seed':cfg['seed'],'offset_y_m':a.offset_y,'physics_hz':240,'control_hz':20,'sample_hz':20,'rgb_hz':cfg['rgb_hz'],
          'scope':result['scope'],'ik_preflight':ik_info,'preset':preset_info,'grasp_indices':{s:v.tolist() for s,v in anchors.items()},
          'state_writes':'human and cloth initialization only; no later node/pose overwrites','self_collision_enabled':True,
          'cloth_collision_filter':'none','robot_internal_wrist_filtered_pairs':wrist_filters,'sim_only_gripper':True,'human_position_stiffness':0,'human_damping':'implicit velocity drive',
          'upstream_commit':os.environ.get('PHYRC_SOURCE_COMMIT'),'isaac_version':Path('/root/isaacsim/VERSION').read_text().strip(),
          'project_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
          'project_dirty':bool(subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()),
          'code_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),ROOT/'dynadress/robots/h1.py',ROOT/'dynadress/garments/cuff.py']},
          'high_dim_sampling':'cloth initial/final only; RGB configurable; all fields with missing force marked unavailable'}
    episode=Episode(out,meta)
    if not a.no_rgb:
        sensors=RGBDSensors(stage,w,[],{'default_dimensions':{'width':800,'height':600},'cameras':[{'id':'task','mount':'world','eye_world_m':[1.5,-1.0,1.95],'target_world_m':[.5,0,1.55],'horizontal_fov_deg':55,'clip_m':[.05,20]}]})
    cloth=None;state='robot_hold';state_since=0.;command_center=center.copy();commanded={s:v.copy() for s,v in goals.items()};desired=dict(q0)
    local_anchors={};tip_entered=False;wrist_entered=False;success_count=0;retries=0;last_progress=-1e3;last_motion=0.;retreat_goal=None
    previous_commanded={s:v.copy() for s,v in commanded.items()}
    warm_max={s:0. for s in 'LR'};hold_max={s:0. for s in 'LR'};initial_time=float(w.current_time);frame=0;geometry=None
    def transition(new,t,reason):
        global state,state_since
        result['transitions'].append({'from':state,'to':new,'simulation_s':t,'reason':reason});state=new;state_since=t
        print('M11_STATE',new,t,reason,flush=True)
    for step in range(int(cfg['max_seconds']*240)):
        t=float(w.current_time)-initial_time
        qh=cpu(human.get_joint_positions())[hids];dqh=cpu(human.get_joint_velocities())[hids]
        gravity=cpu(human._articulation_view.get_generalized_gravity_forces())[0,hids]
        hold,_=imp.effort(qh,np.zeros(4),gravity);he=np.zeros(4);he[hids]=hold;human.set_joint_efforts(dev(he))
        qr=cpu(robot.get_joint_positions());qdict={n:float(qr[mapping[n]]) for n in model.names}
        actual={s:model.fk(qdict,'tool_'+s) for s in 'LR'}
        tracking={s:float(np.linalg.norm(actual[s][:3,3]-commanded[s][:3,3])) for s in 'LR'}
        hp=cpu(human._articulation_view._physics_view.get_link_transforms())[0,forearm_index]
        hrot=Rotation.from_quat(hp[3:]).as_matrix();haxis=hrot[:,2];htip=hp[:3]-.385*haxis;wrist=hp[:3]-.26*haxis
        target_section=wrist+.26*cfg['target_forearm_fraction']*haxis
        grasp_error={};positions=None
        if cloth is not None and step%12==0:
            positions=cloth.get_vertice_positions()
            if not np.isfinite(positions).all():result['termination_reason']='nonfinite_cloth';break
            ring=positions[cuff];plane=opening(ring,haxis)
            axial=float((plane['center']-wrist)@haxis);hand_test=contains_section(ring,htip,.035,haxis)
            forearm_test=contains_section(ring,wrist,.035,haxis)
            geometry={'center_m':plane['center'],'normal':plane['normal'],'area_m2':plane['area_m2'],'spans_m':plane['spans_m'],
                      'planarity_rms_m':plane['planarity_rms_m'],'axial_from_wrist_m':axial,'target_from_wrist_m':.26*cfg['target_forearm_fraction'],
                      'hand_section':hand_test,'forearm_section':forearm_test}
            live=cpu(robot._articulation_view._physics_view.get_link_transforms())[0]
            for s in 'LR':
                pose=live[link_indices[s]];expected=local_anchors[s]@Rotation.from_quat(pose[3:]).as_matrix().T+pose[:3]
                grasp_error[s]=float(np.max(np.linalg.norm(positions[anchors[s]]-expected,axis=1)))
            if hand_test['inside'] and hand_test['plane_distance_m']<-.005:tip_entered=True
            if forearm_test['inside'] and forearm_test['plane_distance_m']<-.005:wrist_entered=True
        if step%12==0:
            if not np.isfinite(np.r_[qr,qh]).all() or (positions is not None and not np.isfinite(positions).all()):result['termination_reason']='nonfinite'
            elif any(not model.limits[n][0]-.002<=qdict[n]<=model.limits[n][1]+.002 for n in model.names):result['termination_reason']='robot_joint_limit'
            elif max(tracking.values())>cfg['tracking_abort_m']:result['termination_reason']='end_effector_tracking_error'
            elif max(grasp_error.values(),default=0.)>.02:result['termination_reason']='grasp_lost'
            elif np.any(qh>np.radians([100,70,100,145])-.01) or np.any(qh<np.radians([-100,-150,-100,0])+.01):result['termination_reason']='human_near_joint_limit'
            if state=='robot_hold':
                for s in 'LR':warm_max[s]=max(warm_max[s],tracking[s])
                if t>=.75 and max(tracking.values())<cfg['tracking_gate_m']:
                    cloth=Particle_Garment(w,usd_path='/workspace/PhyRC_Sim/Assets/Garment/Tops/Modelink/t_shirt_short.usd',pos=np.zeros(3),ori=np.array([0.,180.,180.]),scale=np.ones(3)*1.08,solver_position_iteration_count=64,friction=.3)
                    mesh=UsdGeom.Mesh(stage.GetPrimAtPath(cloth.garment_mesh_prim_path))
                    if not np.array_equal(np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3),tri):raise RuntimeError('Asset/refinement mismatch')
                    live=cpu(robot._articulation_view._physics_view.get_link_transforms())[0]
                    for s in 'LR':
                        pose=live[link_indices[s]];rot=Rotation.from_quat(pose[3:]).as_matrix();ids=anchors[s]
                        local_anchors[s]=(preset[ids]-pose[:3])@rot
                        prim=stage.DefinePrim('/World/Grasp'+s,'OmniPhysicsVtxXformAttachment')
                        prim.GetRelationship('omniphysics:src0').SetTargets([cloth.garment_mesh_prim_path]);prim.GetRelationship('omniphysics:src1').SetTargets([robot._articulation_view._physics_view.link_paths[0][link_indices[s]]])
                        prim.GetAttribute('omniphysics:vtxIndicesSrc0').Set(ids.tolist());prim.GetAttribute('omniphysics:localPositionsSrc1').Set([Gf.Vec3f(*map(float,v)) for v in local_anchors[s]])
                    # Finish USD attachment edits before the one-time nodal write.
                    # Parsing a newly added attachment can rebuild the cloth actor.
                    from isaacsim.core.simulation_manager import SimulationManager
                    SimulationManager._physics_sim_interface.flush_changes()
                    cloth.garment_mesh.initialize()
                    cloth.garment_mesh.set_world_positions(dev(preset[None,:,:]));cloth.garment_mesh.set_velocities(dev(np.zeros_like(preset)[None,:,:]))
                    readback=cloth.get_vertice_positions()
                    (out/'initialization_checks.json').write_text(json.dumps({'node_write_max_error_m':float(np.max(np.linalg.norm(readback-preset,axis=1))),'readback_cuff_center_m':readback[cuff].mean(0).tolist()},indent=2))
                    np.savez_compressed(out/'cloth_initial.npz',positions=preset,triangles=tri,cuff=cuff)
                    transition('hold',t,'robot-only hold passed; one-time opened preset and native grasp initialized')
                elif t>3:result['termination_reason']='robot_hold_not_stable'
            elif state=='hold':
                for s in 'LR':hold_max[s]=max(hold_max[s],tracking[s])
                if t-state_since>=1.:
                    if max(tracking.values())<cfg['tracking_gate_m'] and geometry['hand_section']['inside']:
                        transition('align',t,'measured stable grasp and usable aperture');last_motion=t
                    elif t-state_since>3:result['termination_reason']='initial_aperture_or_hold_not_stable'
            elif state in ('align','insert','advance','retreat'):
                if state=='align':dest=htip-cfg['initial_hand_gap_m']*haxis
                elif state=='insert':dest=wrist+.012*haxis
                elif state=='advance':dest=target_section+.004*haxis
                else:dest=retreat_goal
                if max(tracking.values())<cfg['tracking_gate_m']:
                    # Servo the measured rim, including its settled deformation
                    # offset, rather than only moving an ideal rigid preset.
                    delta=dest-geometry['center_m'];length=np.linalg.norm(delta)
                    command_center+=delta*min(1.,cfg['end_effector_speed_m_s']*.05/max(length,1e-9))
                if state=='align' and np.linalg.norm(geometry['center_m']-dest)<.008 and geometry['hand_section']['inside']:
                    transition('insert',t,'aligned cuff contains projected hand section')
                if state=='insert' and wrist_entered:
                    transition('advance',t,'hand and wrist passed the measured opening')
                if state=='advance':
                    criteria=tip_entered and wrist_entered and geometry['forearm_section']['inside'] and geometry['axial_from_wrist_m']>=.26*cfg['target_forearm_fraction']-.004 and max(grasp_error.values())<.01 and max(tracking.values())<.008
                    success_count=success_count+1 if criteria else 0
                    if success_count>=10:transition('complete',t,'containment, target forearm section and stable grasp held for 0.5 s');result['success']=True;result['termination_reason']='geometric_completion'
                if state=='retreat' and np.linalg.norm(command_center-dest)<.003:transition('align',t,'bounded retreat completed');last_motion=t
                if state in ('insert','advance'):
                    prog=geometry['axial_from_wrist_m']
                    if prog>last_progress+.002:last_progress=prog;last_motion=t
                    if t-last_motion>5:
                        if retries<2:retries+=1;retreat_goal=command_center-.015*haxis;transition('retreat',t,'no geometric progress for 5 s')
                        else:result['termination_reason']='persistent_no_progress'
                for s in 'LR':
                    commanded[s]=goals[s].copy();commanded[s][:3,3]+=command_center-center
                    sol,err=model.solve(s,commanded[s],desired,35,margin=.05)
                    if err['position_error_m']>.002 or err['rotation_error_rad']>.02:result['termination_reason']='ik_infeasible';break
                    desired.update(sol)
                cmd=model.bounded_step(qdict,desired,.05,speed=cfg['joint_speed_rad_s'])
                for n,v in cmd.items():rq[mapping[n]]=v
                robot._articulation_view.set_joint_position_targets(dev(rq[None,:]))
            episode.write(int(w.current_time_step_index),step//12,float(w.current_time),
                action={'state':state,'end_effector_targets':{s:v.tolist() for s,v in commanded.items()},'end_effector_delta_world_m':{s:commanded[s][:3,3]-previous_commanded[s][:3,3] for s in 'LR'},'rotation_delta_rad':[0.,0.,0.],'center_target_m':command_center,'gripper_command':'native attachment hold'},
                solver_state={'human_q_rad':qh,'human_dq_rad_s':dqh,'robot_joint_names':robot.dof_names,'robot_q':qr,'robot_dq':cpu(robot.get_joint_velocities()),
                'end_effector_actual':{s:v.tolist() for s,v in actual.items()},'end_effector_tracking_error_m':tracking,'human_hand_tip_m':htip,'human_wrist_m':wrist,'human_elbow_m':hp[:3],
                'cuff_keypoints_m':None if positions is None else positions[cuff],'geometry':geometry,'grasp_error_m':grasp_error,'hand_entered':tip_entered,'wrist_entered':wrist_entered},
                commanded_wrench={'human_explicit_Nm':hold,'gravity_compensation_Nm':gravity},privileged={'K':imp.stiffness,'D':imp.damping,'q_ref':imp.reference})
            previous_commanded={s:v.copy() for s,v in commanded.items()}
            if step%240==0:print('M11_STEP',step,state,'TRACK',tracking,'GEOMETRY',json.dumps(geometry,default=serial),'GRASP',grasp_error,flush=True)
        if sensors is not None and step%max(1,round(240/cfg['rgb_hz']))==0:
            imgs,_=sensors.capture();Image.fromarray(imgs['rgb'][0]).save(out/f'frame_{frame:05d}.jpg',quality=90)
            with (out/'frame_index.jsonl').open('a') as f:f.write(json.dumps({'frame':frame,'physics_step':int(w.current_time_step_index),'simulation_time_s':float(w.current_time),'capture_wall_time_ns':time.time_ns(),'state':state})+'\n')
            frame+=1
        if result['termination_reason']:break
        robot.set_joint_efforts(robot._articulation_view.get_generalized_gravity_forces()[0]);w.step(render=False)
    if result['termination_reason'] is None:result['termination_reason']='time_budget_exhausted'
    if cloth is not None:np.savez_compressed(out/'cloth_final.npz',positions=cloth.get_vertice_positions(),cuff=cuff)
    result.update(simulation_duration_s=float(w.current_time)-initial_time,robot_only_hold_max_error_m=warm_max,cloth_hold_max_error_m=hold_max,
                  hand_entered=tip_entered,wrist_entered=wrist_entered,final_geometry=geometry,rigid_contact_pairs=sorted(contacts),retries=retries,frames=frame)
    (out/'result.json').write_text(json.dumps(result,indent=2,default=serial,allow_nan=False))
    episode.close('success' if result['success'] else 'failure',result['termination_reason'])
    if sensors is not None:sensors.close()
    print('M11_RESULT',out,json.dumps(result,default=serial),flush=True);w.stop()
except BaseException as exc:
    traceback.print_exc();(out/'ERROR.txt').write_text(traceback.format_exc())
    result.update(success=False,termination_reason='exception:'+type(exc).__name__,error=str(exc))
    (out/'result.json').write_text(json.dumps(result,indent=2,default=serial))
    if episode is not None:episode.close('failure',result['termination_reason'])
    app.close(exit_code=1);os._exit(1)
finally:app.close()
