"""Load real supplied H1 plus assumed tools; verify bounded dual-arm pose IK in PhysX."""
import sys,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,'/workspace/PhyRC_Sim')
from isaacsim import SimulationApp
app=SimulationApp({'headless':'--gui' not in sys.argv})
try:
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.viewports import set_camera_view
    from pxr import UsdLux,UsdPhysics,Usd,PhysxSchema,UsdGeom
    from Policy.sensors import RGBDSensors
    from pxr import PhysicsSchemaTools
    import omni.physx
    from dynadress.robots.h1 import H1Kinematics
    w=World(physics_dt=1/240,rendering_dt=1/60);w.get_physics_context().enable_gpu_dynamics(True);w.get_physics_context().set_broadphase_type('GPU')
    asset=(ROOT/'assets/h1/generated/usd_path.txt').read_text().strip();add_reference_to_stage(asset,'/World/H1')
    if '--force-decomposition' in sys.argv:
        for _ in range(4):
            paths=[str(prim.GetPath()) for prim in Usd.PrimRange(w.stage.GetPrimAtPath('/World/H1'),Usd.TraverseInstanceProxies()) if prim.IsInstance()]
            for path in paths:
                prim=w.stage.GetPrimAtPath(path)
                if prim and not prim.IsInstanceProxy():prim.SetInstanceable(False)
        changed=[]
        for prim in Usd.PrimRange(w.stage.GetPrimAtPath('/World/H1')):
            if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexDecomposition');changed.append(str(prim.GetPath()))
        print('FORCED_DECOMPOSITION',len(changed),flush=True)
    for prim in Usd.PrimRange(w.stage.GetPrimAtPath('/World/H1')):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):print('ARTICULATION_ROOT',prim.GetPath(),flush=True)
    for prim in Usd.PrimRange(w.stage.GetPrimAtPath('/World/H1')):
        if prim.IsA(UsdPhysics.RevoluteJoint):PhysxSchema.PhysxJointAPI.Apply(prim).CreateMaxJointVelocityAttr(float(np.degrees(.5)))
        elif prim.IsA(UsdPhysics.PrismaticJoint):PhysxSchema.PhysxJointAPI.Apply(prim).CreateMaxJointVelocityAttr(.05)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            api=PhysxSchema.PhysxArticulationAPI.Apply(prim);api.CreateEnabledSelfCollisionsAttr('--no-self-collision' not in sys.argv);api.CreateSolverPositionIterationCountAttr(64);api.CreateSolverVelocityIterationCountAttr(16)
    contacts=set()
    for prim in Usd.PrimRange(w.stage.GetPrimAtPath('/World/H1')):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
    def on_contact(headers,data):
        for h in headers:
            pair=(str(PhysicsSchemaTools.intToSdfPath(h.actor0)),str(PhysicsSchemaTools.intToSdfPath(h.actor1)))
            contacts.add(pair)
    subscription=omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
    robot=w.scene.add(SingleArticulation('/World/H1',name='H1'));w.scene.add_default_ground_plane();UsdLux.DomeLight.Define(w.stage,'/World/Light').CreateIntensityAttr(1000.)
    set_camera_view(eye=[2,-3,2],target=[0,0,1]);w.reset();robot.initialize()
    robot._articulation_view.set_gains(kps=np.full((1,robot.num_dof),200.),kds=np.full((1,robot.num_dof),15.))
    print('GAINS',robot._articulation_view.get_gains(),flush=True)
    print('DOF',robot.dof_names,flush=True)
    model=H1Kinematics(ROOT/'assets/h1/generated/h1_task.urdf');names=robot.dof_names
    mapping={}
    for n in model.names:
        side='L' if '08L' in n else 'R';index=n.rsplit('_',1)[-1]
        hits=[v for v in names if '08'+side in v and '_joint_'+index+'_' in v]
        if len(hits)!=1:raise ValueError(f'Cannot map joint {n}: {hits}')
        mapping[n]=hits[0]
    curr={n:(.5 if n.endswith('_4') else 0.) for n in model.names};q=robot.get_joint_positions()
    for n in model.names:q[robot.get_dof_index(mapping.get(n,n))]=curr[n]
    robot.set_joint_positions(q);robot.set_joint_velocities(np.zeros_like(q));robot._articulation_view.set_joint_position_targets(q[None,:])
    out=ROOT/'artifacts/m1/h1_motion'/time.strftime('%Y%m%dT%H%M%S');out.mkdir(parents=True)
    sensors=RGBDSensors(w.stage,w,[],{'default_dimensions':{'width':800,'height':600},'cameras':[{'id':'h1','mount':'world','eye_world_m':[2,-3,2],'target_world_m':[0,0,1],'horizontal_fov_deg':65,'clip_m':[.05,20]}]})
    initial=robot.get_joint_positions().copy();goals={};ik={}
    for side in ['L','R']:
        target=model.fk(curr,'tool_'+side);target[0,3]+=.025;target[1,3]+=(-.025 if side=='L' else .025);target[2,3]+=.01
        sol,info=model.solve(side,target,curr);curr.update(sol);goals[side]=target.tolist();ik[side]=info
    desired=dict(curr);curr={n:(.5 if n.endswith('_4') else 0.) for n in model.names};rows=[];velocities=[]
    from PIL import Image
    for step in range(1440):
        curr=model.bounded_step(curr,desired,1/240)
        cmd=robot.get_joint_positions().copy()
        for n in model.names:cmd[robot.get_dof_index(mapping.get(n,n))]=curr[n]
        for n in names:
            if '_jaw_' in n:cmd[robot.get_dof_index(mapping.get(n,n))]=.015
        robot._articulation_view.set_joint_position_targets(cmd[None,:])
        robot.set_joint_efforts(robot._articulation_view.get_generalized_gravity_forces()[0])
        w.step(render=False)
        rows.append(np.r_[step,float(w.current_time),robot.get_joint_positions(),cmd]);velocities.append(robot.get_joint_velocities().copy())
        if step%12==0:
            imgs,_=sensors.capture();Image.fromarray(imgs['rgb'][0]).save(out/f'frame_{step//12:04d}.png')
        if step%240==0:print('STEP',step,flush=True)
    actual={n:float(robot.get_joint_positions()[robot.get_dof_index(mapping.get(n,n))]) for n in model.names}
    error={s:float(np.linalg.norm(model.fk(actual,'tool_'+s)[:3,3]-np.array(goals[s])[:3,3])) for s in ['L','R']}
    link_names=robot._articulation_view.body_names;link_tf=robot._articulation_view._physics_view.get_link_transforms()[0]
    measured={}
    for side in ['L','R']:
        idx=next(i for i,n in enumerate(link_names) if '08'+side in n and '_link7_' in n)
        from scipy.spatial.transform import Rotation as Rotation
        live=np.eye(4);live[:3,3]=link_tf[idx,:3];live[:3,:3]=Rotation.from_quat(link_tf[idx,3:]).as_matrix()
        tool=live@np.linalg.inv(model.fk(actual,f'AR5-5_08{side}-W4C1C9-ZY2_link7'))@model.fk(actual,'tool_'+side)
        measured[side]={'matrix_world':tool.tolist(),'fk_world_error_m':float(np.linalg.norm(model.fk(actual,'tool_'+side)[:3,3]-tool[:3,3])),'goal_error_m':float(np.linalg.norm(tool[:3,3]-np.array(goals[side])[:3,3]))}
    result={'rigid_contact_pairs':sorted(contacts),'live_end_effectors':measured,'scope':'independent H1 dual-arm IK motion; not dressing','ik':ik,'tracking_error_fk_m':error,'joint_limit_violations':[n for n in model.names if not model.limits[n][0]-1e-4<=actual[n]<=model.limits[n][1]+1e-4],'all_finite':bool(np.isfinite(rows).all()),'gripper':'user-authorized simulation-only parallel jaw tool','self_collision':'--no-self-collision' not in sys.argv,'joint_names':names}
    result['maximum_joint_velocity_rad_s']={n:float(np.max(np.abs(np.array(velocities)[:,robot.get_dof_index(mapping[n])]))) for n in model.names}
    result['velocity_limit_violations']=[n for n,v in result['maximum_joint_velocity_rad_s'].items() if v>model.velocity[n]+.01]
    result['passed']=result['self_collision'] and not result['joint_limit_violations'] and not result['velocity_limit_violations'] and result['all_finite'] and max(v['goal_error_m'] for v in measured.values())<.005
    np.savez_compressed(out/'motion.npz',rows=np.array(rows),joint_velocity=np.array(velocities));(out/'result.json').write_text(json.dumps(result,indent=2));print('M1-H1-MOTION-RESULT',out,json.dumps(result),flush=True);sensors.close();w.stop()
except BaseException:
    import traceback,os
    traceback.print_exc()
    app.close(exit_code=1)
    os._exit(1)
finally:app.close()
