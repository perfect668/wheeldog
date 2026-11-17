"""Matched commanded wrist trajectory without cloth: load-channel validation.

Uses the same H1, articulation gains, wrist collision exclusions, gravity feed
forward and physics clock. No human holding torque enters this experiment.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),'/workspace/PhyRC_Sim']
from dynadress.robots.h1 import H1Kinematics
from dynadress.interaction.wrist_load import WristLoad
from dynadress.recording.episode import serial
p=argparse.ArgumentParser();p.add_argument('episode',type=Path);p.add_argument('--output',type=Path,default=ROOT/'artifacts/m2/load_validation');a,_=p.parse_known_args()
cfg=json.loads((a.episode/'config.json').read_text());source=[json.loads(r) for r in (a.episode/'samples.jsonl').read_text().splitlines()]
a.output.mkdir(parents=True,exist_ok=False)
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'limit_cpu_threads':2})
try:
    import torch
    torch.set_num_threads(1)
    from pxr import Usd,UsdPhysics,PhysxSchema,UsdGeom
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage
    w=World(physics_dt=1/240,rendering_dt=1/60,backend='torch',device='cuda:0')
    pc=w.get_physics_context();pc.enable_gpu_dynamics(True);pc.set_broadphase_type('GPU');pc.set_solver_type('TGS');stage=w.stage
    add_reference_to_stage((ROOT/'assets/h1/generated/usd_path.txt').read_text().strip(),'/World/H1')
    for _ in range(4):
        for path in [str(v.GetPath()) for v in Usd.PrimRange(stage.GetPrimAtPath('/World/H1'),Usd.TraverseInstanceProxies()) if v.IsInstance()]:
            v=stage.GetPrimAtPath(path)
            if v and not v.IsInstanceProxy():v.SetInstanceable(False)
    for v in Usd.PrimRange(stage.GetPrimAtPath('/World/H1')):
        if v.HasAPI(UsdPhysics.CollisionAPI) and 'tool_' in str(v.GetPath()):
            col=PhysxSchema.PhysxCollisionAPI.Apply(v);col.CreateContactOffsetAttr(.002);col.CreateRestOffsetAttr(.0002)
        if v.IsA(UsdGeom.Mesh) and v.HasAPI(UsdPhysics.CollisionAPI):UsdPhysics.MeshCollisionAPI.Apply(v).CreateApproximationAttr('convexDecomposition')
        if v.HasAPI(UsdPhysics.ArticulationRootAPI):
            api=PhysxSchema.PhysxArticulationAPI.Apply(v);api.CreateEnabledSelfCollisionsAttr(True);api.CreateSolverPositionIterationCountAttr(64);api.CreateSolverVelocityIterationCountAttr(16)
        if v.IsA(UsdPhysics.RevoluteJoint):PhysxSchema.PhysxJointAPI.Apply(v).CreateMaxJointVelocityAttr(float(np.degrees(cfg['joint_speed_rad_s'])))
        elif v.IsA(UsdPhysics.PrismaticJoint):PhysxSchema.PhysxJointAPI.Apply(v).CreateMaxJointVelocityAttr(.05)
    for side in 'LR':
        bodies={i:next(v for v in Usd.PrimRange(stage.GetPrimAtPath('/World/H1')) if v.HasAPI(UsdPhysics.RigidBodyAPI) and '08'+side in str(v.GetPath()) and '_link'+str(i)+'_' in str(v.GetPath())) for i in (5,7)}
        UsdPhysics.FilteredPairsAPI.Apply(bodies[5]).CreateFilteredPairsRel().SetTargets([bodies[7].GetPath()])
    robot=w.scene.add(SingleArticulation('/World/H1',name='h1'));w.reset();robot.initialize()
    cpu=lambda x:x.detach().cpu().numpy().copy();dev=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda:0')
    model=H1Kinematics(ROOT/'assets/h1/generated/h1_task.urdf')
    mapping={n:robot.get_dof_index(next(v for v in robot.dof_names if '08'+('L' if '08L' in n else 'R') in v and '_joint_'+n.rsplit('_',1)[-1]+'_' in v)) for n in model.names}
    first=source[0]['solver_state']['value'];rq=np.array([first['robot_q'][first['robot_joint_names'].index(n)] for n in robot.dof_names])
    robot.set_joint_positions(dev(rq));robot.set_joint_velocities(dev(np.zeros(robot.num_dof)))
    view=robot._articulation_view;view.set_gains(kps=dev(np.full((1,robot.num_dof),cfg['robot_gains']['stiffness'])),kds=dev(np.full((1,robot.num_dof),cfg['robot_gains']['damping'])))
    view.set_joint_position_targets(dev(rq[None]));desired={n:float(rq[mapping[n]]) for n in model.names}
    links={s:next(i for i,n in enumerate(view.body_names) if '08'+s in n and '_link7_' in n) for s in 'LR'}
    sensor=WristLoad(robot,stage,links,cpu);rows=[]
    for step in range((len(source)-1)*12+1):
        if step%12==0:
            row=source[step//12];qr=cpu(robot.get_joint_positions());qd={n:float(qr[mapping[n]]) for n in model.names}
            targets={s:np.asarray(row['action']['end_effector_targets'][s]) for s in 'LR'}
            for side in 'LR':
                sol,err=model.solve(side,targets[side],desired,35,margin=.05);desired.update(sol)
                if err['position_error_m']>.002:raise RuntimeError('Replay IK failed')
            cmd=model.bounded_step(qd,desired,.05,speed=cfg['joint_speed_rad_s'])
            for n,v in cmd.items():rq[mapping[n]]=v
            view.set_joint_position_targets(dev(rq[None]))
            raw,estimate=sensor.read()
            rows.append(dict(simulation_time_s=float(w.current_time),source_simulation_time_s=row['simulation_time_s'],physics_step=int(w.current_time_step_index),
                raw_wrist=raw,estimated_load=estimate,robot_q=qr,robot_dq=cpu(robot.get_joint_velocities()),
                commanded_end_effectors=targets,tracking_m={s:float(np.linalg.norm(model.fk(qd,'tool_'+s)[:3,3]-targets[s][:3,3])) for s in 'LR'}))
        robot.set_joint_efforts(view.get_generalized_gravity_forces()[0]);w.step(render=False)
    with (a.output/'samples.jsonl').open('w') as f:
        for row in rows:f.write(json.dumps(row,default=serial,allow_nan=False)+'\n')
    # Exclude initial zero-time sensor buffer; retain it in the raw file.
    load=np.array([[r['estimated_load']['external_world_N_Nm'][s] for s in 'LR'] for r in rows[4:]])
    result=dict(source_episode=str(a.episode),physics_hz=240,control_hz=20,samples=len(rows),
        replay='identical recorded end-effector command sequence; scene contains H1 only, no cloth or human',load_channel=sensor.description,
        force_rmse_N=float(np.sqrt(np.mean(load[:,:,:3]**2))),torque_rmse_Nm=float(np.sqrt(np.mean(load[:,:,3:]**2))),
        maximum_force_norm_N=float(np.max(np.linalg.norm(load[:,:,:3],axis=-1))),maximum_torque_norm_Nm=float(np.max(np.linalg.norm(load[:,:,3:],axis=-1))),
        maximum_target_tracking_m=max(max(r['tracking_m'].values()) for r in rows),
        scope='validates own-weight and inertial compensation; not a calibration of garment-only contact or a physical sensor')
    (a.output/'result.json').write_text(json.dumps(result,indent=2));print('M2_WRIST_REPLAY',json.dumps(result),flush=True);w.stop()
except BaseException:
    import traceback,os
    (a.output/'ERROR.txt').write_text(traceback.format_exc());traceback.print_exc();app.close(exit_code=1);os._exit(1)
finally:app.close()
