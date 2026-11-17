"""Actual PhysX passive response test; pulse is commanded joint torque, not cloth contact."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
p=argparse.ArgumentParser();p.add_argument('--stiffness',type=float,default=None);p.add_argument('--config',default=str(ROOT/'dynadress/configs/m1_passive.json'));p.add_argument('--contact',action='store_true');p.add_argument('--no-pulse',action='store_true');p.add_argument('--disable-human-collision',action='store_true');p.add_argument('--gui',action='store_true');args,_=p.parse_known_args()
from isaacsim import SimulationApp
app=SimulationApp({'headless':not args.gui})
try:
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from pxr import UsdLux,Gf,Usd,UsdPhysics
    from dynadress.human.passive_arm import build,Impedance,JOINT_NAMES
    from isaacsim.core.utils.viewports import set_camera_view
    w=World(physics_dt=1/240,rendering_dt=1/60,backend='torch' if args.contact else 'numpy',device='cuda:0' if args.contact else 'cpu');pc=w.get_physics_context();pc.enable_gpu_dynamics(True);pc.set_broadphase_type('GPU');pc.set_solver_type('TGS')
    config=json.loads(Path(args.config).read_text());params=config['impedance'].copy()
    if args.stiffness is not None:params['stiffness']=[args.stiffness]*4
    imp=Impedance(**params);args.stiffness=imp.stiffness[0]
    build(w.stage,**config['geometry']);UsdLux.DomeLight.Define(w.stage,'/World/Light').CreateIntensityAttr(800.)
    arm=w.scene.add(SingleArticulation('/World/PassiveArm',name='human'))
    cloth=None
    if args.contact:
        sys.path.insert(0,'/workspace/PhyRC_Sim')
        from Env_Config.Garment.Particle_Garment import Particle_Garment
        cloth=Particle_Garment(w,usd_path='/workspace/PhyRC_Sim/Assets/Garment/Tops/Modelink/t_shirt_short.usd',pos=np.array([.38,0,1.5]),ori=np.array([0.,180.,180.]),scale=np.ones(3)*1.08,solver_position_iteration_count=64,friction=.3)
    if args.disable_human_collision:
        for prim in Usd.PrimRange(w.stage.GetPrimAtPath('/World/PassiveArm')):
            if prim.HasAPI(UsdPhysics.CollisionAPI):UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
    from isaacsim.core.simulation_manager import SimulationManager
    if args.contact:SimulationManager.set_physics_sim_device('cuda:0')
    w.reset();arm.initialize()
    import torch
    def dev(value):return torch.as_tensor(value,dtype=torch.float32,device=arm._articulation_view._device) if arm._articulation_view._backend=='torch' else np.asarray(value,dtype=np.float32)
    def cpu(value):return value.detach().cpu().numpy().copy() if isinstance(value,torch.Tensor) else np.asarray(value).copy()
    print('DOF',arm.dof_names,'BACKEND',arm._articulation_view._backend,flush=True)
    ids=np.array([arm.get_dof_index(n) for n in JOINT_NAMES])
    # Set state once at reset, never during the response.
    q=np.zeros(4);q[ids]=config['initial_q_rad'];arm.set_joint_positions(dev(q));arm.set_joint_velocities(dev(config['initial_dq_rad_s']));arm._articulation_view.set_gains(kps=dev(np.zeros((1,4))),kds=dev(np.array([imp.damping])[...,np.argsort(ids)]))
    actual_kp,actual_kd=arm._articulation_view.get_gains()
    if np.max(np.abs(cpu(actual_kp)))>1e-8 or not np.allclose(cpu(actual_kd)[0,ids],imp.damping):raise RuntimeError('Human drive gains differ from requested passive damping')
    out=ROOT/'artifacts/m1'/('contact' if args.contact else ('free' if args.no_pulse else 'pulse'))/f'k{args.stiffness:g}'/time.strftime('%Y%m%dT%H%M%S');out.mkdir(parents=True)
    from dynadress.recording.episode import Episode
    episode=Episode(out,{'pulse_enabled':not args.no_pulse and not args.contact,'config':config,'scope':'isolated passive arm response; no H1 or dressing', 'seed':0,'physics_hz':240,'control_hz':240,'stiffness':list(imp.stiffness),'damping':list(imp.damping),'q_ref':list(imp.reference),'human_collision_enabled':not args.disable_human_collision,'cloth_present':args.contact,'cloth_state_sampling_hz':10,'isaac_version':(Path('/root/isaacsim/VERSION').read_text().strip()),'source_commit':__import__('os').environ.get('PHYRC_SOURCE_COMMIT'),'damping_source':'zero stiffness implicit velocity drive; explicit efforts exclude damping'})
    sys.path.insert(0,'/workspace/PhyRC_Sim')
    from Policy.sensors import RGBDSensors
    from PIL import Image
    sensors=RGBDSensors(w.stage,w,[],{'default_dimensions':{'width':640,'height':480},'cameras':[{'id':'human','mount':'world','eye_world_m':[1.6,-1.6,1.8],'target_world_m':[.25,0,1.1],'horizontal_fov_deg':60,'clip_m':[.05,20]}]})
    initial_sim_time=float(w.current_time)
    rows=[];set_camera_view(eye=[1.7,-2,1.8],target=[.2,0,.9])
    for step in range(1440):
        if abs(float(w.current_time)-initial_sim_time-step/240)>1e-6:raise RuntimeError('Physics clock drift from explicit 240 Hz stepping')
        q=cpu(arm.get_joint_positions())[ids];dq=cpu(arm.get_joint_velocities())[ids];g=cpu(arm._articulation_view.get_generalized_gravity_forces())[0,ids]
        hold,raw=imp.effort(q,np.zeros(4),g);pulse=np.zeros(4)
        if not args.contact and not args.no_pulse and 240<=step<360:pulse[1]=1.
        effort=np.zeros(4);effort[ids]=hold+pulse;arm.set_joint_efforts(dev(effort))
        rows.append([step,float(w.current_time),*q,*dq,*hold,*g,*pulse])
        if step%24==0:
            cloth_positions=None
            if cloth is not None:
                cloth_positions=cloth.get_vertice_positions()
                if step==0:print('CLOTH_BOUNDS',cloth_positions.min(axis=0),cloth_positions.max(axis=0),flush=True)
                if not np.isfinite(cloth_positions).all():raise FloatingPointError('Nonfinite cloth')
                np.savez_compressed(out/f'cloth_{step:06d}.npz',positions=cloth_positions)
            episode.write(int(w.current_time_step_index),step,float(w.current_time),action={'human_external_test_torque_Nm':pulse,'robot_target':None,'gripper':None},solver_state={'human_q_rad':q,'human_dq_rad_s':dq,'robot':None,'cloth_centroid_m':None if cloth_positions is None else cloth_positions.mean(axis=0)},commanded_wrench={'human_explicit_effort_Nm':hold+pulse,'gravity_compensation_Nm':g,'implicit_damping_coefficient_Nms_rad':imp.damping},privileged={'q_ref':imp.reference,'stiffness_Nm_rad':imp.stiffness,'damping_Nms_rad':imp.damping})
            imgs,_=sensors.capture();Image.fromarray(imgs['rgb'][0]).save(out/f'frame_{step//24:04d}.png')
        w.step(render=False)
        if step%240==0:print('STEP',step,'q',q.tolist(),flush=True)
    data=np.array(rows);np.savez_compressed(out/'response.npz',rows=data)
    result={'clock_contract':'explicit one physics step per effort update; rendering delta_time=0','simulation_duration_s':float(w.current_time)-initial_sim_time,'test':'cloth_drop_contact' if args.contact else 'commanded_torque_pulse','stiffness':list(imp.stiffness),'damping':list(imp.damping),'q_ref':list(imp.reference),'state_writes':'reset only','drive_stiffness':0,'drive_damping':list(imp.damping),'actual_drive_stiffness':cpu(actual_kp).tolist(),'actual_drive_damping':cpu(actual_kd).tolist(),'damping_implementation':'implicit velocity drive, stiffness zero, target velocity zero; explicit effort excludes D*dq to avoid double damping', 'gravity':'PhysX generalized gravity compensation; effort added each physics tick','max_deviation_rad':np.max(np.abs(data[:,2:6]-imp.reference),axis=0).tolist(),'final_error_rad':(data[-1,2:6]-imp.reference).tolist(),'finite':bool(np.isfinite(data).all()),'source_gravity_contact':'solver generalized gravity is not contact force','passed':False}
    result['passed']=bool(result['finite'] and np.max(np.abs(data[:240,2:6]-imp.reference))<.01 and (args.contact or np.max(np.abs(data[-240:,2:6]-imp.reference))<.01))
    if args.contact:result['passed']=None  # Coupling verdict requires the matched collision-disabled run.
    (out/'result.json').write_text(json.dumps(result,indent=2));print('M1-HUMAN-RESULT',out,json.dumps(result),flush=True)
    episode.close('diagnostic_complete','6 simulated seconds reached; not dressing success');sensors.close();w.stop()
except BaseException:
    import traceback,os
    traceback.print_exc()
    app.close(exit_code=1)
    os._exit(1)
finally:app.close()
