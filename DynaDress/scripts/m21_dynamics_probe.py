"""Extract PhysX human inertia and cloth-free pulse trajectories (real native run)."""
import argparse,json,sys,time,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'artifacts/m21/dynamics'));p.add_argument('--seconds',type=float,default=6.);a,_=p.parse_known_args()
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'limit_cpu_threads':2})
try:
 from isaacsim.core.api import World
 from isaacsim.core.prims import SingleArticulation
 from isaacsim.core.simulation_manager import SimulationManager
 from dynadress.human.passive_arm import build,Impedance,JOINT_NAMES
 import torch
 torch.set_num_threads(1)
 w=World(physics_dt=1/240,rendering_dt=1/60,backend='torch',device='cuda:0');pc=w.get_physics_context();pc.enable_gpu_dynamics(True);pc.set_broadphase_type('GPU');pc.set_solver_type('TGS')
 build(w.stage);arm=w.scene.add(SingleArticulation('/World/PassiveArm',name='human'));SimulationManager.set_physics_sim_device('cuda:0');w.reset();arm.initialize()
 ids=np.array([arm.get_dof_index(n) for n in JOINT_NAMES]);order=np.argsort(ids)
 def cpu(v):return v.detach().cpu().numpy().copy() if isinstance(v,torch.Tensor) else np.array(v)
 def dev(v):return torch.as_tensor(v,dtype=torch.float32,device='cuda:0')
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 matrices={};results=[]
 for tag,ref,K,D in [('m2_k4_d05',[0.,1.,0.,.35],4.,.5),('m2_k20_d2',[0.,1.,0.,.35],20.,2.),('m1_k4_d1',[0.,-1.3,0.,.3],4.,1.)]:
  if results:w.reset();arm.initialize()
  imp=Impedance(stiffness=(K,)*4,damping=(D,)*4,reference=tuple(ref))
  arm.set_joint_positions(dev(np.array(ref)[order]));arm.set_joint_velocities(dev(np.zeros(4)));arm._articulation_view.set_gains(kps=dev(np.zeros((1,4))),kds=dev(np.full((1,4),D)))
  # Update articulation cache at the reset pose without moving it: first zero-motion tick.
  g=cpu(arm._articulation_view.get_generalized_gravity_forces())[0,ids];hold,_=imp.effort(np.array(ref),np.zeros(4),g);arm.set_joint_efforts(dev(hold[order]));w.step(render=False)
  mass=cpu(arm._articulation_view.get_mass_matrices())[0][np.ix_(ids,ids)];g=cpu(arm._articulation_view.get_generalized_gravity_forces())[0,ids]
  matrices[tag]={'matrix_kg_m2':mass.tolist(),'gravity_compensation_Nm':g.tolist(),'q_ref_rad':ref,'eigenvalues':np.linalg.eigvalsh(mass).tolist(),'dof_names':arm.dof_names,'joint_indices':ids.tolist()}
  print('M21-MASS',tag,json.dumps(matrices[tag]),flush=True)
  rows=[];start=float(w.current_time)
  for step in range(round(a.seconds*240)):
   q=cpu(arm.get_joint_positions())[ids];dq=cpu(arm.get_joint_velocities())[ids];g=cpu(arm._articulation_view.get_generalized_gravity_forces())[0,ids];hold,raw=imp.effort(q,np.zeros(4),g)
   pulse=np.zeros(4)
   if 240<=step<360:pulse[1]=1.
   if 600<=step<624:pulse[0]=.15
   arm.set_joint_efforts(dev((hold+pulse)[order]));rows.append([step,float(w.current_time)-start,*q,*dq,*hold,*g,*pulse]);w.step(render=False)
  rows=np.array(rows);np.savez_compressed(out/f'{tag}.npz',rows=rows)
  result={'tag':tag,'q_ref':ref,'K':[K]*4,'D':[D]*4,'max_deviation_rad':np.max(np.abs(rows[:,2:6]-ref),axis=0).tolist(),'final_error_rad':(rows[-1,2:6]-ref).tolist(),'finite':bool(np.isfinite(rows).all()),'physics_hz':240,'no_cloth':True,'state_writes':'reset only','pulse_intervals':[{'steps':[240,360],'joint':'shoulder_y','Nm':1.},{'steps':[600,624],'joint':'shoulder_x','Nm':.15}],'torque_limit_Nm':list(imp.torque_limit)};results.append(result)
  print('M21-PULSE',json.dumps(result),flush=True)
 (out/'physx_reference.json').write_text(json.dumps({'isaac_version':Path('/root/isaacsim/VERSION').read_text().strip(),'joint_order':JOINT_NAMES,'geometry':'dynadress.human.passive_arm.build defaults; unchanged from M2','matrices':matrices,'results':results,'columns':['physics_step','time_s']+[f'{prefix}_{joint}' for prefix in ['q_rad','dq_rad_s','hold_Nm','gravity_Nm','test_pulse_Nm'] for joint in JOINT_NAMES]},indent=2))
 print('M21-DYNAMICS-PROBE-COMPLETE',out,flush=True);w.stop()
except BaseException:
 import traceback;traceback.print_exc();app.close(exit_code=1);os._exit(1)
finally:app.close()
