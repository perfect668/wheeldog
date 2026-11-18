"""Offline comparison of known impedance predictor against actual cloth-free PhysX."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dynadress.models.impedance import FixedImpedanceDynamics,gravity_compensation
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=ROOT/'artifacts/m21/dynamics');a=p.parse_args();torch.set_num_threads(1)
meta=json.loads((a.input/'physx_reference.json').read_text());report={'source':'real PhysX cloth-free torque-pulse runs; not garment contact labels','cases':{}}
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(3,2,figsize=(12,10))
for n,item in enumerate(meta['results']):
 tag=item['tag'];rows=np.load(a.input/f'{tag}.npz')['rows'];dt=1/20;s=rows[::12];q=torch.tensor(s[:,2:6],dtype=torch.float64);v=torch.tensor(s[:,6:10],dtype=torch.float64);tau=torch.tensor(s[:,18:22],dtype=torch.float64);ref=torch.tensor(item['q_ref'],dtype=torch.float64);K=torch.tensor(item['K'],dtype=torch.float64);D=torch.tensor(item['D'],dtype=torch.float64)
 # Shared M2 reference matrix, even for the older M1 reference pose.
 m=FixedImpedanceDynamics().double();g=gravity_compensation(torch.tensor(rows[:,2:6],dtype=torch.float64)).numpy();g_error=float(np.max(abs(g-rows[:,14:18])))
 horizons={}
 with torch.no_grad():
  for h in [5,10,20]:
   count=len(q)-h;pred=q[:count].clone();vel=v[:count].clone()
   for step in range(h):pred,vel=m(pred,vel,tau[step:step+count],K,D,ref,dt)
   truth=q[h:];hold=q[:count];linear=q[:count]+h*dt*v[:count]
   horizons[str(h*dt)]={'structured_q_rmse_rad':float(torch.mean((pred-truth)**2).sqrt()),'persistence_q_rmse_rad':float(torch.mean((hold-truth)**2).sqrt()),'velocity_q_rmse_rad':float(torch.mean((linear-truth)**2).sqrt())}
  pred=q[0];vel=v[0];preds=[pred.numpy()];vels=[vel.numpy()]
  for step in range(len(q)-1):
   pred,vel=m(pred,vel,tau[step],K,D,ref,dt);preds.append(pred.numpy());vels.append(vel.numpy())
 pred=np.array(preds);vel=np.array(vels)
 bound=np.radians(70);limit_steps=int(np.count_nonzero(abs(rows[:,3]-bound)<1e-5))
 report['cases'][tag]={'horizons':horizons,'open_6s_q_rmse_rad':float(np.mean((pred-q.numpy())**2)**.5),'open_6s_max_abs_q_error_rad':float(np.max(abs(pred-q.numpy()))),'analytical_gravity_max_abs_error_Nm':g_error,'shoulder_y_upper_limit_physics_steps':limit_steps,'explicit_hold_clip_steps':int(np.count_nonzero(np.any(abs(rows[:,10:14])>=np.array([30,30,20,20])-.001,axis=1)))}
 for j in [0,1,3]:
  axes[n,0].plot(s[:,1],q[:,j],label=f'{j} PhysX');axes[n,0].plot(s[:,1],pred[:,j],'--',label=f'{j} fixed M')
  axes[n,1].plot(s[:,1],v[:,j],label=f'{j} PhysX');axes[n,1].plot(s[:,1],vel[:,j],'--',label=f'{j} fixed M')
 axes[n,0].set_title(tag+' angles');axes[n,1].set_title(tag+' velocities');axes[n,0].set_ylabel('rad');axes[n,1].set_ylabel('rad/s')
 for ax in axes[n]:ax.set_xlabel('Time (s)');ax.grid(alpha=.25);ax.legend(ncol=2,fontsize=7)
fig.tight_layout();fig.savefig(a.input/'impedance_validation.png',dpi=150);plt.close(fig)
(a.input/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
