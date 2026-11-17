"""Export the original refined FEM geometry and available native attachment schema."""
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,'/workspace/PhyRC_Sim')
from isaacsim import SimulationApp
app=SimulationApp({'headless':True})
try:
 from isaacsim.core.api import World
 from Env_Config.Garment.Particle_Garment import Particle_Garment
 from pxr import UsdGeom,Usd
 w=World(backend='torch',device='cuda:0');w.get_physics_context().enable_gpu_dynamics(True)
 c=Particle_Garment(w,usd_path='/workspace/PhyRC_Sim/Assets/Garment/Tops/Modelink/t_shirt_short.usd',pos=np.zeros(3),ori=np.array([0,180,180]),scale=np.ones(3)*1.08,solver_position_iteration_count=64,friction=.3)
 mesh=UsdGeom.Mesh(w.stage.GetPrimAtPath(c.garment_mesh_prim_path));p=np.array(mesh.GetPointsAttr().Get());tf=np.array(UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(Usd.TimeCode.Default()));p=(np.c_[p,np.ones(len(p))]@tf)[:,:3]
 np.savez_compressed(ROOT/'assets/h1/generated/m11_garment_geometry.npz',points=p,triangles=np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3))
 prim=w.stage.DefinePrim('/World/Probe','OmniPhysicsVtxXformAttachment');props={n:str(prim.GetAttribute(n).Get()) for n in prim.GetPropertyNames() if prim.GetAttribute(n)}
 print('ATTACHMENT_SCHEMA',json.dumps(props),flush=True)
except BaseException:
 import traceback;traceback.print_exc();app.close(exit_code=1)
finally:app.close()
