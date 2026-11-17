"""Audit-only cameras to show the whole original scene and FEM garment.

These additional camera poses are for deployment verification, not competition
policy observations or training inputs. Scene physics and upstream config remain unchanged.
"""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, '/workspace/PhyRC_Sim')
from policy_cli import install_failure_handler
install_failure_handler()
from Policy.environment import DressingEnv
env = DressingEnv(profile='measured_state', max_episode_steps=20)
import numpy as np
from PIL import Image, ImageDraw
from Policy.sensors import RGBDSensors
from Policy.state import snapshot
out = Path('/output/native_scene_audit') / time.strftime('%Y%m%dT%H%M%S')
out.mkdir(parents=True, exist_ok=False)
print('AUDIT_OUTPUT=' + str(out), flush=True)
obs, info = env.reset(seed=42)
meta, arrays = snapshot(env.M, env.cloths, env.rigs)
center = np.asarray(meta['garments'][0]['centroid_world'])
config = {'default_dimensions': {'width': 1024, 'height': 768}, 'cameras': [
    {'id':'audit_scene', 'mount':'world', 'eye_world_m':[4,-7,5],
     'target_world_m':[0,-.4,.6], 'horizontal_fov_deg':80, 'clip_m':[.05,30]},
    {'id':'audit_garment', 'mount':'world', 'eye_world_m':(center+[0,-1,1.5]).tolist(),
     'target_world_m':center.tolist(), 'horizontal_fov_deg':60, 'clip_m':[.01,10]},
]}
sensors = RGBDSensors(env.stage, env.world, env.rigs, config)
report = {'passed':False, 'purpose':'additional audit cameras, not policy sensors',
          'camera_config':config, 'source_commit':info['source_commit'],
          'robot_count':len(env.rigs), 'garment_count':len(env.cloths), 'frames':[]}
try:
    for i in range(21):
        if i:
            action=np.zeros((2,9),np.float32)
            action[:,3]=.1
            obs,_,_,_,info=env.step(action)
        before=float(env.world.current_time)
        images,cameras=sensors.capture()
        canvas=Image.new('RGB',(2048,792),(22,25,30))
        draw=ImageDraw.Draw(canvas)
        for j in range(2):
            canvas.paste(Image.fromarray(images['rgb'][j]),(j*1024,24))
            draw.text((j*1024+8,6),f"{config['cameras'][j]['id']} | simulation {before:.3f} s | audit camera",fill='white')
        canvas.save(out/f'frame_{i:04d}.png')
        if i in (0,20):
            for j in range(2):
                Image.fromarray(images['rgb'][j]).save(out/f"{config['cameras'][j]['id']}_{i:04d}.png")
        report['frames'].append({'step':i,'simulation_time_s':before,'cameras':cameras})
        if i%5==0: print(f'AUDIT_FRAME {i}/20',flush=True)
    report['passed']=True
    report['final_state_finite']=snapshot(env.M,env.cloths,env.rigs)[0]['all_arrays_finite']
    (out/'capture.json').write_text(json.dumps(report,indent=2))
    print('NATIVE-SCENE-AUDIT-PASS '+str(out),flush=True)
except BaseException as error:
    report['error']=repr(error)
    (out/'capture.json').write_text(json.dumps(report,indent=2))
    raise
finally:
    sensors.close()
    env.close()
