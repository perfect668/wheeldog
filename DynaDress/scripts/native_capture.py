"""Capture real, newly simulated PhyRC frames; no trajectory replay or policy training."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, '/workspace/PhyRC_Sim')
from policy_cli import install_failure_handler
install_failure_handler()
from Policy.environment import DressingEnv

out = Path('/output/native_capture') / time.strftime('%Y%m%dT%H%M%S')
out.mkdir(parents=True, exist_ok=False)
print('CAPTURE_OUTPUT=' + str(out), flush=True)
env = DressingEnv(profile='actor_rgbd', resolution=(512, 384), max_episode_steps=80,
                  render_mode='rgb_array')
import numpy as np
from PIL import Image, ImageDraw
from Policy.state import snapshot

report = {'passed': False, 'source': 'new headless physics steps; not replay',
          'scope': 'original two-Stretch4 scene, FEM cloth, cameras and bounded unloaded arm control',
          'force_ground_truth': 'not collected; normalized traction is not a calibrated force',
          'output': str(out), 'render_hz': 20, 'frames': []}
try:
    obs, info = env.reset(seed=42)
    first_meta, first_arrays = snapshot(env.M, env.cloths, env.rigs)
    (out / 'initial_state.json').write_text(json.dumps(first_meta, indent=2))
    start = time.perf_counter()
    samples = []
    for i in range(81):
        if i:
            action = np.zeros((2, 9), dtype=np.float32)
            action[:, 3] = .15 if i <= 40 else -.15
            if i == 15:
                action[:, 8] = 1
            elif i == 30:
                action[:, 8] = -1
            obs, _, terminated, truncated, info = env.step(action)
        rgb = obs['rgb']
        if not np.isfinite(obs['depth']).all() or rgb.shape != (5, 384, 512, 3):
            raise RuntimeError('Unexpected sensor frame')
        t = float(info['state']['simulation_time_s'])
        canvas = Image.new('RGB', (1536, 800), (22, 25, 30))
        draw = ImageDraw.Draw(canvas)
        for j in range(5):
            x, y = (j % 3)*512, (j // 3)*400
            canvas.paste(Image.fromarray(rgb[j]), (x, y + 16))
            draw.text((x+5, y+2), f"{env.contract['cameras'][j]['id']} | simulation {t:.3f} s", fill='white')
        canvas.save(out / f'frame_{i:04d}.png')
        if i in (0, 40, 80):
            Image.fromarray(rgb[0]).save(out / f'overview_{i:04d}.png')
            for j in range(5):
                Image.fromarray(rgb[j]).save(out / f'camera_{j}_step_{i:04d}.png')
        samples.append({'frame': i, 'simulation_time_s': t,
                        'joint_position': obs['joint_position'].tolist(),
                        'depth_valid_fraction': float(obs['depth_valid'].mean())})
        if i % 10 == 0:
            print(f'CAPTURE_FRAME {i}/80 simulation={t:.3f}', flush=True)
    final_meta, final_arrays = snapshot(env.M, env.cloths, env.rigs)
    report.update(passed=True, frames=samples, wall_seconds=time.perf_counter()-start,
                  physics_hz=1/env.world.get_physics_dt(), control_hz=60,
                  source_commit=info['source_commit'], source_is_dirty=info['source_is_dirty'],
                  robot_count=len(env.rigs), garment_count=len(env.cloths),
                  human_collision_present=bool(env.stage.GetPrimAtPath('/World/Human/CollisionBody')),
                  max_joint_change=max(float(np.max(np.abs(final_arrays[f'robot_{j}/joint_position'] - first_arrays[f'robot_{j}/joint_position']))) for j in range(2)),
                  cloth_max_node_change_m=float(np.max(np.linalg.norm(final_arrays['garment_0/position_world']-first_arrays['garment_0/position_world'],axis=-1))),
                  all_arrays_finite=final_meta['all_arrays_finite'])
    (out / 'final_state.json').write_text(json.dumps(final_meta, indent=2))
    np.savez_compressed(out / 'endpoint_states.npz', **{'first_'+k:v for k,v in first_arrays.items()}, **{'last_'+k:v for k,v in final_arrays.items()})
    (out / 'capture.json').write_text(json.dumps(report, indent=2))
    print('NATIVE-CAPTURE-PASS ' + str(out), flush=True)
except BaseException as exc:
    report['error'] = repr(exc)
    (out / 'capture.json').write_text(json.dumps(report, indent=2))
    raise
finally:
    env.close()
