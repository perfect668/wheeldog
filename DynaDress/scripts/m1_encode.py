"""Encode actual diagnostic frames, including failures, and verify full decode."""
import argparse,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--fps',type=int,required=True);a=p.parse_args();run=Path(a.run).resolve();frames=sorted(run.glob('frame_*.png'))
if not frames or [x.name for x in frames]!=[f'frame_{i:04d}.png' for i in range(len(frames))]:raise ValueError('Missing or noncontiguous frames')
output=run/'diagnostic.mp4'
subprocess.run(['ffmpeg','-v','error','-y','-framerate',str(a.fps),'-i',str(run/'frame_%04d.png'),'-c:v','libx264','-crf','20','-pix_fmt','yuv420p',str(output)],check=True)
probe=subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,nb_read_frames,r_frame_rate:format=duration,size','-of','json',str(output)],text=True)
d=json.loads(probe)
if int(d['streams'][0]['nb_read_frames'])!=len(frames):raise ValueError('Frame count mismatch')
subprocess.run(['ffmpeg','-v','error','-i',str(output),'-f','null','-'],check=True)
(run/'video_probe.json').write_text(probe)
paths=[output,*run.glob('*.json'),*run.glob('response.npz'),*run.glob('motion.npz')]
(run/'evidence.sha256').write_text(''.join(hashlib.sha256(x.read_bytes()).hexdigest()+'  '+x.name+'\n' for x in paths))
print(json.dumps({'path':str(output),'frames':len(frames),'decode':'passed'}))
