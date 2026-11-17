"""Encode the continuous 10 Hz task frames and verify every decoded frame."""
import argparse,subprocess,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('episode',type=Path);a=p.parse_args();run=a.episode.resolve()
frames=sorted(run.glob('frame_*.jpg'))
if not frames or [f.name for f in frames]!=[f'frame_{i:05d}.jpg' for i in range(len(frames))]:raise ValueError('Noncontiguous frames')
fps=json.loads((run/'metadata.json').read_text())['rgb_hz'];output=run/'trajectory.mp4'
subprocess.run(['ffmpeg','-v','error','-y','-framerate',str(fps),'-i',str(run/'frame_%05d.jpg'),'-c:v','libx264','-crf','20','-pix_fmt','yuv420p',str(output)],check=True)
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,nb_read_frames,r_frame_rate:format=duration,size','-of','json',str(output)],text=True))
if int(probe['streams'][0]['nb_read_frames'])!=len(frames):raise ValueError('Lost encoded frames')
subprocess.run(['ffmpeg','-v','error','-i',str(output),'-f','null','-'],check=True)
probe.update(sha256=hashlib.sha256(output.read_bytes()).hexdigest(),decode='passed',source_frame_count=len(frames))
(run/'video_probe.json').write_text(json.dumps(probe,indent=2));print(output)
