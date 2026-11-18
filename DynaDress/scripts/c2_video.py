"""Encode actual rendered frames with time-aligned C2 belief/probe annotations."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def clock(t):
    ms=round(t*1000);return f'{ms//3600000:02d}:{ms//60000%60:02d}:{ms//1000%60:02d},{ms%1000:03d}'


def main():
    p=argparse.ArgumentParser();p.add_argument('--episode',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from PIL import Image,ImageDraw,ImageFont
    path=a.episode;a.output.mkdir(parents=True,exist_ok=True)
    frames=[json.loads(x) for x in (path/'frame_index.jsonl').read_text().splitlines()]
    updates=[json.loads(x) for x in (path/'belief.jsonl').read_text().splitlines()]
    raw=json.loads((path/'samples.jsonl').read_text().splitlines()[0]);offset=raw['simulation_time_s']
    probe=json.loads((path/'probe_selection.json').read_text());font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    rendered=a.output/'annotated_frames';rendered.mkdir(exist_ok=True)
    for item in frames:
        t=item['simulation_time_s']-offset;arrived=[u for u in updates if u['observed_until_s']<=t+1e-6]
        if arrived:
            s=arrived[-1]['after']['full'];label=f"K={s['mean'][0]:.1f} D={s['mean'][1]:.2f} | effective resistance: "+max(s['effective_sliding_resistance'],key=s['effective_sliding_resistance'].get)
        else:label='Uniform parameter prior'
        active=probe['trigger_s']<=t<probe['trigger_s']+probe['duration_s']
        banner=f"C2 {probe['mode']} | t={t:.2f}s | {item['state']} | "+(probe['selected'] if active else 'ordinary servo')
        image=Image.open(path/f"frame_{item['frame']:05d}.jpg").convert('RGB');draw=ImageDraw.Draw(image)
        draw.rectangle((0,0,image.width,48),fill=(12,18,25));draw.text((8,3),banner,font=font,fill=(255,225,80) if active else 'white');draw.text((8,25),label,font=font,fill='white')
        image.save(rendered/f"frame_{item['frame']:05d}.jpg",quality=90)
    out=a.output/'active_probe.mp4'
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate','8','-i',str(rendered/'frame_%05d.jpg'),'-c:v','libx264','-preset','medium','-crf','24','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],check=True)
    # Keep a genuine in-probe still, not a schematic illustration.
    closest=min(frames,key=lambda v:abs(v['simulation_time_s']-offset-probe['trigger_s']-.5))
    Image.open(rendered/f"frame_{closest['frame']:05d}.jpg").save(a.output/'active_probe.png')
    (a.output/'video_receipt.json').write_text(json.dumps(dict(source=str(path),frames=len(frames),fps=8,
        probe=probe['selected'],trigger_s=probe['trigger_s'],video=str(out),time_offset_s=offset,
        note='Actual native RGB frames; annotations use only already-arrived online beliefs'),indent=2))
    print('C2_VIDEO_COMPLETE',out,len(frames))
if __name__=='__main__':main()
