#!/usr/bin/env bash
set -euo pipefail
: "${1:?Usage: encode_native_capture.sh CAPTURE_DIRECTORY}"
CAPTURE=$(realpath "$1")
[[ -f "$CAPTURE/capture.json" ]]
/usr/bin/python3 - "$CAPTURE/capture.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    r = json.load(f)
if not r.get('passed') or len(r.get('frames', [])) != 81:
    raise SystemExit('Refusing to encode incomplete physics capture as validated evidence')
PY
ffmpeg -nostdin -y -framerate 20 -i "$CAPTURE/frame_%04d.png" -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -movflags +faststart "$CAPTURE/phyrc_native_multicamera.mp4"
ffprobe -v error -show_entries stream=codec_name,width,height,nb_frames,r_frame_rate -show_entries format=duration,size -of json "$CAPTURE/phyrc_native_multicamera.mp4" > "$CAPTURE/video_probe.json"
ffmpeg -nostdin -v error -i "$CAPTURE/phyrc_native_multicamera.mp4" -f null -
sha256sum "$CAPTURE/phyrc_native_multicamera.mp4" "$CAPTURE/overview_0000.png" "$CAPTURE/frame_0040.png" > "$CAPTURE/evidence.sha256"
