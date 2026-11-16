#!/usr/bin/env python3
"""Stage exact upstream downloads via curl, then verify upstream SHA-256.

Optional --hf-mirror / --github-proxy select transport mirrors for the same
pinned commits. No upstream source or lock file is changed.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1] / 'upstream'
sys.path.insert(0, str(root / 'scripts'))
from remote_zip import HTTPRangeReader
import zipfile

def fetch(url, span=None):
    with tempfile.TemporaryDirectory() as td:
        body, headers = Path(td) / 'body', Path(td) / 'headers'
        cmd = ['curl', '-fsSL', '--retry', '2', '--retry-all-errors', '--connect-timeout', '15', '--max-time', '90', '-D', str(headers), '-o', str(body)]
        if span:
            start, end = span
            url += ('&' if '?' in url else '?') + f'phyrc_range={start}-{end}'
            cmd += ['--range', f'{start}-{end}', '--max-filesize', str(max(65536, end-start+1))]
        subprocess.run(cmd + [url], check=True)
        data = body.read_bytes()
        header = headers.read_text()
        if span:
            ranges = re.findall(r'^content-range:\s*(.*?)\s*$', header, re.I | re.M)
            if not ranges or not ranges[-1].startswith(f'bytes {start}-{end}/') or len(data) != end-start+1:
                raise RuntimeError('HTTP Range response invalid')
            return data, ranges[-1]
        return data

class CurlRangeReader(HTTPRangeReader):
    def _request(self, start, end):
        return fetch(self.url, (start, end))

for entry in json.loads((root / 'config/assets.lock.json').read_text())['assets']:
    if not any(key in entry for key in ('url', 'archive_url')):
        continue
    target = root / entry['path']
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == entry['sha256']:
        print('CACHED', entry['path'], flush=True)
        continue
    print('FETCH', entry['path'], flush=True)
    if 'archive_url' in entry:
        url = entry['archive_url']
        if '--hf-mirror' in sys.argv:
            url = url.replace('https://huggingface.co/', 'https://hf-mirror.com/', 1)
        print('TRANSPORT', url, flush=True)
        with zipfile.ZipFile(CurlRangeReader(url)) as z:
            data = z.read(entry['member'])
    else:
        url = entry['url']
        if '--github-proxy' in sys.argv:
            url = 'https://github.chenc.dev/' + url
        print('TRANSPORT', url, flush=True)
        data = fetch(url)
    if hashlib.sha256(data).hexdigest() != entry['sha256']:
        raise ValueError('Locked SHA-256 mismatch: ' + entry['path'])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    print('VERIFIED', entry['path'], len(data), entry['sha256'], flush=True)
