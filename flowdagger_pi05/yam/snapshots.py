"""Download immutable dataset files and verify LFS SHA256 / Git blob hashes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import requests


def matches(path, entry):
    if not path.is_file() or path.stat().st_size != entry['size']:
        return False
    h = hashlib.sha256() if entry.get('sha256') else hashlib.sha1()
    if not entry.get('sha256'):
        h.update(f"blob {entry['size']}\0".encode())
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest() == (entry.get('sha256') or entry['blob'])


def fetch(manifest, root, workers=3):
    manifest = json.loads(Path(manifest).read_text())
    root = Path(root)
    def one(entry):
        relative = Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe snapshot path')
        path = root / relative
        if matches(path, entry):
            print('Verified', relative, flush=True)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix+'.download')
        url = f"https://huggingface.co/datasets/{manifest['repo_id']}/resolve/{manifest['revision']}/{entry['path']}"
        for attempt in range(5):
            try:
                with requests.get(url, stream=True, timeout=(30,120)) as response:
                    response.raise_for_status()
                    with tmp.open('wb') as output:
                        for block in response.iter_content(1024*1024):
                            output.write(block)
                if not matches(tmp, entry):
                    raise ValueError(f'Snapshot hash mismatch: {relative}')
                tmp.replace(path)
                print('Downloaded and verified', relative, flush=True)
                return
            except (requests.RequestException, ValueError):
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, manifest['files']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--root', required=True)
    a = p.parse_args()
    fetch(a.manifest, a.root)
