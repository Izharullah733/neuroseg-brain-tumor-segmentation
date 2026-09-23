"""Fetch a reproducible labeled pilot from the official MSD tar using HTTP ranges.

No archive-wide extraction or account credentials. Resume reruns validate SHA256.
The saved index allows subsequent runs to request additional patients cheaply.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import random
import tarfile
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

URL = 'https://msd-for-monai.s3.us-west-2.amazonaws.com/Task01_BrainTumour.tar'


class RemoteArchive(io.RawIOBase):
    def __init__(self, session, size, etag):
        self.session, self.size, self.etag = session, size, etag
        self.pos, self.cache_start, self.cache = 0, 0, b''

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def read(self, size=-1):
        if size < 0:
            size = self.size - self.pos
        size = min(size, self.size - self.pos)
        if size <= 0:
            return b''
        if not (self.cache_start <= self.pos and self.pos + size <= self.cache_start + len(self.cache)):
            end = min(self.size - 1, self.pos + max(size, 16384) - 1)
            r = self.session.get(URL, headers={'Range': f'bytes={self.pos}-{end}', 'If-Match': self.etag}, timeout=(15, 90))
            r.raise_for_status()
            if r.status_code != 206 or len(r.content) != end - self.pos + 1:
                raise RuntimeError('Server did not honor byte range')
            self.cache_start, self.cache = self.pos, r.content
        start = self.pos - self.cache_start
        self.pos += size
        return self.cache[start:start + size]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--patients', type=int, default=24)
    p.add_argument('--root', type=Path, default=Path('data/raw'))
    args = p.parse_args()
    if args.patients < 3:
        p.error('Use at least three patients')
    args.root.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
    head = session.head(URL, timeout=30)
    head.raise_for_status()
    etag, size = head.headers['ETag'], int(head.headers['Content-Length'])
    index_path = args.root / 'archive_index.json'
    if index_path.exists():
        index = json.loads(index_path.read_text())
        if index['etag'] != etag:
            raise RuntimeError('Remote archive changed; use a new root to preserve provenance')
    else:
        remote = RemoteArchive(session, size, etag)
        index = {'url': URL, 'etag': etag, 'archive_bytes': size, 'members': {}}
        print('Indexing official archive without downloading image payloads...', flush=True)
        for i, member in enumerate(tarfile.open(fileobj=remote, mode='r:')):
            path = PurePosixPath(member.name)
            if member.isfile() and not path.name.startswith('._'):
                if path.is_absolute() or '..' in path.parts:
                    raise RuntimeError('Unsafe archive path')
                index['members'][member.name] = {'offset': member.offset_data, 'size': member.size}
            if i % 100 == 0:
                print(f'Indexed {i} entries; archive position {remote.tell()/size:.0%}', flush=True)
        index_path.write_text(json.dumps(index, indent=2))

    manifest_path = args.root / 'download_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        'source': URL, 'etag': etag, 'registry': 'https://registry.opendata.aws/msd/',
        'license': 'CC-BY-SA-4.0', 'selection_seed': 42, 'files': {}}

    def download(name):
        dest = args.root.joinpath(*PurePosixPath(name).parts)
        info = index['members'][name]
        known = manifest['files'].get(name)
        if dest.exists() and known and dest.stat().st_size == info['size'] and sha(dest) == known['sha256']:
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        offset, count = info['offset'], info['size']
        with session.get(URL, headers={'Range': f'bytes={offset}-{offset+count-1}', 'If-Match': etag}, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            if response.status_code != 206:
                raise RuntimeError('Expected partial response')
            temp = dest.with_suffix(dest.suffix + '.part')
            with temp.open('wb') as f:
                for chunk in response.iter_content(1024 * 1024):
                    f.write(chunk)
        if temp.stat().st_size != count:
            raise RuntimeError('Incomplete download; rerun to retry')
        temp.replace(dest)
        manifest['files'][name] = {'bytes': count, 'sha256': sha(dest)}
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f'Downloaded {name}: {count/1024**2:.1f} MiB', flush=True)
        return dest

    dataset_path = download('Task01_BrainTumour/dataset.json')
    metadata = json.loads(dataset_path.read_text())
    patients = sorted(metadata['training'], key=lambda x: x['image'])
    random.Random(42).shuffle(patients)
    selected = patients[:args.patients]
    for i, pair in enumerate(selected, 1):
        for field in ('image', 'label'):
            download('Task01_BrainTumour/' + pair[field].removeprefix('./'))
        print(f'Patient {i}/{len(selected)} complete', flush=True)
    manifest['selected_patients'] = [Path(x['image']).name for x in selected]
    manifest['completed_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print('Pilot download complete.', flush=True)


if __name__ == '__main__':
    main()
