"""MSD Task01 data validation, normalization and bounded slice preparation."""
import hashlib
import json
from pathlib import Path
import random

import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F

MODALITIES = ['FLAIR', 'T1w', 't1gd', 'T2w']
LABELS = {0: 'background', 1: 'edema', 2: 'non-enhancing tumor', 3: 'enhancing tumor'}


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_case(image_path, label_path=None):
    image = nib.load(image_path)
    if len(image.shape) != 4 or image.shape[-1] != 4:
        raise ValueError('Expected a 4D NIfTI with FLAIR, T1w, t1gd, T2w channels')
    if np.prod(image.shape, dtype=np.int64) > 80_000_000:
        raise ValueError('Volume exceeds the local pilot memory limit (80 million values)')
    x = image.get_fdata(dtype=np.float32)
    if x.ndim != 4 or x.shape[-1] != 4:
        raise ValueError('Expected a 4D NIfTI with FLAIR, T1w, t1gd, T2w channels')
    if not np.isfinite(x).all() or not np.isfinite(image.affine).all():
        raise ValueError('Non-finite MRI values or affine')
    if abs(np.linalg.det(image.affine[:3, :3])) < 1e-8:
        raise ValueError('Degenerate spatial affine')
    axes = image.affine[:3, :3]
    unit_axes = axes / np.linalg.norm(axes, axis=0)
    if not np.allclose(unit_axes.T @ unit_axes, np.eye(3), atol=1e-4):
        raise ValueError('Sheared grids must be resampled before spacing-based evaluation')
    if image.header.get_xyzt_units()[0] != 'mm':
        raise ValueError('MRI spatial units must explicitly be millimeters')
    for c in range(4):
        if np.count_nonzero(x[..., c]) < 10:
            raise ValueError(f'Modality {MODALITIES[c]} is empty')
    y = None
    if label_path is not None:
        label = nib.load(label_path)
        if label.header.get_xyzt_units()[0] != 'mm':
            raise ValueError('Mask spatial units must explicitly be millimeters')
        y = label.get_fdata(dtype=np.float32)
        if y.shape != x.shape[:3] or not np.allclose(label.affine, image.affine, atol=1e-4):
            raise ValueError('MRI and mask are not on the same spatial grid')
        if not np.isin(y, [0, 1, 2, 3]).all():
            raise ValueError('Expected MSD labels 0,1,2,3; do not use BraTS 0,1,2,4 without conversion')
        y = y.astype(np.uint8)
    return image, x, y


def normalized_slices(x, size=128):
    """Return [Z,C,H,W] and in-plane crop bounds; preserve every axial slice."""
    brain = np.any(x != 0, axis=-1)
    positions = np.where(np.any(brain, axis=2))
    if not len(positions[0]):
        raise ValueError('Empty brain volume')
    x0, x1 = int(positions[0].min()), int(positions[0].max()) + 1
    y0, y1 = int(positions[1].min()), int(positions[1].max()) + 1
    cropped = x[x0:x1, y0:y1].copy()
    for c in range(4):
        v = cropped[..., c]
        foreground = v != 0
        values = v[foreground]
        if values.std() < 1e-6:
            raise ValueError(f'Constant foreground modality {MODALITIES[c]}')
        v[foreground] = np.clip((values - values.mean()) / values.std(), -5, 5)
    slices = torch.from_numpy(np.ascontiguousarray(cropped.transpose(2, 3, 0, 1)))
    slices = F.interpolate(slices, size=(size, size), mode='bilinear', align_corners=False)
    return slices.numpy(), (x0, x1, y0, y1)


def make_splits(ids, seed=42):
    ids = sorted(ids)
    if len(ids) < 6 or len(set(ids)) != len(ids):
        raise ValueError('At least six unique case IDs required')
    random.Random(seed).shuffle(ids)
    held = max(1, round(len(ids) * .15))
    return {'train': ids[2 * held:], 'validation': ids[held:2 * held], 'test': ids[:held]}


def prepare(root=Path('data/raw/Task01_BrainTumour'), out=Path('data/processed'), size=128):
    root, out = Path(root), Path(out)
    metadata = json.loads((root / 'dataset.json').read_text())
    if [metadata['modality'][str(i)] for i in range(4)] != MODALITIES:
        raise ValueError('Unexpected modality order')
    if metadata['labels'] != {'0': 'background', '1': 'edema', '2': 'non-enhancing tumor', '3': 'enhancing tumour'}:
        raise ValueError('Unexpected release-specific labels')
    out.mkdir(parents=True, exist_ok=True)
    existing = out / 'manifest.json'
    if existing.exists():
        previous = json.loads(existing.read_text())
        available = {}
        for entry in metadata['training']:
            ip, lp = root / entry['image'].removeprefix('./'), root / entry['label'].removeprefix('./')
            if ip.exists() and lp.exists():
                available[ip.name.removesuffix('.nii.gz')] = (ip, lp)
        if set(available) != set(previous['cases']) or previous['size'] != size:
            raise ValueError('Existing cohort/size differs; use a new output directory to preserve splits')
        for case, (ip, lp) in available.items():
            info = previous['cases'][case]
            if file_hash(ip) != info['sha256'] or file_hash(lp) != info['label_sha256']:
                raise ValueError('Source data changed; use a new output directory')
            if not (out / f'{case}_x.npy').exists() or not (out / f'{case}_y.npy').exists():
                raise ValueError('Existing cache is incomplete; prepare into a new directory')
        print('Existing manifest and source hashes verified; keeping original splits.', flush=True)
        return previous
    cases, seen = {}, set()
    for entry in metadata['training']:
        ip, lp = root / entry['image'].removeprefix('./'), root / entry['label'].removeprefix('./')
        if not ip.exists() or not lp.exists():
            continue
        case = ip.name.removesuffix('.nii.gz')
        digest = file_hash(ip)
        if digest in seen:
            raise ValueError('Duplicate MRI file content')
        seen.add(digest)
        image, x, y = load_case(ip, lp)
        slices, crop = normalized_slices(x, size)
        a, b, c, d = crop
        masks = torch.from_numpy(y[a:b, c:d].transpose(2, 0, 1).copy()).float().unsqueeze(1)
        masks = F.interpolate(masks, size=(size, size), mode='nearest')[:, 0].numpy().astype(np.uint8)
        np.save(out / f'{case}_x.npy', slices.astype(np.float16))
        np.save(out / f'{case}_y.npy', masks)
        cases[case] = {'image': str(ip.resolve()), 'label': str(lp.resolve()),
            'sha256': digest, 'label_sha256': file_hash(lp), 'shape': list(x.shape),
            'spacing_mm': [float(v) for v in image.header.get_zooms()[:3]], 'crop': crop,
            'tumor_slices': np.flatnonzero(np.any(masks > 0, axis=(1, 2))).tolist(),
            'brain_slices': np.flatnonzero(np.any(slices != 0, axis=(1, 2, 3))).tolist()}
        print(f'Validated and cached {case}', flush=True)
    manifest = {'dataset': 'MSD Task01 BrainTumour', 'size': size, 'modalities': MODALITIES,
        'labels': LABELS, 'cases': cases, 'splits': make_splits(list(cases)), 'seed': 42}
    existing.write_text(json.dumps(manifest, indent=2))
    print(f'Prepared {len(cases)} cases; splits: ' + str({k:len(v) for k,v in manifest['splits'].items()}), flush=True)
    return manifest
