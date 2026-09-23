"""Reproducible CPU pilot training with patient-separated validation."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import random
import time

import numpy as np
import psutil
import torch
from torch.nn import functional as F

from neuroseg.data import file_hash
from neuroseg.metrics import binary_metrics, region_masks
from neuroseg.model import CompactUNet


def context_slices(array, indices, context):
    offsets = range(-(context // 2), context // 2 + 1)
    return np.concatenate([np.asarray(array[np.clip(np.asarray(indices) + offset, 0, len(array) - 1)], dtype=np.float32) for offset in offsets], axis=1)


def segmentation_loss(logits, target, class_weights=None):
    probabilities = logits.softmax(1)
    one_hot = F.one_hot(target, 4).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    dice = (2 * (probabilities * one_hot).sum(dims) + 1e-5) / (probabilities.sum(dims) + one_hot.sum(dims) + 1e-5)
    return F.cross_entropy(logits, target, weight=class_weights) + (1 - dice[1:].mean())


def training_class_weights(manifest, root):
    counts = np.zeros(4, dtype=np.int64)
    for case in manifest['splits']['train']:
        labels = np.load(root / f'{case}_y.npy', mmap_mode='r')
        counts += np.bincount(labels.ravel(), minlength=4)
    weights = 1 / np.sqrt(np.maximum(counts, 1) / counts.sum())
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


@torch.inference_mode()
def validate(model, manifest, root, context, batch_size):
    model.eval()
    patients = {}
    for case in manifest['splits']['validation']:
        x = np.load(root / f'{case}_x.npy', mmap_mode='r')
        y = np.load(root / f'{case}_y.npy', mmap_mode='r')
        prediction = np.empty(y.shape, dtype=np.uint8)
        for start in range(0, len(x), batch_size):
            indices = np.arange(start, min(start + batch_size, len(x)))
            prediction[indices] = model(torch.from_numpy(context_slices(x, indices, context))).argmax(1).numpy().astype(np.uint8)
        pred_regions, true_regions = region_masks(prediction), region_masks(y)
        patients[case] = {name: binary_metrics(pred_regions[name], true_regions[name], surface=False)['dice'] for name in pred_regions}
    return float(np.mean([v for regions in patients.values() for v in regions.values()])), patients


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    root, out = Path(args.data), Path(args.out)
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    fingerprint = file_hash(manifest_path)
    config = {k: v for k, v in vars(args).items() if k not in ('resume', 'epochs')}
    model_config = {'in_channels': 4 * args.context, 'classes': 4, 'width': args.width}
    model = CompactUNet(**model_config)
    weights = training_class_weights(manifest, root) if args.class_balance == 'sqrt_inverse' else None
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    epoch_start, best, stale, steps = 0, -1.0, 0, 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        checkpoint_config = dict(checkpoint['config'])
        checkpoint_config.setdefault('class_balance', 'none')
        if checkpoint['manifest_sha256'] != fingerprint or checkpoint_config != config:
            raise ValueError('Resume data/config differ from checkpoint')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_start, best, stale, steps = checkpoint['epoch'] + 1, checkpoint['best_score'], checkpoint['stale'], checkpoint['steps']
        rng.bit_generator.state = checkpoint['numpy_rng']
        torch.set_rng_state(checkpoint['torch_rng'])
    elif out.exists() and any(out.iterdir()):
        raise ValueError('Run directory is not empty; resume or use another output path')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))

    @lru_cache(maxsize=4)
    def arrays(case):
        return (np.load(root / f'{case}_x.npy', mmap_mode='r'), np.load(root / f'{case}_y.npy', mmap_mode='r'))

    for epoch in range(epoch_start, args.epochs):
        started = time.perf_counter()
        model.train()
        losses = []
        for step in range(args.steps_per_epoch):
            case = str(rng.choice(manifest['splits']['train']))
            x, y = arrays(case)
            info = manifest['cases'][case]
            indices = []
            for _ in range(args.batch_size):
                pool = info['tumor_slices'] if rng.random() < .65 and info['tumor_slices'] else info['brain_slices']
                indices.append(int(rng.choice(pool)))
            batch = context_slices(x, indices, args.context)
            target = np.asarray(y[indices], dtype=np.int64)
            if args.augment:
                for axis in (-1, -2):
                    if rng.random() < .5:
                        batch, target = np.flip(batch, axis), np.flip(target, axis)
            optimizer.zero_grad(set_to_none=True)
            loss = segmentation_loss(model(torch.from_numpy(batch.copy())), torch.from_numpy(target.copy()), weights)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(float(loss.detach()))
            steps += 1
        score, patients = validate(model, manifest, root, args.context, args.batch_size)
        improved = score > best
        best, stale = (score, 0) if improved else (best, stale + 1)
        result = {'epoch': epoch + 1, 'steps': steps, 'loss': float(np.mean(losses)),
            'validation_resized_macro_dice': score, 'validation_patients': patients,
            'elapsed_seconds': time.perf_counter() - started,
            'rss_mb_at_epoch_end': psutil.Process().memory_info().rss / 1024**2}
        with (out / 'history.jsonl').open('a') as f:
            f.write(json.dumps(result) + '\n')
        checkpoint = {'format_version': 1, 'model': model.state_dict(), 'model_config': model_config,
            'optimizer': optimizer.state_dict(), 'config': config, 'manifest_sha256': fingerprint,
            'epoch': epoch, 'steps': steps, 'best_score': best, 'stale': stale,
            'numpy_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
            'size': manifest['size'], 'modalities': manifest['modalities'], 'labels': manifest['labels'],
            'status': 'research_pilot', 'validation_resized_macro_dice': score}
        checkpoint['class_weights'] = weights.tolist() if weights is not None else None
        temp = out / 'last.pt.tmp'
        torch.save(checkpoint, temp)
        temp.replace(out / 'last.pt')
        if improved:
            best_temp = out / 'best.pt.tmp'
            torch.save(checkpoint, best_temp)
            best_temp.replace(out / 'best.pt')
        print(json.dumps(result), flush=True)
        if stale >= args.patience:
            print('Early stopping.', flush=True)
            break
    return out / 'best.pt'


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/processed')
    p.add_argument('--out', default='runs/baseline')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--steps-per-epoch', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--width', type=int, default=8)
    p.add_argument('--context', type=int, choices=[1, 3], default=1)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--augment', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--class-balance', choices=['none', 'sqrt_inverse'], default='none')
    p.add_argument('--resume')
    return p
