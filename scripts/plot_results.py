"""Generate standalone training curves and original-grid validation examples."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nibabel as nib
import numpy as np


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, default=Path('runs/baseline'))
    p.add_argument('--evaluation', type=Path, default=Path('runs/baseline_validation'))
    p.add_argument('--out', type=Path, default=Path('docs/figures'))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    history = [json.loads(line) for line in (args.run / 'history.jsonl').read_text().splitlines() if line]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([r['epoch'] for r in history], [r['loss'] for r in history], 'o-', color='#247b83')
    axes[0].set(title='Training objective', xlabel='Epoch', ylabel='CE + foreground Dice loss')
    axes[1].plot([r['epoch'] for r in history], [r['validation_resized_macro_dice'] for r in history], 'o-', color='#c76535')
    axes[1].set(title='Resized-grid validation (pilot)', xlabel='Epoch', ylabel='Mean case/region Dice', ylim=(0, 1))
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(args.out / f'{args.run.name}_training.png', dpi=180)
    plt.close(fig)
    manifest = json.loads((args.run / 'manifest.json').read_text())
    colors = ListedColormap(['black', '#4ed7b1', '#ffbd69', '#ee6b97'])
    for case in manifest['splits']['validation']:
        info = manifest['cases'][case]
        prediction_path = args.evaluation / case / 'segmentation.nii.gz'
        if not prediction_path.exists():
            continue
        x = nib.load(info['image']).get_fdata(dtype=np.float32)
        y = nib.load(info['label']).get_fdata().astype(np.uint8)
        pred = nib.load(prediction_path).get_fdata().astype(np.uint8)
        z = int(np.argmax((y > 0).sum(axis=(0, 1))))
        fig, axes = plt.subplots(1, 3, figsize=(11, 4))
        for ax, mask, title in zip(axes, [None, y, pred], ['FLAIR', 'Reference', 'Prediction']):
            ax.imshow(x[:, :, z, 0].T, origin='lower', cmap='gray')
            if mask is not None:
                ax.imshow(np.ma.masked_equal(mask[:, :, z].T, 0), origin='lower', cmap=colors, vmin=0, vmax=3, alpha=.55, interpolation='nearest')
            ax.set_title(title)
            ax.axis('off')
        fig.suptitle(f'{case} | slice {z} with largest reference tumor area | research pilot')
        fig.tight_layout()
        fig.savefig(args.out / f'{args.run.name}_{case}.png', dpi=150)
        plt.close(fig)
    print(f'Figures saved to {args.out}')
