import json

import numpy as np
import torch

from neuroseg.training import parser, train, training_class_weights


def test_resume_matches_uninterrupted_training(tmp_path):
    root = tmp_path / 'data'
    root.mkdir()
    rng = np.random.default_rng(7)
    cases = {}
    for case in ('a', 'b'):
        np.save(root / f'{case}_x.npy', rng.normal(size=(3, 4, 16, 16)).astype(np.float16))
        np.save(root / f'{case}_y.npy', rng.integers(0, 4, size=(3, 16, 16), dtype=np.uint8))
        cases[case] = {'tumor_slices': [0, 1, 2], 'brain_slices': [0, 1, 2]}
    manifest = {'size': 16, 'cases': cases, 'splits': {'train': ['a'], 'validation': ['b'], 'test': []},
        'modalities': ['FLAIR', 'T1w', 't1gd', 'T2w'], 'labels': {'0': 'background'}}
    (root / 'manifest.json').write_text(json.dumps(manifest))
    common = ['--data', str(root), '--steps-per-epoch', '2', '--batch-size', '2']
    full = parser().parse_args(common + ['--out', str(tmp_path / 'full'), '--epochs', '2'])
    train(full)
    resumed = parser().parse_args(common + ['--out', str(tmp_path / 'resumed'), '--epochs', '1'])
    train(resumed)
    resumed.epochs = 2
    resumed.resume = str(tmp_path / 'resumed/last.pt')
    train(resumed)
    a = torch.load(tmp_path / 'full/last.pt', weights_only=True)
    b = torch.load(tmp_path / 'resumed/last.pt', weights_only=True)
    assert a['steps'] == b['steps'] == 4
    for key in a['model']:
        assert torch.equal(a['model'][key], b['model'][key])


def test_class_weights_use_training_labels_only(tmp_path):
    np.save(tmp_path / 'train_y.npy', np.array([0, 0, 0, 0, 1, 2, 3], dtype=np.uint8))
    manifest = {'splits': {'train': ['train'], 'validation': ['missing_validation'], 'test': ['missing_test']}}
    weights = training_class_weights(manifest, tmp_path)
    assert weights[0] < weights[1]
    assert torch.isfinite(weights).all()
    assert torch.isclose(weights.mean(), torch.tensor(1.0))
