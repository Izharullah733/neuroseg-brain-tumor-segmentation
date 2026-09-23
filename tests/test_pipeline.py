import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from neuroseg.data import load_case, make_splits, normalized_slices
from neuroseg.inference import predict, load_model
from neuroseg.metrics import binary_metrics, volumes_ml, region_masks
from neuroseg.model import CompactUNet
from neuroseg.training import context_slices


def fixture_volume(tmp_path):
    x = np.zeros((24, 28, 8, 4), dtype=np.float32)
    x[2:22, 3:25, 1:7] = np.random.default_rng(1).normal(100, 20, (20, 22, 6, 4))
    affine = np.diag([2., 3., 4., 1.])
    image = nib.Nifti1Image(x, affine)
    image.header.set_xyzt_units('mm')
    path = tmp_path / 'image.nii.gz'
    nib.save(image, path)
    return path, image, x


def test_case_split_no_leakage():
    ids = [f'case{i}' for i in range(24)]
    splits = make_splits(ids)
    assert splits == make_splits(ids)
    assert set(sum(splits.values(), [])) == set(ids)
    assert sum(map(len, splits.values())) == len(ids)
    with pytest.raises(ValueError):
        make_splits(ids + [ids[0]])


def test_release_labels_and_spatial_alignment(tmp_path):
    path, image, x = fixture_volume(tmp_path)
    label = nib.Nifti1Image(np.zeros(x.shape[:3], dtype=np.uint8), image.affine)
    label.header.set_xyzt_units('mm')
    label_path = tmp_path / 'label.nii.gz'
    nib.save(label, label_path)
    load_case(path, label_path)
    bad = np.zeros(x.shape[:3], dtype=np.uint8)
    bad[0, 0, 0] = 4
    nib.save(nib.Nifti1Image(bad, image.affine, label.header), label_path)
    with pytest.raises(ValueError, match='MSD labels'):
        load_case(path, label_path)
    nib.save(nib.Nifti1Image(np.zeros(x.shape[:3]), np.eye(4), label.header), label_path)
    with pytest.raises(ValueError, match='same spatial grid'):
        load_case(path, label_path)


def test_empty_metrics_and_physical_distance():
    empty = np.zeros((5, 5, 5), dtype=bool)
    point = empty.copy(); point[2, 2, 2] = True
    shifted = empty.copy(); shifted[3, 2, 2] = True
    assert binary_metrics(empty, empty)['dice'] == 1
    assert binary_metrics(point, empty)['hd95_mm'] is None
    assert binary_metrics(empty, point)['dice'] == 0
    assert binary_metrics(point, shifted, (2, 1, 1))['hd95_mm'] == 2
    assert binary_metrics(point, point)['dice'] == 1


def test_region_mapping_and_volume():
    labels = np.array([0, 1, 2, 3]).reshape(2, 2, 1)
    regions = region_masks(labels)
    assert [int(regions[k].sum()) for k in ('WT', 'TC', 'ET')] == [3, 2, 1]
    assert volumes_ml(labels, np.diag([2, 3, 4, 1]))['ET'] == pytest.approx(.024)


def test_context_boundary_has_no_wraparound():
    x = np.arange(4, dtype=np.float32).reshape(4, 1, 1, 1)
    assert context_slices(x, [0], 3).ravel().tolist() == [0, 0, 1]
    assert context_slices(x, [3], 3).ravel().tolist() == [2, 3, 3]


def test_original_grid_reconstruction_and_untrained_guard(tmp_path):
    torch.set_num_threads(2)
    path, image, x = fixture_volume(tmp_path)
    model = CompactUNet()
    for param in model.parameters():
        param.data.zero_()
    model.head.bias.data[3] = 5
    checkpoint = {'format_version': 1, 'steps': 1, 'modalities': ['FLAIR', 'T1w', 't1gd', 'T2w'],
        'model_config': {'in_channels': 4, 'classes': 4, 'width': 8}, 'model': model.state_dict(),
        'size': 16, 'config': {'context': 1}, 'status': 'synthetic_test_only',
        'labels': {'0': 'background', '1': 'edema', '2': 'non-enhancing tumor', '3': 'enhancing tumor'}}
    checkpoint_path = tmp_path / 'synthetic.pt'
    torch.save(checkpoint, checkpoint_path)
    _, _, mask, result = predict(path, checkpoint_path, tmp_path / 'out')
    saved = nib.load(tmp_path / 'out/segmentation.nii.gz')
    assert mask.shape == x.shape[:3]
    assert np.allclose(saved.affine, image.affine)
    assert saved.header.get_xyzt_units()[0] == 'mm'
    assert np.all(mask[np.any(x != 0, axis=-1)] == 3)
    assert np.all(mask[~np.any(x != 0, axis=-1)] == 0)
    checkpoint['steps'] = 0
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match='trained'):
        load_model(checkpoint_path)
