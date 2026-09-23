"""MSD region metrics on original grids. Empty-mask policies are explicit."""
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def region_masks(labels):
    return {'WT': labels > 0, 'TC': np.isin(labels, [2, 3]), 'ET': labels == 3}


def binary_metrics(pred, truth, spacing=(1, 1, 1), surface=True):
    pred, truth = np.asarray(pred, dtype=bool), np.asarray(truth, dtype=bool)
    if pred.shape != truth.shape:
        raise ValueError('Prediction/reference shapes differ')
    p, t = int(pred.sum()), int(truth.sum())
    intersection = int(np.logical_and(pred, truth).sum())
    result = {'dice': 2 * intersection / (p + t) if p + t else 1.0,
        'iou': intersection / (p + t - intersection) if p + t else 1.0,
        'sensitivity': intersection / t if t else None,
        'reference_voxels': t, 'predicted_voxels': p,
        'empty_status': 'both_empty' if not (p or t) else 'prediction_empty' if not p else 'reference_empty' if not t else 'neither_empty'}
    if surface:
        if p and t:
            ps = pred ^ binary_erosion(pred)
            ts = truth ^ binary_erosion(truth)
            distances = np.concatenate((distance_transform_edt(~ts, sampling=spacing)[ps], distance_transform_edt(~ps, sampling=spacing)[ts]))
            result['hd95_mm'] = float(np.percentile(distances, 95))
        else:
            result['hd95_mm'] = 0.0 if not (p or t) else None
    return result


def evaluate(pred, truth, spacing):
    pm, tm = region_masks(pred), region_masks(truth)
    return {'regions': {name: binary_metrics(pm[name], tm[name], spacing) for name in pm},
        'voxel_accuracy': float(np.mean(pred == truth))}


def volumes_ml(labels, affine):
    voxel_ml = abs(float(np.linalg.det(np.asarray(affine)[:3, :3]))) / 1000
    return {name: float(np.count_nonzero(mask) * voxel_ml) for name, mask in region_masks(labels).items()}
