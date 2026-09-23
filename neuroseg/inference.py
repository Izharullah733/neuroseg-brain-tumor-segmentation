"""CPU inference with original-grid reconstruction and provenance."""
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F

from neuroseg.data import file_hash, load_case, normalized_slices, MODALITIES
from neuroseg.metrics import volumes_ml
from neuroseg.model import CompactUNet
from neuroseg.training import context_slices


def load_model(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format_version') != 1 or checkpoint.get('steps', 0) < 1:
        raise ValueError('A trained NeuroSeg checkpoint is required')
    if checkpoint['modalities'] != MODALITIES:
        raise ValueError('Checkpoint modality schema mismatch')
    model = CompactUNet(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, checkpoint


@torch.inference_mode()
def predict(image_path, checkpoint_path, output_dir, batch_size=4):
    torch.set_num_threads(2)
    started = time.perf_counter()
    image, x, _ = load_case(image_path)
    model, checkpoint = load_model(checkpoint_path)
    slices, crop = normalized_slices(x, checkpoint['size'])
    a, b, c, d = crop
    mask = np.zeros(x.shape[:3], dtype=np.uint8)
    for start in range(0, len(slices), batch_size):
        indices = np.arange(start, min(start + batch_size, len(slices)))
        batch = context_slices(slices, indices, checkpoint['config']['context'])
        logits = model(torch.from_numpy(batch))
        restored = F.interpolate(logits, size=(b - a, d - c), mode='bilinear', align_corners=False)
        labels = restored.argmax(1).numpy().astype(np.uint8)
        mask[a:b, c:d, start:start + len(indices)] = labels.transpose(1, 2, 0)
    mask[~np.any(x != 0, axis=-1)] = 0
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    header = image.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_slope_inter(1, 0)
    segmentation = nib.Nifti1Image(mask, image.affine, header)
    nib.save(segmentation, out / 'segmentation.nii.gz')
    result = {'input_sha256': file_hash(image_path), 'checkpoint_sha256': file_hash(checkpoint_path),
        'checkpoint': str(Path(checkpoint_path).resolve()), 'training_steps': checkpoint['steps'],
        'model_status': checkpoint['status'], 'modality_order': MODALITIES,
        'label_schema': checkpoint['labels'], 'shape': list(mask.shape),
        'volumes_ml': volumes_ml(mask, image.affine), 'elapsed_seconds': time.perf_counter() - started,
        'postprocessing': 'Force all-modalities-zero voxels to background',
        'warning': 'Small-cohort research pilot; not clinically validated.'}
    return image, x, mask, result
