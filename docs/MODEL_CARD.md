# Model card: NeuroSeg compact U-Net pilot

## Intended use

Local educational and research exploration of multi-modal tumor segmentation
on prepared BraTS-style MRI. Suitable for studying failure modes, reproducible
training and CPU inference. It is not validated for diagnosis, treatment planning,
screening, unsupervised triage or replacing a radiologist.

## Model and inputs

- Compact U-Net, GroupNorm/ReLU blocks, bilinear decoder and skip connections.
- Four input modalities in the exact order FLAIR, T1w, contrast-enhanced T1, T2w.
- 2D mode: 4 channels. Implemented 2.5D mode: 12 channels from neighboring slices.
- Pixel output classes: background, edema, non-enhancing tumor, enhancing tumor.
- Checkpoints include input schema, configuration, training step count, optimizer
  state, random-generator state and processed-manifest fingerprint.
- See METHODS.md for sampling, normalization and reconstruction details.

## Training data and performance

The current pilot uses 16 training cases from a 24-case random subset of the
official MSD Task01 BrainTumour release. Four validation cases select checkpoints;
four held-out cases are reserved for later testing. See DATASET_CARD.md.

Measured results and checkpoint fingerprints appear in EXPERIMENT_RESULTS.md.
Do not substitute training accuracy, resized validation scores, or the best case
for cohort-level original-grid results. The unweighted baseline failed to recover
tumor core and enhancing tumor; this failure is retained in the results.

## Known limits

- Small cohort, one split and seed; no site-stratified or external evaluation.
- Validation-informed development means validation is not an independent final test.
- Prepared input assumptions exclude arbitrary raw clinical MRI and DICOM.
- Square resizing loses detail and can distort in-plane aspect ratios; native-grid
  reconstruction restores array geometry but cannot recover lost information.
- Two-dimensional inference may be inconsistent between slices.
- Predicted zero volume does not establish absence of a tumor sub-region.
- Validation case BRATS_185 has only 108 reference enhancing-tumor voxels
  (0.108 mL at its 1 mm isotropic spacing). Small-region failures should be
  inspected separately; in-plane downsampling can lose important detail.
- Softmax scores are not calibrated probabilities of clinical correctness.
- No demographic subgroup analysis, prospective evaluation or uncertainty calibration.
- Native CPU inference time is measured locally and varies with other system load.

## Operational behavior

The app binds to 127.0.0.1, runs inference locally, and records local report paths
and model fingerprints in SQLite. NIfTI uploads must satisfy shape, modality and
geometry checks. Errors are shown without substituting fabricated segmentations.
Only weights-only compatible checkpoints with recorded training steps are loaded.

## Next evidence needed

More training subjects, a fixed experimental protocol, multiple seeds, matched
compute comparisons of 2D/2.5D models, modality ablations, calibrated uncertainty,
independent external validation, and a final held-out test after configuration
selection. A literature review must establish novelty before publication claims.
Higher-resolution or patch-based experiments should specifically investigate
small-region failures before attributing every error to model architecture.
