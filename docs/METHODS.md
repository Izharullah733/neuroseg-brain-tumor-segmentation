# Methods and evaluation protocol

## Baseline

Compact 2D U-Net with two pooling levels, width 8 and 30,020 parameters. Blocks
use two convolutions, GroupNorm and ReLU. Bilinear decoder interpolation aligns
features with skip connections. No pretrained weights are used.

Each MRI modality is standardized using its nonzero voxels, clipped to +/-5
standard deviations and kept zero outside its foreground. The union of nonzero
voxels defines an in-plane crop. Every slice is resized to 128x128. Image resizing
uses bilinear interpolation; class masks use nearest-neighbor interpolation.
Original raw data remain unchanged. Cached input slices use float16 and are
converted to float32 for CPU computation.

For the 2.5D variant, preceding/current/following slices form 12 channels.
Boundary slices are replicated rather than wrapped. Both variants predict the
central slice. The architecture family and evaluation procedure are shared.

## Optimization

Adam, learning rate 0.001, batch size 4, cross-entropy plus foreground soft Dice.
Training samples one case uniformly per batch, then uses a 65% foreground-slice
sampling branch; remaining samples come from all brain-containing slices.
The latter branch can also contain tumor slices. Random in-plane flips augment
inputs and masks together. Clip gradient norm to 5. Fixed seed 42; deterministic
PyTorch algorithms; two CPU threads. There is no accuracy or convergence promise.

An optional balanced-loss experiment uses inverse-square-root class frequencies,
normalized to mean one, as cross-entropy weights. Counts come exclusively from
cached training masks; validation and test labels never set the weights. The
unweighted baseline is retained to expose class-collapse failures rather than
replaced or omitted. The balanced configuration is recorded with each checkpoint.

An epoch is a configured number of sampled optimizer steps, NOT one exhaustive
pass over all cached slices. Epoch/step counts are recorded for this reason.

## Splits and checkpoint selection

24 downloaded cases yield 16 training, 4 validation, 4 held-out test cases.
Validation evaluates all slices for each validation case. Best checkpoint is
selected using mean per-case/per-region Dice on the resized grid. This efficient
selection score is explicitly separate from the original-grid report.

The test split must remain unused for development and model selection. The initial
end-to-end pilot reports validation only. A scientifically meaningful comparison
requires adequate training, repeated seeds and a cohort larger than the pilot.

## Original-grid inference and metrics

Resize logits to the native crop before argmax; place labels in the original
volume and force voxels that are zero in all four modalities to background.
Save NIfTI geometry and spatial units. Report WT, TC and ET volumes using the
absolute determinant of the spatial affine divided by 1000 (mm^3 to mL).

Metrics: case-level Dice, IoU, sensitivity, surface HD95 in millimeters, and
secondary whole-volume voxel accuracy. HD95 here is the 95th percentile of
concatenated bidirectional surface distances, using scipy erosion surfaces. This
implementation is not claimed identical to a particular challenge scorer.

Both masks empty: Dice/IoU = 1 and HD95 = 0. Exactly one mask empty: Dice/IoU = 0,
HD95 = null, accompanied by an explicit empty-status field. Sensitivity is null
when the reference is empty. Summaries include patient-bootstrap Dice intervals
(2,000 resamples, seed 42); intervals from four patients are unstable.

Runtime is measured locally. Training logs currently report process RSS at epoch
end, not peak memory; a dedicated peak-memory benchmark remains future work.

## What this milestone does not establish

No novel algorithm claim, publication-ready result, calibrated uncertainty,
clinical readiness, external validation or regulatory claim. These require
additional work and evidence. The software is a reproducible starting point.
