# Local pilot experiment results

This is a development pilot on 24 MSD BraTS-derived cases: 16 train, 4 validation,
4 held-out test. **Only validation is reported; the test split remains unused.**

## Original-grid validation Dice

| Experiment | WT | TC | ET | Best epoch | Completed epochs | Logged training + validation seconds |
|---|---:|---:|---:|---:|---:|---:|
| balanced_2d | 0.8490 | 0.7632 | 0.6369 | 26 | 34 | 600.4 |
| baseline | 0.7201 | 0.0000 | 0.0000 | 10 | 15 | 265.0 |

WT = whole tumor; TC = tumor core; ET = enhancing tumor.

A zero region score is a model failure, not a missing or omitted result.
Checkpoint selection used resized-grid validation Dice. The table above uses
predictions reconstructed on the native MRI grid, with background masking.

## Interpretation and limits

- Four validation cases cannot establish clinical accuracy or population generalization.
- These are exploratory single-seed runs; some settings were changed after observing validation failures.
- Training runs may have different stopping points; this is not a fixed-runtime causal comparison.
- Logged durations include validation and sometimes concurrent verification; they are not isolated hardware benchmarks.
- Class weights use only training labels. Held-out test labels did not guide these adjustments.
- HD95, IoU, sensitivity, empty-region status and voxel accuracy are in the per-case JSON reports.
- Confidence intervals from four patients are unstable; zero-width intervals do not prove certainty.
- Prospective use, external validation, uncertainty calibration and publication novelty remain unestablished.

## Artifact details

### balanced_2d

- Optimizer steps completed: 1700; selected checkpoint steps: 1300.
- Input channels: 4; width: 8.
- Class weighting: sqrt_inverse.
- Checkpoint SHA256: `ac7659be99c7b880758b70286a0d97cd6e8f37d6c65e6f2559d282f18c7f1ede`.
- Detailed metrics and per-case PDF reports: `runs/balanced_2d_validation/`.

- WT patient-bootstrap 95% interval: [0.7842, 0.9118].
- TC patient-bootstrap 95% interval: [0.5471, 0.8997].
- ET patient-bootstrap 95% interval: [0.2315, 0.8916].

### baseline

- Optimizer steps completed: 750; selected checkpoint steps: 500.
- Input channels: 4; width: 8.
- Class weighting: none.
- Checkpoint SHA256: `263eb86d69c1332dfc4b2b2834a37ba40a6091ca2645b8709dcda53ad1728ec3`.
- Detailed metrics and per-case PDF reports: `runs/baseline_validation/`.

- WT patient-bootstrap 95% interval: [0.6363, 0.8082].
- TC patient-bootstrap 95% interval: [0.0000, 0.0000].
- ET patient-bootstrap 95% interval: [0.0000, 0.0000].
