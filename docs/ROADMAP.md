# Implementation and research roadmap

## Constraints and current status

Target machine: Windows, i5-7300U (2 cores/4 threads), approximately 16 GB RAM,
Intel HD 620, no detected NVIDIA GPU, approximately 70 GB free on D: at inspection.
All training, inference, reports and application state should remain local.
CPU PyTorch is installed. The official MSD BraTS-derived 24-case pilot has been
downloaded and validated. Local training, evaluation, reporting and the app are
implemented; see EXPERIMENT_RESULTS.md for measured progress. The later research
and external-validation milestones below are not all complete.

Current milestone: trained unweighted and class-balanced 2D pilots, original-grid
validation, three-plane MRI viewing, reference overlays, reports and local run
history. The 2.5D implementation exists, but a trained controlled 2D-vs-2.5D study
has not yet been performed. Clinical and publication-level claims remain out of scope.

## 1. Data integrity

- Obtain a documented BraTS release through its official access route.
- Record source, version, license, citations, file inventory and checksums.
- Validate four modalities, finite values, dimensions, affine alignment, voxel
  spacing and release-specific label definitions. Reject incompatible inputs.
- Split by patient before generating slices. Check duplicate patients across
  releases if more than one cohort is used.
- Normalize nonzero brain voxels per modality; preserve spatial metadata.
- Keep source volumes immutable. Use bounded caching and lazy loading.

## 2. CPU baseline

- Compact 2D U-Net, four input modalities, four output classes, initial 128x128
  slices and small batches. Map dataset labels explicitly into contiguous IDs.
- Cross-entropy plus soft Dice objective; sampled foreground and background.
- Deterministic patient splits, recorded seeds/configuration, validation-based
  checkpoint selection, early stopping and resumable training.
- Benchmark a small patient pilot before committing to full experiments.
- Reconstruct predictions on the original grid for evaluation and volume reports.

## 3. Research question

Can adjacent-slice context improve tumor sub-region segmentation under a fixed
CPU runtime and memory budget compared with a four-channel 2D baseline?

- Compare the 2D baseline with a 2.5D variant (three slices x four modalities).
- Run controlled ablations for context, augmentation and loss components.
- Evaluate missing-modality robustness only after defining a training strategy;
  the baseline requires all four modalities.
- Optional uncertainty study: repeated stochastic predictions, calibration and
  error correlation. Uncertainty is not a probability of clinical correctness.
- Establish related work before claiming originality. Novelty and publication
  readiness cannot be guaranteed by adding model features.

## 4. Evaluation

- Patient-level Dice and IoU for whole tumor, tumor core and enhancing tumor.
- HD95 in physical units with documented handling of empty reference/prediction.
- Report voxel accuracy as secondary, plus region sensitivity and failure cases.
- Bootstrap confidence intervals over patients, not slices; report cohort sizes.
- Keep the held-out test split untouched by tuning and checkpoint selection.
- Record CPU runtime, peak process memory and parameter count alongside quality.
- A small pilot establishes functionality, not reliable generalization.

## 5. Local application

- Streamlit MRI upload with modality assignment and input validation.
- Slice browsing, modality selection, segmentation overlays and opacity controls.
- Original-grid NIfTI mask export and volume measurements from voxel spacing.
- PDF/CSV reports with model version, configuration, timing and provenance.
- Local SQLite run history; no upload to external inference services.
- Never display random/untrained outputs as meaningful segmentations.
- Initially accept BraTS-style aligned, skull-stripped inputs. Raw clinical MRI
  requires a separately validated preprocessing pipeline.

## 6. Engineering and portfolio delivery

- Modular data, models, training, evaluation and reporting code.
- Isolated dependencies; tests for label mapping, patient isolation, spatial
  metadata, empty-mask metrics, volume calculation and checkpoint loading.
- Structured logs, explicit error handling and reproducible experiment configs.
- Dataset card, model card, architecture diagram, experiment report, limitations
  and a short reproducible demonstration.
- CI can be supplied for future repository use; actual development and execution
  remain local. Publish code and results only when separately requested.

## Completion gates

1. Validated real dataset and reproducible manifest/splits.
2. Measured baseline training and original-grid evaluation.
3. Functional application using a trained, versioned checkpoint.
4. Controlled comparative study and documented results, including failures.

An industry-oriented portfolio demonstrates reliability and maintainability.
A PhD-oriented study additionally needs defensible novelty and robust evidence;
neither label is implied by the scaffold alone.
