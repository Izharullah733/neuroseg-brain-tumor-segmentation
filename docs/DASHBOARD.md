# Dashboard presentation update

The local Streamlit app uses a light research workspace with a dark navigation
sidebar, teal accents, responsive cards and locally bundled CSS. It does not
request remote fonts or images.

## MRI analysis

- Local cohort and trained-checkpoint counts are read from actual artifacts.
- A local validation example shows a real, unannotated FLAIR preview before inference.
- Upload help distinguishes one four-channel volume from four aligned volumes.
- The 256 MB upload cap is explained as a maximum, not an expected file size.
- Four-file upload controls use a compact two-column arrangement.
- The imaging workspace groups modality, plane, slice and overlay controls.
- Predicted volume cards use full region names and explain overlapping regions.
- PDF, CSV and NIfTI exports are labeled by their purpose.
- Model provenance is available in an expandable detail panel.

## Experiments and history

The experiment page separates original-grid validation results from the
resized-grid scores used during training. Loss and Dice have separate charts;
Dice uses a fixed 0-to-1 scale. No performance figures are hardcoded.

Run history shows readable IDs and timestamps, reports unavailable files, and
provides downloads for existing reports. Report timestamps are explicitly UTC.

## Verification

The existing real-checkpoint AppTest completed prediction, three viewing planes,
reference comparison, volume cards and all navigation pages. Additional AppTest
checks covered both upload modes and disabled prediction without complete input.
The local HTTP health endpoint returned 200. CSS includes narrow-screen rules;
these functional checks do not establish a visual review at every viewport size.

The underlying trained models, dataset splits and evaluation results are unchanged.
Larger-cohort studies, external validation and small-region improvements remain
research priorities; visual polish does not establish model reliability.
