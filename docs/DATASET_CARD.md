# Dataset card: MSD Task01 BrainTumour pilot

## Source

- Official registry: https://registry.opendata.aws/msd/
- Archive: https://msd-for-monai.s3.us-west-2.amazonaws.com/Task01_BrainTumour.tar
- Dataset metadata name: BRATS
- Release: 2.0 04/05/2018
- Metadata reference: https://www.med.upenn.edu/sbia/brats2017.html
- License: CC-BY-SA 4.0. Dataset material is not included in the code repository.
- Citation: Antonelli et al., The Medical Segmentation Decathlon, Nature
  Communications 13, 4128 (2022), https://doi.org/10.1038/s41467-022-30695-9.
- Also consult and cite the underlying BraTS source publications described by
  the dataset providers.

This is the MSD BraTS-derived release, not BraTS 2021, BraTS 2023 or BraTS-Africa.
Its metadata lists 484 labeled training cases and 266 unlabeled test cases.
Our pilot selects 24 labeled cases with seed 42. It is not a benchmark result
on the complete challenge cohort. Official unlabeled test scans are not used.

## Schema (verified from archive dataset.json)

Channel order: FLAIR, T1w, t1gd (contrast-enhanced T1), T2w.

| ID | Dataset annotation |
|---|---|
| 0 | Background |
| 1 | Edema |
| 2 | Non-enhancing tumor |
| 3 | Enhancing tumor |

Do not silently apply the 0/1/2/4 convention of other BraTS releases. The class-2
annotation is reported using the dataset's own name, not claimed to be a separate
histologically confirmed necrotic-core measurement.

Evaluation regions: WT = {1,2,3}; TC = {2,3}; ET = {3}. Regions overlap.

## Provenance and checks

`data/raw/download_manifest.json` records the remote ETag, file sizes, SHA256
checksums, source and selection. `archive_index.json` stores byte offsets so
additional cases can be retrieved without downloading the entire archive.
The ETag is an object-version guard, not a claimed cryptographic file checksum.

`data/processed/manifest.json` records the selected case IDs, fixed split, source
hashes, geometry and preprocessing crop. Splits are made before slice sampling.
Case IDs are assumed to represent distinct subjects according to the release;
content hashes detect identical files, not all possible repeated-subject scans.
No cross-release cohort mixing is performed.

## Limits

Data are curated, aligned and skull-stripped. Performance does not establish
robustness to unprocessed clinical scans, other diseases, scanner distributions
or missing modalities. The pilot is too small for clinical or broad population
claims. Spatial resampling, preprocessing and site shift require separate study.
