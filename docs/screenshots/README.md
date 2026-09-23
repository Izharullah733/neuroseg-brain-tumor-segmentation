# Dashboard screenshots

These screenshots were supplied by the project owner from the running local
NeuroSeg application on 23 September 2026. They are included unchanged.

- `dashboard.png`: the MRI analysis dashboard with a completed analysis status.
- `segmentation-output.png`: an original MRI slice next to predicted colored
  regions. The overlay is a model prediction, not an expert reference annotation.

## MRI visualization attribution

The MRI content in `segmentation-output.png` derives from **Medical Segmentation
Decathlon, Task01 BrainTumour**, a BraTS-derived dataset. The modification shown
is NeuroSeg's predicted segmentation overlay. The data-derived visualization is
provided under [CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

Source: [official MSD dataset registry](https://registry.opendata.aws/msd/).
Citation: Antonelli et al., *The Medical Segmentation Decathlon*, Nature
Communications 13, 4128 (2022), https://doi.org/10.1038/s41467-022-30695-9.
Further BraTS references and dataset details are in [the dataset card](../DATASET_CARD.md).

This selected visual example does not establish generalization or clinical
readiness. Consult the [complete pilot report](../EXPERIMENT_RESULTS.md), which
also describes small-region failures and the limited validation cohort.
