# Repository Structure

This repository keeps the main scripts separate by function.

```text
src/preprocessing/  DICOM preprocessing and H5 generation
src/training/       2D MIL and 3D CNN training scripts
src/evaluation/     Checkpoint evaluation templates
src/analysis/       Rib fracture feature analysis templates
src/utils/          Shared utility functions
configs/            Example configuration files
sample/             Example CSV formats only
```

Raw CT data, labels, and model weights are not included.

- `docs/configuration.md`: examples for environment-variable-based configuration.
