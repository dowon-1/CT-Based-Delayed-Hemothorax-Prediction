# CT-Based Prediction of Delayed Hemothorax after Rib Fracture

![Project Status](https://img.shields.io/badge/status-under%20development-yellow)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/framework-PyTorch-red)
![Data](https://img.shields.io/badge/clinical%20data-not%20included-lightgrey)

This repository contains preprocessing, quality-control, and deep learning code for CT-based prediction of delayed hemothorax in patients with rib fractures.

The repository is maintained for research documentation, project management, and reproducibility. Raw DICOM images, processed H5 volumes, clinical labels, and trained model checkpoints are not included because they contain protected clinical information.

---

## Project Summary

Delayed hemothorax may develop after traumatic rib fracture even when the initial CT scan shows no clinically significant hemothorax. This project aims to build an imaging-based AI pipeline that can support risk analysis using chest CT data and rib-fracture-related imaging features.

The current workflow includes:

1. DICOM series selection and geometry quality control
2. Slice-position-based DICOM sorting
3. HU conversion and image orientation standardization
4. Lung and rib segmentation-based CT crop generation
5. H5 dataset generation for model training
6. 2D multiple instance learning using ResNet50
7. Optional 3D CNN baseline training
8. Rib fracture feature analysis by fracture extent and anatomical location

---

## Key Features

- **DICOM preprocessing** with FileStart/FileEnd-based series selection
- **Slice sorting** using `ImagePositionPatient` and `ImageOrientationPatient`
- **HU conversion and QC** to verify CT intensity validity
- **Orientation standardization** to a consistent LPS coordinate system
- **Lung and rib crop generation** using segmentation masks
- **H5-based training dataset** with multiple CT windows
- **2D MIL model** using CT slices as instances and each CT volume as a bag
- **3D CNN baseline** using video ResNet-style backbones
- **Rib fracture feature analysis** including 50% and 100% fracture extent features
- **DDP support** for multi-GPU training
- **TensorBoard logging** for training monitoring

---

## Repository Structure

```text
CT_Delayed_Hemothorax_AI/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── configs/
│   ├── preprocessing_config.example.yaml
│   ├── train_2d_mil_config.example.yaml
│   └── train_3d_cnn_config.example.yaml
├── docs/
│   ├── project_overview.md
│   └── repository_structure.md
├── sample/
│   ├── sample_h5_index.csv
│   └── sample_dicom_series_geometry_qc_result.csv
└── src/
    ├── preprocessing/
    │   └── preprocess_dicom_to_lung_rib_h5_full.py
    ├── training/
    │   ├── train_2d_mil_resnet50.py
    │   └── train_3d_cnn.py
    ├── evaluation/
    │   └── evaluate_checkpoints.py
    ├── analysis/
    │   └── rib_fracture_feature_analysis.py
    └── utils/
        ├── metrics.py
        └── seed.py
```

---

## Main Components

### 1. DICOM Preprocessing

```text
src/preprocessing/preprocess_dicom_to_lung_rib_h5_full.py
```

This script converts selected DICOM series into model-ready H5 files.

Main procedures:

- Load a DICOM series list from a geometry QC CSV file
- Select DICOM files between `FileStart` and `FileEnd`
- Sort slices by physical slice position
- Convert DICOM pixel values to HU values
- Verify HU range and air-region intensity
- Standardize orientation to LPS
- Run TotalSegmentator for lung and rib masks
- Generate lung-crop and rib-crop CT volumes
- Resample the z-spacing to the target spacing
- Save H5 files with HU image, CT windows, and segmentation masks
- Export preprocessing summary and QC reports

Expected outputs:

```text
preprocessing_summary.csv
qc_errors.csv
h5_index_summary.csv
orientation_geometry_qc.csv
lung_crop/*.h5
rib_crop/*.h5
qc_png/*.png
```

---

### 2. 2D MIL ResNet50 Training

```text
src/training/train_2d_mil_resnet50.py
```

This script trains a 2D multiple instance learning model. Each CT slice is treated as an instance, and each CT volume is treated as a bag.

Main procedures:

- Load H5 CT volumes
- Keep the original number of slices
- Resize only height and width to 224 × 224
- Apply dataset-level z-score normalization
- Convert a volume into a variable-length slice bag
- Extract slice-level features using ResNet50
- Aggregate slice features using attention-based MIL
- Train with class-imbalance handling
- Save best validation-loss, best composite-score, and last checkpoints
- Log training curves using TensorBoard

Input CSV example:

```csv
id,path,label,split
10016020,/path/to/case.h5,0,train
10016234,/path/to/case.h5,1,val
10017345,/path/to/case.h5,0,test
```

---

### 3. 3D CNN Baseline

```text
src/training/train_3d_cnn.py
```

This script provides an optional 3D CNN baseline for CT volume classification.

Supported backbone examples:

- `r3d_18`
- `r2plus1d_18`
- `mc3_18`

Main procedures:

- Load H5 CT volumes
- Resize X/Y dimensions
- Pad or crop the depth dimension
- Apply dataset-level z-score normalization
- Train a 3D CNN classifier
- Save checkpoints and TensorBoard logs
- Evaluate validation and test performance using a validation-selected threshold

---

### 4. Evaluation

```text
src/evaluation/evaluate_checkpoints.py
```

This file provides a lightweight evaluation template. The current training scripts also include final checkpoint evaluation, including validation-based threshold selection and test-set reporting.

Common metrics:

- AUROC
- AUPRC
- Accuracy
- Balanced accuracy
- Sensitivity
- Specificity
- Precision
- F1 score
- Confusion matrix

---

### 5. Rib Fracture Feature Analysis

```text
src/analysis/rib_fracture_feature_analysis.py
```

This module is intended for patient-level rib fracture feature extraction and comparison.

Example features:

- Total fractured rib count
- Number of ribs with fracture extent ≥ 50%
- Number of ribs with fracture extent = 100%
- R1–R7 and R8–R12 grouped features
- Anterior, lateral, and posterior location features
- Label-wise comparison between delayed hemothorax and non-delayed hemothorax groups

---

## Data Privacy

The following files are intentionally excluded from this repository:

- Raw DICOM files
- Processed H5 volumes
- NIfTI segmentation files
- Patient-level labels
- Full clinical CSV files
- Model checkpoints
- Pickle files and trained weights

Only sample CSV templates are included to show the expected format.

---


For private/local paths, use environment variables instead of editing the source code directly.
See [`docs/configuration.md`](docs/configuration.md) for examples.

## Installation

Create a conda environment:

```bash
conda create -n delayed_hemo_ct python=3.8
conda activate delayed_hemo_ct
```

Install basic dependencies:

```bash
pip install -r requirements.txt
```

Install TotalSegmentator separately if segmentation is required:

```bash
pip install TotalSegmentator
```

---

## Quick Start

### 1. Prepare input metadata

Prepare a DICOM series metadata CSV similar to:

```text
sample/sample_dicom_series_geometry_qc_result.csv
```

The metadata should contain case ID, DICOM folder path, series information, and FileStart/FileEnd information.

---

### 2. Run DICOM preprocessing

Modify the path configuration in:

```text
src/preprocessing/preprocess_dicom_to_lung_rib_h5_full.py
```

Then run:

```bash
python src/preprocessing/preprocess_dicom_to_lung_rib_h5_full.py
```

---

### 3. Prepare H5 index CSV

Prepare an H5 index CSV similar to:

```text
sample/sample_h5_index.csv
```

Required columns:

```text
id,path,label,split
```

---

### 4. Train the 2D MIL model

Single GPU:

```bash
python src/training/train_2d_mil_resnet50.py
```

Multi-GPU:

```bash
torchrun --nproc_per_node=4 src/training/train_2d_mil_resnet50.py
```

---

### 5. Train the 3D CNN baseline

```bash
torchrun --nproc_per_node=4 src/training/train_3d_cnn.py
```

---

## Configuration

Example configuration files are provided in:

```text
configs/
```

These files are templates only. Actual paths, H5 keys, GPU settings, and model parameters should be modified according to the local environment.

Important configurable items include:

- Dataset path
- H5 image key
- Target image size
- Target z-spacing
- Batch size
- Learning rate
- Class-imbalance handling
- Checkpoint directory
- TensorBoard log directory

---

## Current Status

This repository is under active development.

Implemented:

- DICOM preprocessing pipeline
- Slice-position-based sorting
- HU and geometry QC
- Lung/rib segmentation-based crop generation
- H5 dataset generation
- 2D MIL training pipeline
- 3D CNN baseline pipeline
- Rib fracture feature analysis templates

Not included yet:

- Final trained model weights
- Final locked model performance
- Public sample CT data
- Deployment code

---

## Disclaimer

This repository is for research and project documentation purposes only. It is not a clinical decision-support system and should not be used for real-world medical decision-making.

All clinical data must be handled according to institutional review board, privacy, and data security policies.
