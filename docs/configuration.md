# Configuration Notes

The scripts are intentionally written so that local/private paths do not need to be committed.
Most path-like settings can be provided with environment variables.

## Preprocessing

```bash
DATASET_ROOT=/path/to/dataset \
PREPROCESS_CSV=/path/to/dicom_series_geometry_qc_result.csv \
PREPROCESS_OUT_ROOT=/path/to/output/dicom_to_h5_lung_rib \
TOTALSEG_PREFIX=/path/to/totalseg/env \
GPU_ID=0 \
python src/preprocessing/preprocess_dicom_to_lung_rib_h5_full.py
```

## 2D MIL Training

```bash
CSV_PATH=/path/to/h5_index.csv \
CKPT_DIR=/path/to/checkpoints \
LOG_DIR=/path/to/logs/train_2d_mil \
H5_IMAGE_KEY=window_lung \
torchrun --nproc_per_node=4 src/training/train_2d_mil_resnet50.py
```

## 3D CNN Training

```bash
CSV_PATH=/path/to/h5_index.csv \
CKPT_DIR=/path/to/checkpoints \
LOG_DIR=/path/to/logs/train_3d_cnn \
BACKBONE_NAME=r3d_18 \
H5_IMAGE_KEY=window_broad \
torchrun --nproc_per_node=4 src/training/train_3d_cnn.py
```
