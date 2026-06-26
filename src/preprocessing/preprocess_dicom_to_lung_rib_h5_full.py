# ============================================================
# DICOM preprocessing pipeline
# This script was organized for GitHub/project management.
# Edit paths and configuration variables before running.
# Raw clinical data and model weights are not included.
# ============================================================

# preprocess_dicom_to_lung_rib_h5_full.py
# ============================================================
# DICOM-to-H5 preprocessing pipeline for lung/rib crops
#
# Purpose
# - Keep the original DICOM files unchanged
# - Select DICOM files from FileStart to FileEnd
# - Sort slices by slice position
# - Convert DICOM pixel values to HU float32 volumes
# - Run HU QC by checking whether air is near -1000 HU
# - Standardize orientation to LPS
# - Run TotalSegmentator directly from the DICOM folder
# - Save lung-mask-based cropped H5 files
# - Save rib-mask-based cropped H5 files
# - Resample final H5 volumes to 2.5-mm z-spacing
# - Store HU int16, four windowed images, and one mask per H5 file
# - Save only a small number of PNG files for QC
# - Output CSV files:
#   1) preprocessing_summary.csv
#   2) qc_errors.csv
#   3) h5_index_summary.csv
#   4) orientation_geometry_qc.csv
# ============================================================

import os
import re
import json
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm


# ============================================================
# 0. CONFIG
# ============================================================

# ============================================================
# 0. CONFIG
# ============================================================

DATASET_ROOT = Path(os.getenv("DATASET_ROOT", "./data"))

# Input CSV
CSV_PATH = Path(
    os.getenv(
        "PREPROCESS_CSV",
        str(DATASET_ROOT / "dicom_series_geometry_qc_result.csv"),
    )
)

# Output directory
OUT_ROOT = Path(
    os.getenv(
        "PREPROCESS_OUT_ROOT",
        str(DATASET_ROOT / "dicom_to_h5_lung_rib"),
    )
)

LUNG_H5_DIR = OUT_ROOT / "lung_crop"
RIB_H5_DIR = OUT_ROOT / "rib_crop"
QC_PNG_DIR = OUT_ROOT / "qc_png"
TMP_ROOT = OUT_ROOT / "_tmp_totalseg"

SUMMARY_CSV = OUT_ROOT / "preprocessing_summary.csv"
QC_ERROR_CSV = OUT_ROOT / "qc_errors.csv"
H5_INDEX_CSV = OUT_ROOT / "h5_index_summary.csv"
ORIENTATION_QC_CSV = OUT_ROOT / "orientation_geometry_qc.csv"

# TotalSegmentator environment
TOTALSEG_PREFIX = Path(os.getenv("TOTALSEG_PREFIX", "./totalseg_env"))
TOTALSEG_BIN = Path(
    os.getenv("TOTALSEG_BIN", str(TOTALSEG_PREFIX / "bin" / "TotalSegmentator"))
)
TOTALSEG_PY = Path(
    os.getenv("TOTALSEG_PY", str(TOTALSEG_PREFIX / "bin" / "python"))
)

GPU_ID = int(os.getenv("GPU_ID", "0"))
USE_ROBUST_CROP = True

# Target z-spacing for model input
TARGET_Z_SPACING = float(os.getenv("TARGET_Z_SPACING", "2.5"))

# Additional z-axis margin around lung/rib masks
CROP_MARGIN_MM = float(os.getenv("CROP_MARGIN_MM", "5"))

# Number of cases saved as QC PNG images
QC_PNG_NUM_CASES = int(os.getenv("QC_PNG_NUM_CASES", "5"))

# H5 compression
H5_COMPRESSION = "gzip"
H5_COMPRESSION_OPTS = 4
# Lower HU clipping value for H5 storage
# Values outside the DICOM FOV or padding values near -3024 are mapped to -1024 for training
HU_STORAGE_MIN = -1024.0
WINDOWS = {
    "lung":  {"WL": -600, "WW": 1500},
    "soft":  {"WL": 40,   "WW": 400},
    "bone":  {"WL": 450,  "WW": 1100},
    "broad": {"WL": 250,  "WW": 2500},
}

LUNG_LABELS = [
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
]

RIB_LABELS = (
    [f"rib_left_{i}" for i in range(1, 13)]
    + [f"rib_right_{i}" for i in range(1, 13)]
)

TARGET_LABELS = LUNG_LABELS + RIB_LABELS


# ============================================================
# 1. Basic utilities
# ============================================================

def ensure_dirs():
    for p in [OUT_ROOT, LUNG_H5_DIR, RIB_H5_DIR, QC_PNG_DIR, TMP_ROOT]:
        p.mkdir(parents=True, exist_ok=True)


def is_valid_value(x):
    if x is None:
        return False
    try:
        if pd.isna(x):
            return False
    except Exception:
        pass
    if str(x).strip() == "":
        return False
    if str(x).strip().lower() == "nan":
        return False
    return True


def sanitize_name(x: str) -> str:
    x = str(x)
    x = x.replace("\\", "/").rstrip("/")
    x = x.split("/")[-1]
    x = re.sub(r"[^A-Za-z0-9_.-]+", "_", x)
    if x == "":
        x = "unknown_case"
    return x


def make_case_id(row):
    case_folder = row.get("CaseFolder", None)
    series_uid = row.get("SeriesInstanceUID", None)
    chest_dir = row.get("ChestDir", None)

    if is_valid_value(case_folder):
        return sanitize_name(case_folder)
    if is_valid_value(series_uid):
        return sanitize_name(series_uid)
    if is_valid_value(chest_dir):
        return sanitize_name(Path(str(chest_dir)).name)

    return "unknown_case"


def natural_key(path):
    name = Path(path).name
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def make_totalseg_env(gpu_id: int):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PATH"] = str(TOTALSEG_PREFIX / "bin") + ":" + env.get("PATH", "")
    env["CONDA_PREFIX"] = str(TOTALSEG_PREFIX)

    # Avoid CUDA library conflicts
    env.pop("LD_LIBRARY_PATH", None)
    env["LD_LIBRARY_PATH"] = str(TOTALSEG_PREFIX / "lib")
    return env


def gpu_healthcheck_or_die():
    code = """
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
x = torch.randn(1, 1, 16, 64, 64, device="cuda")
w = torch.randn(4, 1, 3, 3, 3, device="cuda")
y = torch.nn.functional.conv3d(x, w, padding=1)
torch.cuda.synchronize()
print("CONV3D_OK")
"""
    proc = subprocess.run(
        [str(TOTALSEG_PY), "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=make_totalseg_env(GPU_ID),
    )

    if proc.returncode != 0 or "CONV3D_OK" not in proc.stdout:
        print(proc.stdout)
        raise RuntimeError("GPU healthcheck failed.")


# ============================================================
# 2. String conversion utilities
# ============================================================

def size_to_str(size):
    return "|".join([str(int(x)) for x in size])


def spacing_to_str(spacing, ndigits=6):
    return "|".join([str(round(float(x), ndigits)) for x in spacing])


def origin_to_str(origin, ndigits=6):
    return "|".join([str(round(float(x), ndigits)) for x in origin])


def direction_to_str(direction, ndigits=6):
    return "|".join([str(round(float(x), ndigits)) for x in direction])


def safe_orientation_code(img: sitk.Image):
    try:
        return sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
            img.GetDirection()
        )
    except Exception:
        return "UNKNOWN"


# ============================================================
# 3. DICOM selection and slice-position sorting
# ============================================================

def list_dicom_files(chest_dir: Path):
    files = [p for p in chest_dir.rglob("*") if p.is_file()]
    files = sorted(files, key=natural_key)
    return files


def resolve_file_range(chest_dir: Path, file_start, file_end):
    """
    Select files between FileStart and FileEnd within ChestDir.
    First select by filename range, then reorder by slice position.
    """
    all_files = list_dicom_files(chest_dir)

    if len(all_files) == 0:
        raise FileNotFoundError(f"No DICOM files found in {chest_dir}")

    start_name = Path(str(file_start)).name
    end_name = Path(str(file_end)).name

    name_to_indices = {}
    for i, p in enumerate(all_files):
        name_to_indices.setdefault(p.name, []).append(i)

    if start_name in name_to_indices and end_name in name_to_indices:
        i0 = name_to_indices[start_name][0]
        i1 = name_to_indices[end_name][0]

        if i0 <= i1:
            selected = all_files[i0:i1 + 1]
        else:
            selected = all_files[i1:i0 + 1]

        range_selection_status = "USED_FILESTART_FILEEND"
    else:
        selected = all_files
        range_selection_status = "USED_ALL_FILES_FILESTART_FILEEND_NOT_MATCHED"

    return selected, range_selection_status, len(all_files)


def read_dicom_header(path: Path):
    return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)


def get_slice_position(ds, normal=None):
    """
    Compute slice position using ImagePositionPatient and ImageOrientationPatient.
    """
    if hasattr(ds, "ImagePositionPatient") and hasattr(ds, "ImageOrientationPatient"):
        ipp = np.array(ds.ImagePositionPatient, dtype=np.float64)
        iop = np.array(ds.ImageOrientationPatient, dtype=np.float64)

        row_cos = iop[:3]
        col_cos = iop[3:]

        if normal is None:
            normal = np.cross(row_cos, col_cos)
            normal = normal / (np.linalg.norm(normal) + 1e-12)

        return float(np.dot(ipp, normal))

    if hasattr(ds, "InstanceNumber"):
        return float(ds.InstanceNumber)

    raise ValueError("No ImagePositionPatient/InstanceNumber for slice sorting.")


def sort_by_slice_position(paths):
    """
    Sort selected DICOM files by slice position.
    """
    headers = []
    first_iop = None

    for p in paths:
        ds = read_dicom_header(p)
        headers.append(ds)

        if first_iop is None and hasattr(ds, "ImageOrientationPatient"):
            first_iop = np.array(ds.ImageOrientationPatient, dtype=np.float64)

    normal = None
    if first_iop is not None:
        row_cos = first_iop[:3]
        col_cos = first_iop[3:]
        normal = np.cross(row_cos, col_cos)
        normal = normal / (np.linalg.norm(normal) + 1e-12)

    positions = []
    instance_numbers = []
    for p, ds in zip(paths, headers):
        pos = get_slice_position(ds, normal=normal)
        positions.append(pos)
        instance_numbers.append(getattr(ds, "InstanceNumber", np.nan))

    order = np.argsort(positions)

    sorted_paths = [paths[i] for i in order]
    sorted_positions = np.array([positions[i] for i in order], dtype=np.float32)
    sorted_instance_numbers = np.array([instance_numbers[i] for i in order], dtype=np.float32)

    return sorted_paths, sorted_positions, sorted_instance_numbers


# ============================================================
# 4. DICOM to HU float32 and LPS orientation standardization
# ============================================================

def read_dicom_series_as_sitk(sorted_paths):
    """
    Read the DICOM series using SimpleITK.
    - Use the slice-position-sorted file list
    - Convert image values to HU float32
    - Standardize orientation to LPS
    - Return an orientation QC report
    """
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(p) for p in sorted_paths])
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    img = reader.Execute()
    img = sitk.Cast(img, sitk.sitkFloat32)

    before_size = img.GetSize()
    before_spacing = img.GetSpacing()
    before_origin = img.GetOrigin()
    before_direction = img.GetDirection()
    before_orientation_code = safe_orientation_code(img)

    # Standardize orientation to LPS
    img_lps = sitk.DICOMOrient(img, "LPS")

    after_size = img_lps.GetSize()
    after_spacing = img_lps.GetSpacing()
    after_origin = img_lps.GetOrigin()
    after_direction = img_lps.GetDirection()
    after_orientation_code = safe_orientation_code(img_lps)

    orientation_changed = before_direction != after_direction
    orientation_lps_ok = after_orientation_code == "LPS"

    arr = sitk.GetArrayFromImage(img_lps).astype(np.float32)  # [Z,Y,X]

    orientation_report = {
        "orientation_before_code": before_orientation_code,
        "orientation_after_code": after_orientation_code,
        "orientation_lps_ok": bool(orientation_lps_ok),
        "orientation_changed_to_LPS": bool(orientation_changed),

        "sitk_size_before_xyz": size_to_str(before_size),
        "sitk_size_after_xyz": size_to_str(after_size),
        "sitk_spacing_before_xyz": spacing_to_str(before_spacing),
        "sitk_spacing_after_xyz": spacing_to_str(after_spacing),
        "sitk_origin_before_xyz": origin_to_str(before_origin),
        "sitk_origin_after_xyz": origin_to_str(after_origin),
        "sitk_direction_before": direction_to_str(before_direction),
        "sitk_direction_after": direction_to_str(after_direction),

        # H5 arrays are always stored as [Z, Y, X]
        "h5_axis_order": "ZYX",
        "array_shape_after_lps_zyx": size_to_str(arr.shape),
    }

    return img_lps, arr, orientation_changed, orientation_report


def verify_hu_or_die(hu_zyx: np.ndarray, case_id: str):
    """
    Validate HU values.
    Raise an error on failure and skip saving the case.

    QC checks:
    - dtype is float32
    - all values are finite
    - air voxels are near -1000 HU
    - HU range is plausible for CT
    """
    if hu_zyx.dtype != np.float32:
        raise ValueError(f"[{case_id}] HU dtype is not float32: {hu_zyx.dtype}")

    finite_ratio = float(np.isfinite(hu_zyx).mean())
    if finite_ratio < 0.999:
        raise ValueError(
            f"[{case_id}] HU contains non-finite values. "
            f"finite_ratio={finite_ratio:.6f}"
        )

    hu_min = float(np.min(hu_zyx))
    hu_max = float(np.max(hu_zyx))
    hu_mean = float(np.mean(hu_zyx))
    hu_median = float(np.median(hu_zyx))

    # Check air voxels near -1000 HU
    air_mask = (hu_zyx >= -1050) & (hu_zyx <= -850)
    air_ratio = float(air_mask.mean())

    # Check whether FOV corners are mostly air
    z, y, x = hu_zyx.shape
    patch = max(8, min(32, y // 8, x // 8))

    corner_patches = [
        hu_zyx[:, :patch, :patch],
        hu_zyx[:, :patch, -patch:],
        hu_zyx[:, -patch:, :patch],
        hu_zyx[:, -patch:, -patch:],
    ]

    corner_values = np.concatenate([c.reshape(-1) for c in corner_patches])
    corner_median = float(np.median(corner_values))

    # Strict HU QC rules
    # If RescaleSlope/Intercept was not applied, air may appear near zero.
    if hu_min > -500:
        raise ValueError(
            f"[{case_id}] HU QC failed: min HU too high ({hu_min:.1f}). "
            "Possible RescaleSlope/Intercept conversion failure."
        )

    # Chest CT should generally contain air near -1000 HU or air-like corner values.
    if air_ratio < 0.001 and not (-1100 <= corner_median <= -700):
        raise ValueError(
            f"[{case_id}] HU QC failed: air near -1000 not found. "
            f"air_ratio={air_ratio:.6f}, corner_median={corner_median:.1f}"
        )

    # Prevent cases with no visible bone/contrast-like intensities
    if hu_max < 100:
        raise ValueError(
            f"[{case_id}] HU QC failed: max HU too low ({hu_max:.1f}). "
            "CT bone/contrast-like intensity range is barely visible."
        )

    qc = {
        "hu_qc_pass": True,
        "hu_min": hu_min,
        "hu_max": hu_max,
        "hu_mean": hu_mean,
        "hu_median": hu_median,
        "finite_ratio": finite_ratio,
        "air_ratio_minus1050_to_minus850": air_ratio,
        "corner_median_hu": corner_median,
    }

    return qc


# ============================================================
# 5. TotalSegmentator: direct DICOM-folder input
# ============================================================

def make_temp_dicom_dir(sorted_files, case_id: str):
    """
    Create a temporary folder containing only the selected and sorted DICOM files for TotalSegmentator.
    Fall back to copying files if symlinks are unavailable.
    """
    tmp_case_dir = TMP_ROOT / case_id
    tmp_dicom_dir = tmp_case_dir / "dicom_selected"

    if tmp_case_dir.exists():
        shutil.rmtree(tmp_case_dir, ignore_errors=True)

    tmp_dicom_dir.mkdir(parents=True, exist_ok=True)

    for i, src in enumerate(sorted_files):
        src = Path(src)
        suffix = src.suffix if src.suffix else ".dcm"
        dst = tmp_dicom_dir / f"{i:05d}_{sanitize_name(src.stem)}{suffix}"

        try:
            os.symlink(str(src), str(dst))
        except Exception:
            shutil.copy2(str(src), str(dst))

    return tmp_case_dir, tmp_dicom_dir


def run_totalsegmentator_from_dicom(sorted_files, case_id: str):
    """
    Run TotalSegmentator directly on a DICOM folder.
    """
    tmp_case_dir, tmp_dicom_dir = make_temp_dicom_dir(sorted_files, case_id)

    tmp_seg = tmp_case_dir / "seg"
    tmp_seg.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(TOTALSEG_BIN),
        "-i", str(tmp_dicom_dir),
        "-o", str(tmp_seg),
        "-ta", "total",
        "--roi_subset", *TARGET_LABELS,
        "--device", "gpu",
    ]

    if USE_ROBUST_CROP:
        cmd.append("--robust_crop")

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=make_totalseg_env(GPU_ID),
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"[{case_id}] TotalSegmentator DICOM input failed.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"LOG:\n{proc.stdout}"
        )

    return tmp_case_dir, tmp_seg


# ============================================================
# 6. Geometry / mask alignment QC
# ============================================================

def same_geometry(a: sitk.Image, b: sitk.Image):
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=1e-5)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=1e-3)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-5)
    )


def geometry_qc_report(img: sitk.Image, ref_img: sitk.Image, prefix: str):
    """
    Report whether the mask geometry matches the CT reference image.
    geometry_equal=True only when size, spacing, origin, and direction all match.
    """
    img_size = img.GetSize()
    ref_size = ref_img.GetSize()

    img_spacing = np.array(img.GetSpacing(), dtype=np.float64)
    ref_spacing = np.array(ref_img.GetSpacing(), dtype=np.float64)

    img_origin = np.array(img.GetOrigin(), dtype=np.float64)
    ref_origin = np.array(ref_img.GetOrigin(), dtype=np.float64)

    img_direction = np.array(img.GetDirection(), dtype=np.float64)
    ref_direction = np.array(ref_img.GetDirection(), dtype=np.float64)

    size_equal = img_size == ref_size
    spacing_equal = bool(np.allclose(img_spacing, ref_spacing, atol=1e-5))
    origin_equal = bool(np.allclose(img_origin, ref_origin, atol=1e-3))
    direction_equal = bool(np.allclose(img_direction, ref_direction, atol=1e-5))

    geometry_equal = bool(size_equal and spacing_equal and origin_equal and direction_equal)

    return {
        f"{prefix}_size_equal": bool(size_equal),
        f"{prefix}_spacing_equal": bool(spacing_equal),
        f"{prefix}_origin_equal": bool(origin_equal),
        f"{prefix}_direction_equal": bool(direction_equal),
        f"{prefix}_geometry_equal": bool(geometry_equal),

        f"{prefix}_size": size_to_str(img_size),
        f"{prefix}_ref_size": size_to_str(ref_size),
        f"{prefix}_spacing": spacing_to_str(img.GetSpacing()),
        f"{prefix}_ref_spacing": spacing_to_str(ref_img.GetSpacing()),
        f"{prefix}_origin": origin_to_str(img.GetOrigin()),
        f"{prefix}_ref_origin": origin_to_str(ref_img.GetOrigin()),
        f"{prefix}_direction": direction_to_str(img.GetDirection()),
        f"{prefix}_ref_direction": direction_to_str(ref_img.GetDirection()),

        f"{prefix}_max_spacing_absdiff": float(np.max(np.abs(img_spacing - ref_spacing))),
        f"{prefix}_max_origin_absdiff": float(np.max(np.abs(img_origin - ref_origin))),
        f"{prefix}_max_direction_absdiff": float(np.max(np.abs(img_direction - ref_direction))),
    }


def align_mask_to_reference_with_qc(mask_img, ref_img, prefix: str):
    """
    Align the TotalSegmentator output mask to the CT reference geometry,
    and return geometry QC before and after alignment.
    """
    before_report = geometry_qc_report(mask_img, ref_img, f"{prefix}_before_align")

    if before_report[f"{prefix}_before_align_geometry_equal"]:
        after_img = mask_img
        aligned_needed = False
    else:
        after_img = sitk.Resample(
            mask_img,
            ref_img,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )
        aligned_needed = True

    after_report = geometry_qc_report(after_img, ref_img, f"{prefix}_after_align")

    report = {}
    report.update(before_report)
    report.update(after_report)
    report[f"{prefix}_aligned_to_reference"] = bool(aligned_needed)
    report[f"{prefix}_alignment_ok"] = bool(after_report[f"{prefix}_after_align_geometry_equal"])

    return after_img, report


def load_union_mask(seg_dir: Path, labels, ref_img: sitk.Image, case_id: str, mask_name: str):
    union = None

    existing_labels = []
    nonempty_labels = []
    empty_labels = []
    missing_labels = []
    label_voxel_counts = {}

    any_aligned = False
    all_after_alignment_ok = True

    for lab in labels:
        p = seg_dir / f"{lab}.nii.gz"
        if not p.exists():
            p = seg_dir / f"{lab}.nii"

        if not p.exists():
            missing_labels.append(lab)
            label_voxel_counts[lab] = 0
            continue

        existing_labels.append(lab)

        mask_img = sitk.ReadImage(str(p))
        mask_img = sitk.Cast(mask_img, sitk.sitkUInt8)

        mask_img, report = align_mask_to_reference_with_qc(
            mask_img,
            ref_img,
            prefix=f"{mask_name}_{lab}",
        )

        any_aligned = any_aligned or bool(report[f"{mask_name}_{lab}_aligned_to_reference"])
        all_after_alignment_ok = (
            all_after_alignment_ok
            and bool(report[f"{mask_name}_{lab}_alignment_ok"])
        )

        arr = sitk.GetArrayFromImage(mask_img) > 0
        voxel_count = int(arr.sum())
        label_voxel_counts[lab] = voxel_count

        if union is None:
            union = np.zeros(arr.shape, dtype=bool)

        if voxel_count > 0:
            union |= arr
            nonempty_labels.append(lab)
        else:
            empty_labels.append(lab)

    if union is None or union.sum() == 0:
        raise RuntimeError(f"[{case_id}] No valid {mask_name} mask loaded from TotalSegmentator.")

    union_uint8 = union.astype(np.uint8)

    out_img = sitk.GetImageFromArray(union_uint8)
    out_img.CopyInformation(ref_img)

    union_report = geometry_qc_report(out_img, ref_img, f"{mask_name}_union")

    union_report[f"{mask_name}_requested_label_count"] = int(len(labels))
    union_report[f"{mask_name}_existing_label_count"] = int(len(existing_labels))
    union_report[f"{mask_name}_nonempty_label_count"] = int(len(nonempty_labels))
    union_report[f"{mask_name}_empty_label_count"] = int(len(empty_labels))
    union_report[f"{mask_name}_missing_label_count"] = int(len(missing_labels))

    union_report[f"{mask_name}_existing_labels"] = "|".join(existing_labels)
    union_report[f"{mask_name}_nonempty_labels"] = "|".join(nonempty_labels)
    union_report[f"{mask_name}_empty_labels"] = "|".join(empty_labels)
    union_report[f"{mask_name}_missing_labels"] = "|".join(missing_labels)

    union_report[f"{mask_name}_label_voxel_counts_json"] = json.dumps(
        label_voxel_counts,
        ensure_ascii=False,
    )

    # Backward-compatible columns
    union_report[f"{mask_name}_loaded_label_count"] = int(len(nonempty_labels))
    union_report[f"{mask_name}_loaded_labels"] = "|".join(nonempty_labels)

    union_report[f"{mask_name}_any_label_aligned_to_reference"] = bool(any_aligned)
    union_report[f"{mask_name}_all_labels_alignment_ok"] = bool(all_after_alignment_ok)
    union_report[f"{mask_name}_union_voxel_count"] = int(union_uint8.sum())

    return out_img, union_uint8, nonempty_labels, union_report


# ============================================================
# 7. Index extraction, cropping, and resampling
# ============================================================

def get_z_range_from_mask(mask_zyx: np.ndarray, case_id: str, name: str):
    z_has = mask_zyx.any(axis=(1, 2))
    z_idx = np.where(z_has)[0]

    if len(z_idx) == 0:
        raise RuntimeError(f"[{case_id}] Empty {name} z-index.")

    return int(z_idx.min()), int(z_idx.max())


def add_margin_to_z_range(z_min, z_max, z_size, z_spacing, margin_mm):
    margin_slices = int(np.ceil(float(margin_mm) / float(z_spacing)))

    z0 = max(0, int(z_min) - margin_slices)
    z1 = min(z_size - 1, int(z_max) + margin_slices)

    return z0, z1, margin_slices


def crop_sitk_by_z(img: sitk.Image, z0: int, z1: int):
    """
    SimpleITK image crop.
    SimpleITK size/index order is [X, Y, Z]
    """
    size_x, size_y, size_z = img.GetSize()
    crop_z = int(z1 - z0 + 1)

    roi_size = [size_x, size_y, crop_z]
    roi_index = [0, 0, int(z0)]

    return sitk.RegionOfInterest(img, roi_size, roi_index)


def resample_to_z_spacing(img: sitk.Image, target_z_spacing: float, is_mask: bool):
    """
    Keep X/Y spacing and standardize only the z-spacing.
    """
    old_spacing = np.array(img.GetSpacing(), dtype=np.float64)  # [x,y,z]
    old_size = np.array(img.GetSize(), dtype=np.int64)          # [x,y,z]

    new_spacing = old_spacing.copy()
    new_spacing[2] = float(target_z_spacing)

    new_size = np.round(old_size * old_spacing / new_spacing).astype(np.int64)
    new_size = np.maximum(new_size, 1)

    interpolator = sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
    out_pixel_type = sitk.sitkUInt8 if is_mask else sitk.sitkFloat32

    resampled = sitk.Resample(
        img,
        [int(x) for x in new_size],
        sitk.Transform(),
        interpolator,
        img.GetOrigin(),
        [float(x) for x in new_spacing],
        img.GetDirection(),
        0,
        out_pixel_type,
    )

    return resampled


# ============================================================
# 8. Windowing and H5 export
# ============================================================

def hu_float_to_int16(hu: np.ndarray):
    """
    Prepare HU values for H5 storage.

    Important:
    - HU QC is already performed on raw HU float32 data in verify_hu_or_die().
    - This step only cleans outside-FOV or padding values for the stored image_hu dataset.
    - Only values below -1024 are raised to -1024.
    - Maximum HU values are not clinically clipped.
    """
    hu = hu.astype(np.float32)

    # Protect against NaN/inf values
    hu = np.nan_to_num(
        hu,
        nan=HU_STORAGE_MIN,
        posinf=32767.0,
        neginf=HU_STORAGE_MIN,
    )

    # Key step: clean only low-end padding values
    hu = np.maximum(hu, HU_STORAGE_MIN)

    # Keep values within the safe int16 storage range
    hu = np.clip(hu, -32768, 32767)

    return np.rint(hu).astype(np.int16)


def apply_window_uint8(hu_int16: np.ndarray, wl: float, ww: float):
    hu = hu_int16.astype(np.float32)

    low = wl - ww / 2.0
    high = wl + ww / 2.0

    x = np.clip(hu, low, high)
    x = (x - low) / (high - low)
    x = np.clip(x, 0, 1)

    return np.rint(x * 255).astype(np.uint8)


def create_dataset_compressed(h5, key, data):
    h5.create_dataset(
        key,
        data=data,
        compression=H5_COMPRESSION,
        compression_opts=H5_COMPRESSION_OPTS,
        shuffle=True,
    )


def write_string_array(h5, key, values):
    dt = h5py.string_dtype(encoding="utf-8")
    arr = np.array([str(v) for v in values], dtype=dt)
    h5.create_dataset(key, data=arr)


def write_crop_h5(
    h5_path: Path,
    crop_type: str,
    case_id: str,
    hu_img_resampled: sitk.Image,
    mask_img_resampled: sitk.Image,
    original_z_range,
    original_crop_z_range,
    selected_dicom_files,
    slice_positions_original,
    instance_numbers_original,
    original_spacing_zyx,
    hu_qc,
    attrs_extra,
):
    """
    crop_type:
    - "lung": save mask_lung only
    - "rib" : save mask_rib only
    """
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    hu_float = sitk.GetArrayFromImage(hu_img_resampled).astype(np.float32)  # [Z,Y,X]
    mask = sitk.GetArrayFromImage(mask_img_resampled).astype(np.uint8)
    mask = (mask > 0).astype(np.uint8)

    hu_int16 = hu_float_to_int16(hu_float)

    resampled_spacing_xyz = hu_img_resampled.GetSpacing()
    resampled_spacing_zyx = np.array(
        [resampled_spacing_xyz[2], resampled_spacing_xyz[1], resampled_spacing_xyz[0]],
        dtype=np.float32,
    )

    # index based on the resampled mask
    resampled_z_range = get_z_range_from_mask(mask, case_id, f"resampled_{crop_type}")
    resampled_crop_z_range = np.array([0, hu_int16.shape[0] - 1], dtype=np.int32)

    mask_z_has = mask.any(axis=(1, 2))
    mask_slice_count = int(mask_z_has.sum())
    mask_voxel_count = int(mask.sum())

    with h5py.File(h5_path, "w") as f:
        # main image
        create_dataset_compressed(f, "image_hu", hu_int16)

        # Save four windowed images
        for win_name, cfg in WINDOWS.items():
            win = apply_window_uint8(hu_int16, cfg["WL"], cfg["WW"])
            create_dataset_compressed(f, f"window_{win_name}", win)

        # Save one mask per crop type
        if crop_type == "lung":
            create_dataset_compressed(f, "mask_lung", mask)
        elif crop_type == "rib":
            create_dataset_compressed(f, "mask_rib", mask)
        else:
            raise ValueError(f"Unknown crop_type: {crop_type}")

        # index
        f.create_dataset("original_z_range", data=np.array(original_z_range, dtype=np.int32))
        f.create_dataset("original_crop_z_range", data=np.array(original_crop_z_range, dtype=np.int32))
        f.create_dataset("resampled_z_range", data=np.array(resampled_z_range, dtype=np.int32))
        f.create_dataset("resampled_crop_z_range", data=resampled_crop_z_range)

        # spacing
        f.create_dataset("original_spacing_zyx", data=np.array(original_spacing_zyx, dtype=np.float32))
        f.create_dataset("resampled_spacing_zyx", data=resampled_spacing_zyx)

        # dicom trace
        f.create_dataset("slice_position_original", data=np.array(slice_positions_original, dtype=np.float32))
        f.create_dataset("instance_number_original", data=np.array(instance_numbers_original, dtype=np.float32))
        write_string_array(f, "selected_dicom_files", [str(p) for p in selected_dicom_files])

        # attrs
        f.attrs["case_id"] = str(case_id)
        f.attrs["crop_type"] = str(crop_type)
        f.attrs["axis_order"] = "ZYX"
        f.attrs["hu_dtype"] = "int16"
        f.attrs["window_dtype"] = "uint8"
        f.attrs["mask_dtype"] = "uint8"
        f.attrs["target_z_spacing"] = float(TARGET_Z_SPACING)
        f.attrs["crop_margin_mm"] = float(CROP_MARGIN_MM)
        f.attrs["preprocessing_version"] = "dicom_to_lung_rib_h5_v2_dicom_totalseg"
        f.attrs["windows_json"] = json.dumps(WINDOWS, ensure_ascii=False)

        for k, v in hu_qc.items():
            f.attrs[k] = v

        for k, v in attrs_extra.items():
            if v is None:
                continue
            f.attrs[k] = str(v)

    return {
        "h5_path": str(h5_path),
        "shape_zyx": tuple(hu_int16.shape),
        "z_count": int(hu_int16.shape[0]),
        "y_count": int(hu_int16.shape[1]),
        "x_count": int(hu_int16.shape[2]),
        "resampled_z_range": tuple(resampled_z_range),
        "resampled_crop_z_range": tuple(resampled_crop_z_range.tolist()),
        "resampled_spacing_zyx": tuple(resampled_spacing_zyx.tolist()),
        "mask_slice_count": mask_slice_count,
        "mask_voxel_count": mask_voxel_count,
    }


# ============================================================
# 9. PNG QC export
# ============================================================

def overlay_mask_on_gray(gray_uint8, mask_uint8):
    """
    Overlay the mask in red on top of a grayscale image.
    """
    rgb = np.stack([gray_uint8, gray_uint8, gray_uint8], axis=-1).astype(np.uint8)

    mask = mask_uint8 > 0
    rgb[mask, 0] = 255
    rgb[mask, 1] = (rgb[mask, 1] * 0.35).astype(np.uint8)
    rgb[mask, 2] = (rgb[mask, 2] * 0.35).astype(np.uint8)

    return rgb


def save_qc_png(h5_path: Path, case_id: str, crop_type: str):
    out_dir = QC_PNG_DIR / case_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        if crop_type == "lung":
            gray = f["window_lung"][:]
            mask = f["mask_lung"][:]
        else:
            gray = f["window_bone"][:]
            mask = f["mask_rib"][:]

        z_range = f["resampled_z_range"][:]
        z_mid = int(round((int(z_range[0]) + int(z_range[1])) / 2))

        img = gray[z_mid]
        m = mask[z_mid]

        overlay = overlay_mask_on_gray(img, m)

        out_path = out_dir / f"{case_id}_{crop_type}_mid_overlay_z{z_mid}.png"
        Image.fromarray(overlay).save(out_path)

    return str(out_path)


# ============================================================
# 10. Error categorization
# ============================================================

def classify_error_message(msg: str):
    msg = str(msg)

    if "HU QC failed" in msg or "HU dtype" in msg or "non-finite" in msg:
        return "HU_QC"
    if "TotalSegmentator" in msg:
        return "TOTALSEG"
    if "No valid lung mask" in msg or "Empty lung" in msg:
        return "LUNG_MASK"
    if "No valid rib mask" in msg or "Empty rib" in msg:
        return "RIB_MASK"
    if "DICOM" in msg or "No DICOM" in msg or "ImagePositionPatient" in msg:
        return "DICOM_READ_OR_SORT"
    if "H5" in msg or ".h5" in msg:
        return "H5_WRITE"
    return "OTHER"


# ============================================================
# 11. Single-case processing
# ============================================================

def process_one_case(row, qc_png_save=False):
    case_id = make_case_id(row)

    case_folder = row.get("CaseFolder", None)
    series_uid = row.get("SeriesInstanceUID", None)

    chest_dir = Path(str(row["ChestDir"]))
    file_start = row.get("FileStart", None)
    file_end = row.get("FileEnd", None)

    result = {
        "case_id": case_id,
        "chest_dir": str(chest_dir),
        "file_start": str(file_start),
        "file_end": str(file_end),
        "status": "FAILED",
        "failed_stage": "",
        "qc_error_category": "",
        "error_message": "",
    }

    tmp_case_dir = None
    stage = "init"

    try:
        # ------------------------------------------------------------
        # 1) DICOM selection and slice-position sorting
        # ------------------------------------------------------------
        stage = "dicom_select_and_sort"

        selected_files, range_selection_status, total_file_count = resolve_file_range(
            chest_dir, file_start, file_end
        )
        sorted_files, slice_positions, instance_numbers = sort_by_slice_position(selected_files)

        slice_diffs = np.diff(slice_positions) if len(slice_positions) >= 2 else np.array([])

        slice_position_increasing = bool(np.all(slice_diffs > 0)) if len(slice_diffs) > 0 else True
        slice_position_monotonic = bool(np.all(slice_diffs != 0)) if len(slice_diffs) > 0 else True

        result["total_file_count_in_chest_dir"] = int(total_file_count)
        result["selected_file_count"] = int(len(sorted_files))
        result["range_selection_status"] = range_selection_status
        result["slice_position_min"] = float(np.min(slice_positions))
        result["slice_position_max"] = float(np.max(slice_positions))
        result["slice_position_increasing_after_sort"] = slice_position_increasing
        result["slice_position_monotonic_after_sort"] = slice_position_monotonic
        result["slice_position_median_diff"] = float(np.median(slice_diffs)) if len(slice_diffs) > 0 else np.nan
        result["slice_position_min_diff"] = float(np.min(slice_diffs)) if len(slice_diffs) > 0 else np.nan
        result["slice_position_max_diff"] = float(np.max(slice_diffs)) if len(slice_diffs) > 0 else np.nan

        # ------------------------------------------------------------
        # 2) DICOM to HU float32 and LPS orientation standardization
        # ------------------------------------------------------------
        stage = "dicom_to_hu_and_orientation_lps"

        ct_img_lps, hu_zyx, orientation_changed, orientation_report = read_dicom_series_as_sitk(
            sorted_files
        )

        for k, v in orientation_report.items():
            result[k] = v

        original_spacing_xyz = ct_img_lps.GetSpacing()  # [x,y,z]
        original_spacing_zyx = [
            float(original_spacing_xyz[2]),
            float(original_spacing_xyz[1]),
            float(original_spacing_xyz[0]),
        ]

        result["original_z_slices"] = int(hu_zyx.shape[0])
        result["original_y_size"] = int(hu_zyx.shape[1])
        result["original_x_size"] = int(hu_zyx.shape[2])
        result["original_shape_zyx"] = str(tuple(hu_zyx.shape))
        result["original_spacing_z"] = float(original_spacing_zyx[0])
        result["original_spacing_y"] = float(original_spacing_zyx[1])
        result["original_spacing_x"] = float(original_spacing_zyx[2])
        result["original_spacing_zyx"] = str(tuple(original_spacing_zyx))

        # ------------------------------------------------------------
        # 3) HU QC
        # ------------------------------------------------------------
        stage = "hu_qc"

        hu_qc = verify_hu_or_die(hu_zyx, case_id)

        for k, v in hu_qc.items():
            result[k] = v

        # ------------------------------------------------------------
        # 4) Run TotalSegmentator using direct DICOM-folder input
        # ------------------------------------------------------------
        stage = "totalsegmentator_dicom_input"

        tmp_case_dir, seg_dir = run_totalsegmentator_from_dicom(sorted_files, case_id)

        # ------------------------------------------------------------
        # 5) lung/rib mask load + geometry alignment QC
        # ------------------------------------------------------------
        stage = "load_lung_mask"

        lung_mask_img, lung_mask_zyx, loaded_lung_labels, lung_mask_qc = load_union_mask(
            seg_dir, LUNG_LABELS, ct_img_lps, case_id, "lung"
        )

        for k, v in lung_mask_qc.items():
            result[k] = v

        stage = "load_rib_mask"

        rib_mask_img, rib_mask_zyx, loaded_rib_labels, rib_mask_qc = load_union_mask(
            seg_dir, RIB_LABELS, ct_img_lps, case_id, "rib"
        )

        for k, v in rib_mask_qc.items():
            result[k] = v

        # Summarize final mask-to-CT geometry alignment
        result["lung_mask_ct_geometry_match_final"] = bool(lung_mask_qc["lung_union_geometry_equal"])
        result["rib_mask_ct_geometry_match_final"] = bool(rib_mask_qc["rib_union_geometry_equal"])
        result["all_mask_ct_geometry_match_final"] = bool(
            result["lung_mask_ct_geometry_match_final"]
            and result["rib_mask_ct_geometry_match_final"]
        )

        # ------------------------------------------------------------
        # 6) Extract original z-index range
        # ------------------------------------------------------------
        stage = "extract_original_indices"

        lung_z0, lung_z1 = get_z_range_from_mask(lung_mask_zyx, case_id, "lung")
        rib_z0, rib_z1 = get_z_range_from_mask(rib_mask_zyx, case_id, "rib")

        z_size = hu_zyx.shape[0]
        z_spacing = original_spacing_zyx[0]

        lung_crop_z0, lung_crop_z1, lung_margin_slices = add_margin_to_z_range(
            lung_z0, lung_z1, z_size, z_spacing, CROP_MARGIN_MM
        )
        rib_crop_z0, rib_crop_z1, rib_margin_slices = add_margin_to_z_range(
            rib_z0, rib_z1, z_size, z_spacing, CROP_MARGIN_MM
        )

        result["crop_margin_mm"] = float(CROP_MARGIN_MM)
        result["lung_margin_slices"] = int(lung_margin_slices)
        result["rib_margin_slices"] = int(rib_margin_slices)

        result["lung_original_z_min"] = int(lung_z0)
        result["lung_original_z_max"] = int(lung_z1)
        result["lung_original_z_count"] = int(lung_z1 - lung_z0 + 1)
        result["lung_original_crop_z_min"] = int(lung_crop_z0)
        result["lung_original_crop_z_max"] = int(lung_crop_z1)
        result["lung_original_crop_z_count"] = int(lung_crop_z1 - lung_crop_z0 + 1)

        result["rib_original_z_min"] = int(rib_z0)
        result["rib_original_z_max"] = int(rib_z1)
        result["rib_original_z_count"] = int(rib_z1 - rib_z0 + 1)
        result["rib_original_crop_z_min"] = int(rib_crop_z0)
        result["rib_original_crop_z_max"] = int(rib_crop_z1)
        result["rib_original_crop_z_count"] = int(rib_crop_z1 - rib_crop_z0 + 1)

        result["lung_original_z_range"] = str((lung_z0, lung_z1))
        result["rib_original_z_range"] = str((rib_z0, rib_z1))
        result["lung_original_crop_z_range"] = str((lung_crop_z0, lung_crop_z1))
        result["rib_original_crop_z_range"] = str((rib_crop_z0, rib_crop_z1))

        # ------------------------------------------------------------
        # 7) lung crop -> 2.5-mm z-resampling -> lung H5 export
        # ------------------------------------------------------------
        stage = "write_lung_h5"

        lung_ct_crop = crop_sitk_by_z(ct_img_lps, lung_crop_z0, lung_crop_z1)
        lung_mask_crop = crop_sitk_by_z(lung_mask_img, lung_crop_z0, lung_crop_z1)

        lung_ct_rs = resample_to_z_spacing(
            lung_ct_crop,
            TARGET_Z_SPACING,
            is_mask=False,
        )
        lung_mask_rs = resample_to_z_spacing(
            lung_mask_crop,
            TARGET_Z_SPACING,
            is_mask=True,
        )

        lung_h5_path = LUNG_H5_DIR / f"{case_id}_lung.h5"

        attrs_extra = {
            "case_folder": case_folder,
            "series_instance_uid": series_uid,
            "chest_dir": chest_dir,
            "file_start": file_start,
            "file_end": file_end,
            "orientation_changed_to_LPS": orientation_changed,
            "totalseg_input_type": "DICOM_FOLDER",
        }

        lung_h5_info = write_crop_h5(
            h5_path=lung_h5_path,
            crop_type="lung",
            case_id=case_id,
            hu_img_resampled=lung_ct_rs,
            mask_img_resampled=lung_mask_rs,
            original_z_range=(lung_z0, lung_z1),
            original_crop_z_range=(lung_crop_z0, lung_crop_z1),
            selected_dicom_files=sorted_files,
            slice_positions_original=slice_positions,
            instance_numbers_original=instance_numbers,
            original_spacing_zyx=original_spacing_zyx,
            hu_qc=hu_qc,
            attrs_extra=attrs_extra,
        )

        result["lung_h5_path"] = lung_h5_info["h5_path"]
        result["lung_h5_z_slices"] = lung_h5_info["z_count"]
        result["lung_h5_y_size"] = lung_h5_info["y_count"]
        result["lung_h5_x_size"] = lung_h5_info["x_count"]
        result["lung_h5_shape_zyx"] = str(lung_h5_info["shape_zyx"])

        result["lung_resampled_z_min"] = int(lung_h5_info["resampled_z_range"][0])
        result["lung_resampled_z_max"] = int(lung_h5_info["resampled_z_range"][1])
        result["lung_resampled_z_count"] = int(
            lung_h5_info["resampled_z_range"][1]
            - lung_h5_info["resampled_z_range"][0]
            + 1
        )

        result["lung_resampled_crop_z_min"] = 0
        result["lung_resampled_crop_z_max"] = int(lung_h5_info["z_count"] - 1)
        result["lung_resampled_crop_z_count"] = int(lung_h5_info["z_count"])

        result["lung_mask_resampled_slice_count"] = int(lung_h5_info["mask_slice_count"])
        result["lung_mask_resampled_voxel_count"] = int(lung_h5_info["mask_voxel_count"])
        result["lung_resampled_spacing_zyx"] = str(lung_h5_info["resampled_spacing_zyx"])

        # ------------------------------------------------------------
        # 8) rib crop -> 2.5-mm z-resampling -> rib H5 export
        # ------------------------------------------------------------
        stage = "write_rib_h5"

        rib_ct_crop = crop_sitk_by_z(ct_img_lps, rib_crop_z0, rib_crop_z1)
        rib_mask_crop = crop_sitk_by_z(rib_mask_img, rib_crop_z0, rib_crop_z1)

        rib_ct_rs = resample_to_z_spacing(
            rib_ct_crop,
            TARGET_Z_SPACING,
            is_mask=False,
        )
        rib_mask_rs = resample_to_z_spacing(
            rib_mask_crop,
            TARGET_Z_SPACING,
            is_mask=True,
        )

        rib_h5_path = RIB_H5_DIR / f"{case_id}_rib.h5"

        rib_h5_info = write_crop_h5(
            h5_path=rib_h5_path,
            crop_type="rib",
            case_id=case_id,
            hu_img_resampled=rib_ct_rs,
            mask_img_resampled=rib_mask_rs,
            original_z_range=(rib_z0, rib_z1),
            original_crop_z_range=(rib_crop_z0, rib_crop_z1),
            selected_dicom_files=sorted_files,
            slice_positions_original=slice_positions,
            instance_numbers_original=instance_numbers,
            original_spacing_zyx=original_spacing_zyx,
            hu_qc=hu_qc,
            attrs_extra=attrs_extra,
        )

        result["rib_h5_path"] = rib_h5_info["h5_path"]
        result["rib_h5_z_slices"] = rib_h5_info["z_count"]
        result["rib_h5_y_size"] = rib_h5_info["y_count"]
        result["rib_h5_x_size"] = rib_h5_info["x_count"]
        result["rib_h5_shape_zyx"] = str(rib_h5_info["shape_zyx"])

        result["rib_resampled_z_min"] = int(rib_h5_info["resampled_z_range"][0])
        result["rib_resampled_z_max"] = int(rib_h5_info["resampled_z_range"][1])
        result["rib_resampled_z_count"] = int(
            rib_h5_info["resampled_z_range"][1]
            - rib_h5_info["resampled_z_range"][0]
            + 1
        )

        result["rib_resampled_crop_z_min"] = 0
        result["rib_resampled_crop_z_max"] = int(rib_h5_info["z_count"] - 1)
        result["rib_resampled_crop_z_count"] = int(rib_h5_info["z_count"])

        result["rib_mask_resampled_slice_count"] = int(rib_h5_info["mask_slice_count"])
        result["rib_mask_resampled_voxel_count"] = int(rib_h5_info["mask_voxel_count"])
        result["rib_resampled_spacing_zyx"] = str(rib_h5_info["resampled_spacing_zyx"])

        # ------------------------------------------------------------
        # 9) QC PNG export
        # ------------------------------------------------------------
        stage = "save_qc_png"

        if qc_png_save:
            lung_png = save_qc_png(lung_h5_path, case_id, "lung")
            rib_png = save_qc_png(rib_h5_path, case_id, "rib")
            result["qc_png_lung"] = lung_png
            result["qc_png_rib"] = rib_png

        result["status"] = "OK"
        result["failed_stage"] = ""
        result["qc_error_category"] = ""

    except Exception as e:
        result["status"] = "FAILED"
        result["failed_stage"] = stage
        result["error_message"] = str(e)
        result["qc_error_category"] = classify_error_message(str(e))

    finally:
        if tmp_case_dir is not None and Path(tmp_case_dir).exists():
            shutil.rmtree(tmp_case_dir, ignore_errors=True)

    return result


# ============================================================
# 12. CSV export
# ============================================================

def save_report_csvs(summary: pd.DataFrame):
    """
    Export task-specific CSV files from the complete summary.
    """
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    # 1) Failures and QC errors only
    if "status" in summary.columns:
        error_df = summary[summary["status"] != "OK"].copy()
    else:
        error_df = pd.DataFrame()

    error_df.to_csv(QC_ERROR_CSV, index=False, encoding="utf-8-sig")

    # 2) H5 index summary: two long-format rows per case for lung and rib crops
    h5_rows = []

    for _, r in summary.iterrows():
        case_id = r.get("case_id", "")

        for crop_type in ["lung", "rib"]:
            row = {
                "case_id": case_id,
                "crop_type": crop_type,
                "status": r.get("status", ""),
                "h5_path": r.get(f"{crop_type}_h5_path", ""),
                "h5_z_slices": r.get(f"{crop_type}_h5_z_slices", np.nan),
                "h5_y_size": r.get(f"{crop_type}_h5_y_size", np.nan),
                "h5_x_size": r.get(f"{crop_type}_h5_x_size", np.nan),
                "h5_shape_zyx": r.get(f"{crop_type}_h5_shape_zyx", ""),

                "original_z_min": r.get(f"{crop_type}_original_z_min", np.nan),
                "original_z_max": r.get(f"{crop_type}_original_z_max", np.nan),
                "original_z_count": r.get(f"{crop_type}_original_z_count", np.nan),

                "original_crop_z_min": r.get(f"{crop_type}_original_crop_z_min", np.nan),
                "original_crop_z_max": r.get(f"{crop_type}_original_crop_z_max", np.nan),
                "original_crop_z_count": r.get(f"{crop_type}_original_crop_z_count", np.nan),

                "resampled_z_min": r.get(f"{crop_type}_resampled_z_min", np.nan),
                "resampled_z_max": r.get(f"{crop_type}_resampled_z_max", np.nan),
                "resampled_z_count": r.get(f"{crop_type}_resampled_z_count", np.nan),

                "resampled_crop_z_min": r.get(f"{crop_type}_resampled_crop_z_min", np.nan),
                "resampled_crop_z_max": r.get(f"{crop_type}_resampled_crop_z_max", np.nan),
                "resampled_crop_z_count": r.get(f"{crop_type}_resampled_crop_z_count", np.nan),

                "mask_resampled_slice_count": r.get(f"{crop_type}_mask_resampled_slice_count", np.nan),
                "mask_resampled_voxel_count": r.get(f"{crop_type}_mask_resampled_voxel_count", np.nan),

                "original_spacing_z": r.get("original_spacing_z", np.nan),
                "target_z_spacing": TARGET_Z_SPACING,
                "resampled_spacing_zyx": r.get(f"{crop_type}_resampled_spacing_zyx", ""),

                "crop_margin_mm": r.get("crop_margin_mm", np.nan),
                "margin_slices": r.get(f"{crop_type}_margin_slices", np.nan),

                "failed_stage": r.get("failed_stage", ""),
                "qc_error_category": r.get("qc_error_category", ""),
                "error_message": r.get("error_message", ""),
            }
            h5_rows.append(row)

    h5_index_df = pd.DataFrame(h5_rows)
    h5_index_df.to_csv(H5_INDEX_CSV, index=False, encoding="utf-8-sig")

    # 3) orientation / geometry QC
    orientation_cols = [
        "case_id",
        "status",
        "failed_stage",
        "qc_error_category",
        "error_message",

        "orientation_before_code",
        "orientation_after_code",
        "orientation_lps_ok",
        "orientation_changed_to_LPS",
        "h5_axis_order",

        "sitk_size_before_xyz",
        "sitk_size_after_xyz",
        "sitk_spacing_before_xyz",
        "sitk_spacing_after_xyz",
        "sitk_origin_before_xyz",
        "sitk_origin_after_xyz",
        "sitk_direction_before",
        "sitk_direction_after",

        "slice_position_increasing_after_sort",
        "slice_position_monotonic_after_sort",
        "slice_position_median_diff",
        "slice_position_min_diff",
        "slice_position_max_diff",

        "lung_mask_ct_geometry_match_final",
        "rib_mask_ct_geometry_match_final",
        "all_mask_ct_geometry_match_final",

        # ============================================================
        # lung mask geometry QC
        # ============================================================
        "lung_union_geometry_equal",
        "lung_union_size_equal",
        "lung_union_spacing_equal",
        "lung_union_origin_equal",
        "lung_union_direction_equal",
        "lung_any_label_aligned_to_reference",
        "lung_all_labels_alignment_ok",

        # ============================================================
        # lung Per-label existence flag and voxel count
        # ============================================================
        "lung_requested_label_count",
        "lung_existing_label_count",
        "lung_nonempty_label_count",
        "lung_empty_label_count",
        "lung_missing_label_count",
        "lung_existing_labels",
        "lung_nonempty_labels",
        "lung_empty_labels",
        "lung_missing_labels",
        "lung_label_voxel_counts_json",

        # Backward-compatible columns
        "lung_loaded_label_count",
        "lung_loaded_labels",
        "lung_union_voxel_count",

        # ============================================================
        # rib mask geometry QC
        # ============================================================
        "rib_union_geometry_equal",
        "rib_union_size_equal",
        "rib_union_spacing_equal",
        "rib_union_origin_equal",
        "rib_union_direction_equal",
        "rib_any_label_aligned_to_reference",
        "rib_all_labels_alignment_ok",

        # ============================================================
        # rib Per-label existence flag and voxel count
        # ============================================================
        "rib_requested_label_count",
        "rib_existing_label_count",
        "rib_nonempty_label_count",
        "rib_empty_label_count",
        "rib_missing_label_count",
        "rib_existing_labels",
        "rib_nonempty_labels",
        "rib_empty_labels",
        "rib_missing_labels",
        "rib_label_voxel_counts_json",

        # Backward-compatible columns
        "rib_loaded_label_count",
        "rib_loaded_labels",
        "rib_union_voxel_count",

        # ============================================================
        # HU QC
        # ============================================================
        "hu_qc_pass",
        "hu_min",
        "hu_max",
        "hu_mean",
        "hu_median",
        "finite_ratio",
        "air_ratio_minus1050_to_minus850",
        "corner_median_hu",
    ]

    existing_cols = [c for c in orientation_cols if c in summary.columns]
    orientation_df = summary[existing_cols].copy()
    orientation_df.to_csv(ORIENTATION_QC_CSV, index=False, encoding="utf-8-sig")


# ============================================================
# 13. Main
# ============================================================

def main():
    ensure_dirs()

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    if not TOTALSEG_BIN.exists():
        raise FileNotFoundError(f"TotalSegmentator not found: {TOTALSEG_BIN}")

    if not TOTALSEG_PY.exists():
        raise FileNotFoundError(f"TotalSegmentator python not found: {TOTALSEG_PY}")

    gpu_healthcheck_or_die()

    df = pd.read_csv(CSV_PATH)

    required_cols = ["ChestDir", "FileStart", "FileEnd"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Required column missing in CSV: {c}")

    # ============================================================
    # 1) Read existing summary
    # ============================================================
    results = []

    if SUMMARY_CSV.exists():
        try:
            prev_summary = pd.read_csv(SUMMARY_CSV)
            results = prev_summary.to_dict("records")
            print("=" * 80)
            print("[RESUME] Previous summary loaded")
            print("[RESUME] SUMMARY_CSV:", SUMMARY_CSV)
            print("[RESUME] Previous rows:", len(results))
            if "status" in prev_summary.columns:
                print("[RESUME] Previous OK:", int((prev_summary["status"] == "OK").sum()))
                print("[RESUME] Previous FAILED:", int((prev_summary["status"] == "FAILED").sum()))
            print("=" * 80)
        except Exception as e:
            print("=" * 80)
            print("[RESUME] Previous summary exists but could not be read.")
            print("[RESUME] Error:", e)
            print("[RESUME] Start with empty results.")
            print("[RESUME] Existing H5 files without a summary will be reprocessed to verify label-level QC.")
            print("=" * 80)
            results = []

    # case_id -> result dict
    result_map = {}
    for r in results:
        cid = str(r.get("case_id", ""))
        if cid:
            result_map[cid] = r

    # OK cases according to the existing summary
    summary_ok_case_ids = set()
    for cid, r in result_map.items():
        if str(r.get("status", "")) == "OK":
            summary_ok_case_ids.add(cid)

    print("[RESUME] OK cases in summary:", len(summary_ok_case_ids))

    # ============================================================
    # 2) Main loop: skip cases with OK summary and both H5 outputs
    # ============================================================
    processed_now = 0
    skipped_ok = 0
    reprocessed_partial = 0
    reprocessed_missing_summary = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="DICOM -> lung/rib H5 RESUME"):
        case_id = make_case_id(row)

        lung_h5_path = LUNG_H5_DIR / f"{case_id}_lung.h5"
        rib_h5_path = RIB_H5_DIR / f"{case_id}_rib.h5"

        lung_exists = lung_h5_path.exists()
        rib_exists = rib_h5_path.exists()

        # ------------------------------------------------------------
        # A. Skip if the summary marks the case as OK and both H5 files exist
        # ------------------------------------------------------------
        if case_id in summary_ok_case_ids and lung_exists and rib_exists:
            skipped_ok += 1
            continue

        # ------------------------------------------------------------
        # B. If only one of lung/rib H5 files exists, treat as partial output and reprocess after deletion
        # ------------------------------------------------------------
        if lung_exists != rib_exists:
            print(f"\n[RESUME] Partial H5 detected. Delete and reprocess: {case_id}")
            print(f"  lung_exists={lung_exists}, rib_exists={rib_exists}")

            if lung_h5_path.exists():
                lung_h5_path.unlink()
            if rib_h5_path.exists():
                rib_h5_path.unlink()

            reprocessed_partial += 1

        # ------------------------------------------------------------
        # C. Reprocess if both H5 files exist but the summary has no OK record
        #    Reason: preserve label-level QC fields such as rib_nonempty_labels and missing_labels in CSV outputs
        # ------------------------------------------------------------
        elif lung_exists and rib_exists and case_id not in summary_ok_case_ids:
            print(f"\n[RESUME] H5 exists but summary OK row missing. Reprocess for label QC: {case_id}")
            reprocessed_missing_summary += 1

            # Reprocessing overwrites the same output path using write mode.
            # Delete existing files before reprocessing to avoid partial-file issues.
            lung_h5_path.unlink()
            rib_h5_path.unlink()

        # ------------------------------------------------------------
        # D. Process
        # ------------------------------------------------------------
        qc_png_save = idx < QC_PNG_NUM_CASES

        res = process_one_case(row, qc_png_save=qc_png_save)

        # Remove previous rows with the same case_id and append the latest result
        result_map[case_id] = res

        if res.get("status") == "OK":
            summary_ok_case_ids.add(case_id)

        processed_now += 1

        # ------------------------------------------------------------
        # E. Intermediate save
        #    Save every five cases to reduce I/O overhead
        # ------------------------------------------------------------
        if processed_now % 5 == 0:
            current_summary = pd.DataFrame(list(result_map.values()))
            save_report_csvs(current_summary)
            print(f"\n[RESUME] Intermediate save done. processed_now={processed_now}")

    # ============================================================
    # 3) Final save
    # ============================================================
    summary = pd.DataFrame(list(result_map.values()))
    save_report_csvs(summary)

    print("=" * 80)
    print("ALL DONE")
    print("TOTAL ROWS:", len(summary))
    print("OK:", int((summary["status"] == "OK").sum()) if "status" in summary.columns else "NA")
    print("FAILED:", int((summary["status"] == "FAILED").sum()) if "status" in summary.columns else "NA")
    print("SKIPPED_OK:", skipped_ok)
    print("PROCESSED_NOW:", processed_now)
    print("REPROCESSED_PARTIAL:", reprocessed_partial)
    print("REPROCESSED_MISSING_SUMMARY:", reprocessed_missing_summary)
    print("SUMMARY:", SUMMARY_CSV)
    print("QC_ERROR_CSV:", QC_ERROR_CSV)
    print("H5_INDEX_CSV:", H5_INDEX_CSV)
    print("ORIENTATION_QC_CSV:", ORIENTATION_QC_CSV)
    print("LUNG_H5_DIR:", LUNG_H5_DIR)
    print("RIB_H5_DIR:", RIB_H5_DIR)
    print("QC_PNG_DIR:", QC_PNG_DIR)
    print("=" * 80)


if __name__ == "__main__":
    main()