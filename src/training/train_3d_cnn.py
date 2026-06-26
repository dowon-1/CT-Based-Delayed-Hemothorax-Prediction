# ============================================================
# 3D CNN baseline training pipeline
# This script was organized for GitHub/project management.
# Edit paths and configuration variables before running.
# Raw clinical data and model weights are not included.
# ============================================================

# r3d18_ddp_1ch_fullvolume_v2.py
# - DDP-safe train/val/test
# - DDP-compatible upsampling via Subset indices
# - 3D volume classification with torchvision r3d_18
# - input volume resized to [C, D, H, W] = [1, 64, 224, 224]
# - batch size 1 recommended for memory
# - keep augmentation spirit from original code:
#   left-right flip / small affine / weak intensity scaling
#   (z-flip removed)
# - pretrained r3d_18 supported for 1-channel input by averaging first conv weights
# - save 3 checkpoints:
#   1) best composite
#   2) best val loss
#   3) last model
# - after training, evaluate all 3 checkpoints:
#   choose optimal threshold on validation (Youden J),
#   then report validation/test results at that threshold
import json
import os, random, h5py, socket
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, confusion_matrix
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging

# DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# torchvision 3D backbone + augmentation
from torchvision.models.video import r3d_18, R3D_18_Weights
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torchvision.models.video import r3d_18, R3D_18_Weights
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from torchvision.models.video import mc3_18, MC3_18_Weights


BACKBONE_NAME = os.getenv("BACKBONE_NAME", "r3d_18")  # r3d_18, r2plus1d_18, or mc3_18
# =========================
# Config
# =========================
CSV_PATH = os.getenv("CSV_PATH", "./sample/sample_h5_index.csv")
CKPT_DIR = os.getenv("CKPT_DIR", "./checkpoints")
LOG_DIR = os.getenv("LOG_DIR", os.path.join(CKPT_DIR, "train_3d_cnn"))
JSON_LOG_PATH = os.getenv("JSON_LOG_PATH", os.path.join(LOG_DIR, "train_3d_cnn.jsonl"))
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# 3D input: [C, D, H, W]
# D is automatically set to the maximum depth in the training split
TARGET_D = None
TARGET_HW = tuple(map(int, os.getenv("TARGET_HW", "224,224").split(",")))  # (H, W)

IN_CHANNELS = 1
VOLUME_BATCH_SIZE = int(os.getenv("VOLUME_BATCH_SIZE", "2"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "1"))

# New H5 structure
H5_IMAGE_KEY = os.getenv("H5_IMAGE_KEY", "window_broad")
H5_MASK_KEY = os.getenv("H5_MASK_KEY", "mask_lung")

# Current H5 files are assumed to have z=0 inferior and z=last superior
# Model input is standardized to z=0 superior and increasing z toward inferior direction
FLIP_Z_TO_HEAD_FIRST = os.getenv("FLIP_Z_TO_HEAD_FIRST", "true").lower() == "true"

# Set True only when using the lung-masked region
USE_MASK_INPUT = os.getenv("USE_MASK_INPUT", "false").lower() == "true"

# Padding value is 0.0 because padding is applied after z-score normalization
Z_PAD_VALUE = float(os.getenv("Z_PAD_VALUE", "0.0"))

EPOCHS = int(os.getenv("EPOCHS", "200000"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "1e-5"))
SEED = int(os.getenv("SEED", "42"))

USE_PRETRAIN_3D = os.getenv("USE_PRETRAIN_3D", "true").lower() == "true"

# If H5 volume has multiple channels in last dim, choose one channel for 1ch input.
# None -> average across channels
SINGLE_CHANNEL_INDEX = 0

# Train policy
TRAIN_MODE = "stage1_only"     # keep simple
MIN_EPOCHS_BEFORE_UNFREEZE = 1
UNFREEZE_PATIENCE =1 
EARLY_STOP_PATIENCE = int(os.getenv("EARLY_STOP_PATIENCE", "30"))

# after warmup -> all unfreeze
LR_BACKBONE = float(os.getenv("LR_BACKBONE", "1e-5"))
LR_HEAD = float(os.getenv("LR_HEAD", "1e-4"))
WARMUP_EPOCHS = int(os.getenv("WARMUP_EPOCHS", "0"))

# -------- imbalance handling --------
USE_UPSAMPLING = True
UPSAMPLE_TO_BALANCE = True

USE_MANUAL_POS_WEIGHT = False

CLASS_WEIGHT_NEG = 1.0
CLASS_WEIGHT_POS = 1.0  # Use 2.0 for a two-fold experiment; use 3.0 for a three-fold experiment
# -------- normalization --------
USE_IMAGENET_NORM = False
USE_DATASET_ZSCORE = True

# Values computed from the training split
TRAIN_MEAN_1CH = None
TRAIN_STD_1CH  = None

# Composite metric for "best model"
COMPOSITE_WEIGHTS = (0.7, 0.3)  # 0.3*AUC + 0.7*AUPRC

if not torch.cuda.is_available():
    raise SystemExit("❌ CUDA is not available. This script requires a GPU.")

DEVICE = None

# =========================
# Utils
# =========================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True


def resize_xy_torch(x: torch.Tensor, target_hw: Tuple[int, int]):
    """
    x: [C, D, H, W]
    target_hw: (H, W)

    Resize only X/Y to 224x224 without resizing the z-axis.
    """
    C, D, H, W = x.shape

    if (H, W) == tuple(target_hw):
        return x

    # [C,D,H,W] -> [D,C,H,W]
    imgs = x.permute(1, 0, 2, 3)

    imgs = F.interpolate(
        imgs,
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    )

    # [D,C,H,W] -> [C,D,H,W]
    return imgs.permute(1, 0, 2, 3)


def pad_depth_keep_head(x: torch.Tensor, target_d: int, pad_value: float = 0.0):
    """
    x: [C, D, H, W]

    Assumptions:
      - x is already in superior-first order
      - z=0 corresponds to the superior/apical side
      - increasing z moves toward the inferior/basal side

    Purpose:
      - keep the superior side aligned at z=0 for all patients
      - add zero padding to the inferior side when depth is insufficient
    """
    C, D, H, W = x.shape

    if target_d is None:
        raise ValueError("target_d is None. Compute the maximum depth from the training split first.")

    if D == target_d:
        return x

    if D > target_d:
        # If val/test is longer than the training maximum, keep only the first target_d slices from the superior side
        return x[:, :target_d, :, :].contiguous()

    out = x.new_full((C, target_d, H, W), float(pad_value))

    # Key step: place the real CT volume at the beginning
    # z=0 remains fixed at the superior side; padding is appended toward the inferior side
    out[:, :D, :, :] = x

    return out


def get_h5_depth(path: str, key: str = H5_IMAGE_KEY) -> int:
    with h5py.File(path, "r") as h5:
        if key not in h5:
            raise KeyError(f"{key} not found in {path}. Available keys={list(h5.keys())}")
        return int(h5[key].shape[0])


def compute_train_max_depth_from_csv(csv_path: str, split: str = "train") -> int:
    """
    Use the maximum H5 slice count in the training split as TARGET_D.
    """
    df = pd.read_csv(csv_path)
    items = df[df["split"] == split][["path"]].reset_index(drop=True)

    max_d = 0
    max_path = None

    for _, row in tqdm(items.iterrows(), total=len(items), desc=f"Computing {split} max depth"):
        path = row["path"]
        d = get_h5_depth(path, H5_IMAGE_KEY)

        if d > max_d:
            max_d = d
            max_path = path

    print(f"[MAX_D] split={split}, max_d={max_d}, max_path={max_path}")
    return int(max_d)


def imagenet_normalize_1ch(vol: torch.Tensor) -> torch.Tensor:
    """
    vol: [1,D,H,W], float
    """
    mean = torch.tensor(IMAGENET_MEAN_1CH, device=vol.device, dtype=vol.dtype)[:, None, None, None]
    std  = torch.tensor(IMAGENET_STD_1CH,  device=vol.device, dtype=vol.dtype)[:, None, None, None]
    return (vol - mean) / std

def dataset_zscore_1ch(vol: torch.Tensor, mean_1ch: float, std_1ch: float) -> torch.Tensor:
    """
    vol: [1,D,H,W], float
    Mean/std are computed from the training split
    """
    mean = torch.tensor([mean_1ch], device=vol.device, dtype=vol.dtype)[:, None, None, None]
    std  = torch.tensor([std_1ch],  device=vol.device, dtype=vol.dtype)[:, None, None, None]
    return (vol - mean) / (std + 1e-8)
def make_balanced_indices(labels: np.ndarray, seed: int = 0, to_balance: bool = True) -> List[int]:
    rs = np.random.RandomState(seed)
    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]

    if len(idx0) == 0 or len(idx1) == 0:
        out = np.arange(len(labels))
        rs.shuffle(out)
        return out.tolist()

    if not to_balance:
        out = np.arange(len(labels))
        rs.shuffle(out)
        return out.tolist()

    n0, n1 = len(idx0), len(idx1)
    nmax = max(n0, n1)

    if n0 < nmax:
        extra0 = rs.choice(idx0, size=nmax - n0, replace=True)
        idx0b = np.concatenate([idx0, extra0])
    else:
        idx0b = idx0

    if n1 < nmax:
        extra1 = rs.choice(idx1, size=nmax - n1, replace=True)
        idx1b = np.concatenate([idx1, extra1])
    else:
        idx1b = idx1

    out = np.concatenate([idx0b, idx1b])
    rs.shuffle(out)
    return out.tolist()
def compute_train_mean_std_from_csv(
    csv_path: str,
    split: str = "train",
    target_hw: Tuple[int, int] = TARGET_HW,
) -> Tuple[float, float]:
    """
    Compute dataset-level mean/std using only the training split.

    Rules:
      - use H5_IMAGE_KEY from the new H5 structure
      - divide by 255.0 if the input is uint8
      - flip to superior-first if the current H5 direction is inferior-to-superior
      - resize only X/Y to 224x224
      - do not resize Z
      - do not include Z padding in mean/std computation
    """
    df = pd.read_csv(csv_path)
    items = df[df["split"] == split][["path"]].reset_index(drop=True)

    sum_val = 0.0
    sum_sq = 0.0
    count = 0

    for _, row in tqdm(items.iterrows(), total=len(items), desc=f"Computing {split} mean/std"):
        path = row["path"]

        with h5py.File(path, "r") as h5:
            if H5_IMAGE_KEY not in h5:
                raise KeyError(f"{H5_IMAGE_KEY} not found in {path}. Available keys={list(h5.keys())}")

            vol = h5[H5_IMAGE_KEY][:]   # [Z,H,W] or [Z,H,W,C]

            if USE_MASK_INPUT:
                if H5_MASK_KEY not in h5:
                    raise KeyError(f"{H5_MASK_KEY} not found in {path}. Available keys={list(h5.keys())}")
                mask = h5[H5_MASK_KEY][:]
            else:
                mask = None

        if H5_IMAGE_KEY.startswith("window_") or H5_IMAGE_KEY == "image_hu":
            if vol.dtype != np.int16:
                raise TypeError(
                    f"{H5_IMAGE_KEY} must be int16. "
                    f"Current dtype={vol.dtype}, path={path}"
                )
            vol_f = vol.astype("float32")
        else:
            if vol.dtype == np.uint8:
                vol_f = vol.astype("float32") / 255.0
            else:
                vol_f = vol.astype("float32")

        # Convert to a single channel
        if vol_f.ndim == 3:
            vol_1ch = vol_f.astype("float32")
        elif vol_f.ndim == 4:
            Z, H, W, C = vol_f.shape
            if C == 1:
                vol_1ch = vol_f[..., 0].astype("float32")
            else:
                if SINGLE_CHANNEL_INDEX is None:
                    vol_1ch = vol_f.mean(axis=-1).astype("float32")
                else:
                    vol_1ch = vol_f[..., SINGLE_CHANNEL_INDEX].astype("float32")
        else:
            raise ValueError(f"Unsupported H5 volume shape: {vol_f.shape}")

        # Optional: use only the lung-masked region
        if USE_MASK_INPUT:
            mask_f = (mask > 0).astype("float32")
            if mask_f.shape != vol_1ch.shape:
                raise ValueError(f"mask shape mismatch: vol={vol_1ch.shape}, mask={mask_f.shape}")
            vol_1ch = vol_1ch * mask_f

        vol_t = torch.from_numpy(vol_1ch).unsqueeze(0)  # [1,Z,H,W]

        # Convert from inferior-first to superior-first z-order
        if FLIP_Z_TO_HEAD_FIRST:
            vol_t = vol_t.flip(1)

        # Resize X/Y only
        vol_t = resize_xy_torch(vol_t, target_hw)       # [1,D,224,224]

        vol_np = vol_t.numpy()

        sum_val += float(vol_np.sum())
        sum_sq += float((vol_np ** 2).sum())
        count += int(vol_np.size)

    mean = sum_val / count
    var = (sum_sq / count) - (mean ** 2)
    std = float(np.sqrt(max(var, 1e-12)))

    return float(mean), float(std)
# =========================
# Dataset
# =========================
class H5VolumeDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        split: str,
        augment: bool = False,
        mean_1ch: float = None,
        std_1ch: float = None,
        target_d: int = None,
    ):
        df = pd.read_csv(csv_path)
        self.items = df[df["split"] == split][["id", "path", "label"]].reset_index(drop=True)
        self.augment = augment
        self.mean_1ch = mean_1ch
        self.std_1ch = std_1ch
        self.target_d = target_d

    def __len__(self):
        return len(self.items)

    def _to_single_channel(self, vol_f: np.ndarray) -> np.ndarray:
        if vol_f.ndim == 3:
            return vol_f.astype("float32")

        if vol_f.ndim != 4:
            raise ValueError(f"Unsupported H5 volume shape: {vol_f.shape}")

        Z, H, W, C = vol_f.shape

        if C == 1:
            return vol_f[..., 0].astype("float32")

        if SINGLE_CHANNEL_INDEX is None:
            return vol_f.mean(axis=-1).astype("float32")

        if not (0 <= SINGLE_CHANNEL_INDEX < C):
            raise ValueError(f"SINGLE_CHANNEL_INDEX={SINGLE_CHANNEL_INDEX} out of range for C={C}")

        return vol_f[..., SINGLE_CHANNEL_INDEX].astype("float32")

    def __getitem__(self, idx):
        row = self.items.iloc[idx]
        pid, path, y = str(row["id"]), row["path"], int(row["label"])

        with h5py.File(path, "r") as h5:
            if H5_IMAGE_KEY not in h5:
                raise KeyError(f"{H5_IMAGE_KEY} not found in {path}. Available keys={list(h5.keys())}")

            vol = h5[H5_IMAGE_KEY][:]   # [Z,H,W] or [Z,H,W,C]

            if USE_MASK_INPUT:
                if H5_MASK_KEY not in h5:
                    raise KeyError(f"{H5_MASK_KEY} not found in {path}. Available keys={list(h5.keys())}")
                mask = h5[H5_MASK_KEY][:]
            else:
                mask = None

        if H5_IMAGE_KEY.startswith("window_") or H5_IMAGE_KEY == "image_hu":
            if vol.dtype != np.int16:
                raise TypeError(
                    f"{H5_IMAGE_KEY} must be int16. "
                    f"Current dtype={vol.dtype}, path={path}"
                )
            vol_f = vol.astype("float32")
        else:
            if vol.dtype == np.uint8:
                vol_f = vol.astype("float32") / 255.0
            else:
                vol_f = vol.astype("float32")

        vol_1ch = self._to_single_channel(vol_f)        # [Z,H,W]

        # Optional: use only the lung-masked region as input
        if USE_MASK_INPUT:
            mask_f = (mask > 0).astype("float32")
            if mask_f.shape != vol_1ch.shape:
                raise ValueError(f"mask shape mismatch: vol={vol_1ch.shape}, mask={mask_f.shape}")
            vol_1ch = vol_1ch * mask_f

        vol_t = torch.from_numpy(vol_1ch).unsqueeze(0)  # [1,Z,H,W]

        # Current H5 order is z=0 inferior and z=last superior
        # Standardize model input to z=0 superior and increasing z toward inferior
        if FLIP_Z_TO_HEAD_FIRST:
            vol_t = vol_t.flip(1)

        # Resize only X/Y to 224x224 and keep Z unchanged
        # Resize only X/Y to 224x224 and keep Z unchanged
        vol_t = resize_xy_torch(vol_t, TARGET_HW)       # [1,D,224,224]

        # Apply z-score normalization to int16 HU window values first
        if USE_DATASET_ZSCORE:
            if self.mean_1ch is None or self.std_1ch is None:
                raise ValueError("USE_DATASET_ZSCORE=True, but mean/std were not provided to the dataset.")
            vol_t = dataset_zscore_1ch(vol_t, self.mean_1ch, self.std_1ch)
        elif USE_IMAGENET_NORM:
            vol_t = imagenet_normalize_1ch(vol_t)

        # Apply augmentation after z-score normalization
        # Affine fill=0.0 is acceptable because it corresponds to the mean in z-score space
        if self.augment:
            vol_t = self._augment_volume(vol_t)

        # Apply zero padding toward the inferior side at the end
        # Keep z=0 fixed at the superior side and pad the inferior side
        vol_t = pad_depth_keep_head(
            vol_t,
            target_d=self.target_d,
            pad_value=Z_PAD_VALUE,
        )

        return vol_t, torch.tensor(y, dtype=torch.long), pid

    def _augment_volume(self, vol: torch.Tensor) -> torch.Tensor:
        """

        vol: [1,D,H,W], z-scored float
        z-flip removed
        """
        vol = vol.float()
        C, D, H, W = vol.shape

        # 1) left-right flip
        if torch.rand(1).item() < 0.5:
            vol = vol.flip(-1)

        # 2) same small affine on all slices
        if torch.rand(1).item() < 0.5:
            angle = (torch.rand(1).item() * 2 - 1) * 3.0  # [-3, 3]
            max_dx = int(0.02 * W)
            max_dy = int(0.02 * H)
            tx = int((torch.rand(1).item() * 2 - 1) * max_dx)
            ty = int((torch.rand(1).item() * 2 - 1) * max_dy)
            scale = 1.0 + (torch.rand(1).item() * 0.02 - 0.01)  # [0.99, 1.01]

            imgs = vol.permute(1, 0, 2, 3)  # [D,1,H,W]
            imgs = TF.affine(
                imgs,
                angle=angle,
                translate=[tx, ty],
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )
            vol = imgs.permute(1, 0, 2, 3)

        # 3) weak intensity scale
        if torch.rand(1).item() < 0.3:
            s = 1.0 + (torch.rand(1).item() * 0.06 - 0.03)
            vol = vol * s

        # Important:
        # Do not clamp to [0, 1] because int16 HU values are converted to z-scored inputs
        return vol

# =========================
# Collate
# =========================
def collate_volumes(batch):
    vols, labels, pids = zip(*batch)
    x = torch.stack(vols, dim=0)   # [B,1,D,H,W]
    y = torch.stack(labels)
    return x, y, list(pids)

# =========================
# Backbone / Model
# =========================
def build_backbone_3d(in_channels=1):
    if BACKBONE_NAME == "r3d_18":
        weights = R3D_18_Weights.DEFAULT if USE_PRETRAIN_3D else None
        m = r3d_18(weights=weights)

    elif BACKBONE_NAME == "r2plus1d_18":
        weights = R2Plus1D_18_Weights.DEFAULT if USE_PRETRAIN_3D else None
        m = r2plus1d_18(weights=weights)

    elif BACKBONE_NAME == "mc3_18":
        weights = MC3_18_Weights.DEFAULT if USE_PRETRAIN_3D else None
        m = mc3_18(weights=weights)

    else:
        raise ValueError(f"Unknown BACKBONE_NAME: {BACKBONE_NAME}")

    if in_channels != 3:
        old = m.stem[0]
        new = nn.Conv3d(
            in_channels=in_channels,
            out_channels=old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False
        )

        with torch.no_grad():
            w = old.weight.clone()
            tmpl = w.mean(1, keepdim=True)
            new.weight.copy_(tmpl.repeat(1, in_channels, 1, 1, 1))

        m.stem[0] = new

    feat_dim = m.fc.in_features
    m.fc = nn.Identity()
    return m, feat_dim


class VolumeResNet3D(nn.Module):
    def __init__(self, in_channels=1, dropout_p=0.3):
        super().__init__()
        self.backbone, feat_dim = build_backbone_3d(in_channels=in_channels)
        self.dropout = nn.Dropout(dropout_p)
        self.cls = nn.Linear(feat_dim, 1)

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)   # [B,512]
        feats = self.dropout(feats)
        logits = self.cls(feats)   # [B,1]
        return logits

# =========================
# Freezing / Optimizer
# =========================
def set_backbone_trainable(model, mode: str = "all"):
    m = model.module if isinstance(model, DDP) else model
    backbone = m.backbone

    for p in backbone.parameters():
        p.requires_grad = False

    if mode == "frozen":
        pass
    elif mode == "all":
        for p in backbone.parameters():
            p.requires_grad = True
    elif mode == "layer4_only":
        for p in backbone.layer4.parameters():
            p.requires_grad = True
    elif mode == "layer34":
        for p in backbone.layer3.parameters():
            p.requires_grad = True
        for p in backbone.layer4.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    for p in m.cls.parameters():
        p.requires_grad = True


def build_optimizer_with_two_lrs(model, lr_backbone: float, lr_head: float, weight_decay: float):
    m = model.module if isinstance(model, DDP) else model
    backbone_param_ids = set(id(p) for p in m.backbone.parameters())

    backbone_params, head_params = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in backbone_param_ids:
            backbone_params.append(p)
        else:
            head_params.append(p)

    param_groups = []
    if len(backbone_params) > 0:
        param_groups.append({"params": backbone_params, "lr": lr_backbone})
    if len(head_params) > 0:
        param_groups.append({"params": head_params, "lr": lr_head})

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def set_bn_eval(module):
    for m in module.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()

# =========================
# Metrics
# =========================
def metrics_from_conf(conf: np.ndarray) -> Dict[str, Any]:
    C = conf.shape[0]
    total = conf.sum()
    per = []
    support = conf.sum(1)
    pred_sum = conf.sum(0)

    for c in range(C):
        TP = conf[c, c]
        FN = support[c] - TP
        FP = pred_sum[c] - TP
        TN = total - TP - FP - FN

        sens = TP / (TP + FN + 1e-9)
        spec = TN / (TN + FP + 1e-9)
        prec = TP / (TP + FP + 1e-9)
        f1   = 2 * prec * sens / (prec + sens + 1e-9)

        per.append({
            "class": c,
            "TP": int(TP), "FP": int(FP), "TN": int(TN), "FN": int(FN),
            "sensitivity": sens,
            "specificity": spec,
            "precision": prec,
            "recall": sens,
            "f1": f1,
            "support": int(support[c]),
        })

    macro = {
        "precision": float(np.mean([x["precision"] for x in per])),
        "recall": float(np.mean([x["recall"] for x in per])),
        "f1": float(np.mean([x["f1"] for x in per])),
        "sensitivity": float(np.mean([x["sensitivity"] for x in per])),
        "specificity": float(np.mean([x["specificity"] for x in per])),
    }

    acc = float(np.trace(conf) / max(total, 1))
    bal_acc = float(np.mean([x["recall"] for x in per]))

    return {
        "per_class": per,
        "macro": macro,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "total": int(total),
        "confusion": conf.astype(int),
    }

# =========================
# DDP helpers
# =========================
def ddp_is_on() -> bool:
    return dist.is_available() and dist.is_initialized()

def ddp_world_size() -> int:
    return dist.get_world_size() if ddp_is_on() else 1

def ddp_rank() -> int:
    return dist.get_rank() if ddp_is_on() else 0

def all_gather_1d_tensor(t: torch.Tensor) -> torch.Tensor:
    if not ddp_is_on():
        return t

    ws = ddp_world_size()
    local_n = torch.tensor([t.numel()], device=t.device, dtype=torch.long)
    sizes = [torch.zeros_like(local_n) for _ in range(ws)]
    dist.all_gather(sizes, local_n)
    sizes = [int(s.item()) for s in sizes]
    max_n = max(sizes)

    if t.numel() < max_n:
        pad = torch.zeros(max_n - t.numel(), device=t.device, dtype=t.dtype)
        t_pad = torch.cat([t.flatten(), pad], dim=0)
    else:
        t_pad = t.flatten()

    gathered = [torch.zeros(max_n, device=t.device, dtype=t.dtype) for _ in range(ws)]
    dist.all_gather(gathered, t_pad)

    out = torch.cat([g[:sizes[i]] for i, g in enumerate(gathered)], dim=0)
    return out
def weighted_bce_loss_normalized(logits, y, criterion):
    raw_loss = criterion(logits, y.float())  # [B]

    sample_w = torch.where(
        y == 1,
        torch.full_like(y.float(), CLASS_WEIGHT_POS),
        torch.full_like(y.float(), CLASS_WEIGHT_NEG),
    )

    loss = (raw_loss * sample_w).sum() / sample_w.sum().clamp_min(1e-8)
    return loss
# =========================
# Eval (DDP-safe, threshold-free)
# =========================
@torch.no_grad()
def evaluate_with_loss_ddp(model, loader, criterion):
    model.eval()

    conf_local = torch.zeros((2, 2), device=DEVICE, dtype=torch.long)
    loss_sum_local = torch.zeros((), device=DEVICE, dtype=torch.float32)
    n_local = torch.zeros((), device=DEVICE, dtype=torch.float32)

    y_true_list = []
    y_prob_list = []

    for xb, yb, pids in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)

        logits = model(xb).squeeze(-1)
        loss = weighted_bce_loss_normalized(logits, yb, criterion)

        bs = xb.size(0)
        loss_sum_local += loss.detach() * bs
        n_local += bs

        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).long()
        true = yb.long()

        for t, p in zip(true, pred):
            conf_local[t, p] += 1

        y_true_list.append(true)
        y_prob_list.append(prob)

    if ddp_is_on():
        dist.all_reduce(conf_local, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_sum_local, op=dist.ReduceOp.SUM)
        dist.all_reduce(n_local, op=dist.ReduceOp.SUM)

    avg_loss = (loss_sum_local / torch.clamp(n_local, min=1.0)).item()
    conf = conf_local.detach().cpu().numpy().astype(np.int64)
    mets = metrics_from_conf(conf)

    y_true_local = torch.cat(y_true_list, dim=0) if len(y_true_list) else torch.zeros((0,), device=DEVICE, dtype=torch.long)
    y_prob_local = torch.cat(y_prob_list, dim=0) if len(y_prob_list) else torch.zeros((0,), device=DEVICE, dtype=torch.float32)

    y_true_all = all_gather_1d_tensor(y_true_local)
    y_prob_all = all_gather_1d_tensor(y_prob_local)

    if ddp_rank() == 0:
        y_true_np = y_true_all.detach().cpu().numpy().astype(int)
        y_prob_np = y_prob_all.detach().cpu().numpy().astype(float)
        if len(np.unique(y_true_np)) < 2:
            auc = float("nan")
            auprc = float("nan")
        else:
            auc = float(roc_auc_score(y_true_np, y_prob_np))
            auprc = float(average_precision_score(y_true_np, y_prob_np))
    else:
        auc = 0.0
        auprc = 0.0

    if ddp_is_on():
        scal = torch.tensor([auc, auprc], device=DEVICE, dtype=torch.float32)
        dist.broadcast(scal, src=0)
        auc = float(scal[0].item())
        auprc = float(scal[1].item())

    return mets, avg_loss, auc, auprc

# =========================
# Standalone eval for final checkpoint comparison
# =========================
@torch.no_grad()
def infer_collect_singleprocess(model, loader, criterion, device):
    model.eval()
    y_true_list, y_prob_list = [], []
    loss_sum = 0.0
    n = 0

    for xb, yb, pids in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb).squeeze(-1)
        loss = weighted_bce_loss_normalized(logits, yb, criterion)

        prob = torch.sigmoid(logits)

        bs = xb.size(0)
        loss_sum += loss.item() * bs
        n += bs

        y_true_list.append(yb.detach().cpu())
        y_prob_list.append(prob.detach().cpu())

    y_true = torch.cat(y_true_list).numpy().astype(int) if y_true_list else np.array([], dtype=int)
    y_prob = torch.cat(y_prob_list).numpy().astype(float) if y_prob_list else np.array([], dtype=float)
    avg_loss = loss_sum / max(1, n)

    if len(np.unique(y_true)) < 2:
        auc = float("nan")
        auprc = float("nan")
    else:
        auc = float(roc_auc_score(y_true, y_prob))
        auprc = float(average_precision_score(y_true, y_prob))

    return y_true, y_prob, avg_loss, auc, auprc


def choose_best_threshold_by_youden(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Choose threshold on validation set using Youden's J = TPR - FPR.
    """
    if len(np.unique(y_true)) < 2:
        return 0.5

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    valid = np.isfinite(thresholds)
    if valid.sum() == 0:
        return 0.5

    fpr = fpr[valid]
    tpr = tpr[valid]
    thresholds = thresholds[valid]

    j = tpr - fpr
    best_idx = int(np.argmax(j))
    thr = float(thresholds[best_idx])

    if not np.isfinite(thr):
        thr = 0.5
    return thr


def metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, Any]:
    if y_true.size == 0:
        return {
            "threshold": threshold,
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "specificity": float("nan"),
            "f1": float("nan"),
            "confusion": np.zeros((2, 2), dtype=int),
        }

    y_pred = (y_prob >= threshold).astype(int)
    conf = confusion_matrix(y_true, y_pred, labels=[0, 1])
    mets = metrics_from_conf(conf)

    return {
        "threshold": threshold,
        "accuracy": mets["accuracy"],
        "balanced_accuracy": mets["balanced_accuracy"],
        "precision": mets["macro"]["precision"],
        "recall": mets["macro"]["recall"],
        "specificity": mets["macro"]["specificity"],
        "f1": mets["macro"]["f1"],
        "confusion": mets["confusion"],
    }


def print_threshold_report(tag: str, split_name: str, loss: float, auc: float, auprc: float, thr_metrics: Dict[str, Any]):
    print(
        f"[{tag}] {split_name} | "
        f"loss={loss:.4f} | auc={auc:.4f} | auprc={auprc:.4f} | "
        f"thr={thr_metrics['threshold']:.4f} | "
        f"acc={thr_metrics['accuracy']:.4f} | "
        f"bal_acc={thr_metrics['balanced_accuracy']:.4f} | "
        f"precision={thr_metrics['precision']:.4f} | "
        f"recall={thr_metrics['recall']:.4f} | "
        f"specificity={thr_metrics['specificity']:.4f} | "
        f"f1={thr_metrics['f1']:.4f}"
    )
    print(f"[{tag}] {split_name} confusion:\n{thr_metrics['confusion']}")

# =========================
# Loaders
# =========================
def make_loaders(csv_path, distributed=False, rank=0, world_size=1, mean_1ch=None, std_1ch=None, target_d=None):
    train_base = H5VolumeDataset(
        csv_path, split="train", augment=True,
        mean_1ch=mean_1ch, std_1ch=std_1ch,
        target_d=target_d,
    )
    valid_ds = H5VolumeDataset(
        csv_path, split="val", augment=False,
        mean_1ch=mean_1ch, std_1ch=std_1ch,
        target_d=target_d,
    )
    test_ds = H5VolumeDataset(
        csv_path, split="test", augment=False,
        mean_1ch=mean_1ch, std_1ch=std_1ch,
        target_d=target_d,
    )

    labels = train_base.items["label"].astype(int).to_numpy()
    n0 = int((labels == 0).sum())
    n1 = int((labels == 1).sum())

    pos_weight = 1.0

    train_ds = train_base
    epoch_counts = {"n0_epoch": n0, "n1_epoch": n1}

    if USE_UPSAMPLING:
        indices = make_balanced_indices(labels, seed=SEED, to_balance=UPSAMPLE_TO_BALANCE)
        train_ds = Subset(train_base, indices)

        labs_epoch = labels[np.array(indices)]
        epoch_counts["n0_epoch"] = int((labs_epoch == 0).sum())
        epoch_counts["n1_epoch"] = int((labs_epoch == 1).sum())

    if distributed:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        valid_sampler = DistributedSampler(valid_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        test_sampler  = DistributedSampler(test_ds,  num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        valid_sampler = None
        test_sampler  = None

    train_loader = DataLoader(
        train_ds,
        batch_size=VOLUME_BATCH_SIZE,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_volumes,
        drop_last=False
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=VOLUME_BATCH_SIZE,
        sampler=valid_sampler,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_volumes,
        drop_last=False
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=VOLUME_BATCH_SIZE,
        sampler=test_sampler,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_volumes,
        drop_last=False
    )

    stats = {
        "n0": n0,
        "n1": n1,
        "pos_weight": pos_weight,
        **epoch_counts
    }
    return train_loader, valid_loader, test_loader, train_sampler, stats

# =========================
# Train
# =========================
def train():
    global DEVICE, TARGET_D
    set_seed()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        DEVICE = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl")
        is_main = (local_rank == 0)
    else:
        local_rank = 0
        torch.cuda.set_device(0)
        DEVICE = torch.device("cuda", 0)
        is_main = True

    if is_main:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
        logging.info(f"Host: {socket.gethostname()} | PyTorch {torch.__version__} | Device {DEVICE} | world_size={world_size}")
    else:
        logging.basicConfig(level=logging.ERROR)
    train_mean_1ch = None
    train_std_1ch = None
    # =========================================================
    # Compute maximum Z slice count from the training split
    # =========================================================
    if is_main:
        logging.info("Computing TARGET_D from TRAIN split max depth...")
        TARGET_D = compute_train_max_depth_from_csv(CSV_PATH, split="train")
        logging.info(f"[TARGET_D] train max depth = {TARGET_D}")

    if distributed:
        d_tensor = torch.zeros(1, device=DEVICE, dtype=torch.int64)
        if is_main:
            d_tensor[0] = int(TARGET_D)
        dist.broadcast(d_tensor, src=0)
        TARGET_D = int(d_tensor.item())
    if USE_DATASET_ZSCORE:
        if is_main:
            logging.info("Computing dataset-level z-score stats from TRAIN split (one-time pre-pass)...")
            train_mean_1ch, train_std_1ch = compute_train_mean_std_from_csv(
                CSV_PATH, split="train", target_hw=TARGET_HW
            )
            logging.info(f"[Z-SCORE] TRAIN_MEAN_1CH={train_mean_1ch:.6f}, TRAIN_STD_1CH={train_std_1ch:.6f}")

        if distributed:
            stats_tensor = torch.zeros(2, device=DEVICE, dtype=torch.float32)
            if is_main:
                stats_tensor[0] = float(train_mean_1ch)
                stats_tensor[1] = float(train_std_1ch)
            dist.broadcast(stats_tensor, src=0)
            train_mean_1ch = float(stats_tensor[0].item())
            train_std_1ch = float(stats_tensor[1].item())
    train_loader, valid_loader, test_loader, train_sampler, stats = make_loaders(
        CSV_PATH,
        distributed=distributed,
        rank=local_rank,
        world_size=world_size,
        mean_1ch=train_mean_1ch,
        std_1ch=train_std_1ch,
        target_d=TARGET_D,
    )

    if is_main:
        logging.info(f"Train original counts: n0={stats['n0']} n1={stats['n1']}")
        if USE_UPSAMPLING:
            logging.info(f"Train epoch counts(after upsampling): n0_epoch={stats['n0_epoch']} n1_epoch={stats['n1_epoch']}")
        logging.info(
            f"class weights used: neg={CLASS_WEIGHT_NEG:.2f}, pos={CLASS_WEIGHT_POS:.2f} "
            f"(normalized by batch weight sum)"
        )
        logging.info(f"Input volume = [C,D,H,W] = [{IN_CHANNELS},{TARGET_D},{TARGET_HW[0]},{TARGET_HW[1]}]")
        logging.info(f"H5_IMAGE_KEY = {H5_IMAGE_KEY}")
        logging.info(f"USE_MASK_INPUT = {USE_MASK_INPUT}")
        logging.info(f"FLIP_Z_TO_HEAD_FIRST = {FLIP_Z_TO_HEAD_FIRST}")
        logging.info("Z padding = keep head at z=0, pad inferior/end side")
        logging.info(f"Batch size per GPU = {VOLUME_BATCH_SIZE}")

    
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    model = VolumeResNet3D(in_channels=IN_CHANNELS).to(DEVICE)

    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )

    # warmup: frozen backbone
    set_backbone_trainable(model, mode="frozen")
    if is_main:
        logging.info(f"[WARMUP] first {WARMUP_EPOCHS} epochs: backbone frozen, head only.")
        logging.info(f"After warmup: backbone -> all")

    optimizer = build_optimizer_with_two_lrs(
        model, lr_backbone=LR_BACKBONE, lr_head=LR_HEAD, weight_decay=WEIGHT_DECAY
    )
    scaler = GradScaler(enabled=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=50, min_lr=1e-9, verbose=is_main
    )

    if is_main:
        tb_dir = os.path.join(LOG_DIR, "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
    else:
        writer = None

    best_composite = -float("inf")
    best_val_loss = float("inf")

    exp_name = f"RIB_SyncBatchNorm_715_D{TARGET_D}_{H5_IMAGE_KEY}_R3D18"

    best_composite_path = os.path.join(CKPT_DIR, f"{exp_name}_BEST_COMPOSITE.pth")
    best_valloss_path   = os.path.join(CKPT_DIR, f"{exp_name}_BEST_VALLOSS.pth")
    last_model_path     = os.path.join(CKPT_DIR, f"{exp_name}_LAST.pth")

    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        # after warmup -> all
        if epoch == WARMUP_EPOCHS + 1:
            set_backbone_trainable(model, mode="all")
            optimizer = build_optimizer_with_two_lrs(
                model, lr_backbone=LR_BACKBONE, lr_head=LR_HEAD, weight_decay=WEIGHT_DECAY
            )
            if is_main:
                logging.info(
                    f"[SWITCH] epoch {epoch}: backbone -> all "
                    f"(lr_backbone={LR_BACKBONE}, lr_head={LR_HEAD})"
                )

        model.train()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        run_loss = torch.zeros((), device=DEVICE, dtype=torch.float32)
        n_bags   = torch.zeros((), device=DEVICE, dtype=torch.float32)

        pbar = tqdm(train_loader, ncols=120, desc=f"Epoch {epoch}/{EPOCHS}") if is_main else train_loader

        for xb, yb, pids in pbar:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=True):
                logits = model(xb).squeeze(-1)
                loss = weighted_bce_loss_normalized(logits, yb, criterion)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = xb.size(0)
            run_loss += loss.detach() * bs
            n_bags += bs

            if is_main:
                pbar.set_postfix({"loss": f"{(run_loss / torch.clamp(n_bags, min=1.0)).item():.4f}"})

        if ddp_is_on():
            dist.all_reduce(run_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(n_bags, op=dist.ReduceOp.SUM)

        train_loss_epoch = (run_loss / torch.clamp(n_bags, min=1.0)).item()
        train_mets, train_eval_loss, train_auc, train_auprc = evaluate_with_loss_ddp(
            model, train_loader, criterion
        )
        mets, val_loss, val_auc, val_auprc = evaluate_with_loss_ddp(model, valid_loader, criterion)

        w_auc, w_auprc = COMPOSITE_WEIGHTS
        val_auc_f = val_auc if np.isfinite(val_auc) else -float("inf")
        val_auprc_f = val_auprc if np.isfinite(val_auprc) else -float("inf")
        val_composite = w_auc * val_auc_f + w_auprc * val_auprc_f

        scheduler.step(val_composite)
        cur_lr = optimizer.param_groups[0]["lr"]

        if is_main:
            logging.info(
                f"[E{epoch:04d}] lr={cur_lr:.2e} | "

                f"train_loss={train_loss_epoch:.4f} | "
                f"train_eval_loss={train_eval_loss:.4f} | "
                f"train_acc={train_mets['accuracy']:.4f} | "
                f"train_bal_acc={train_mets['balanced_accuracy']:.4f} | "
                f"train_auc={train_auc:.4f} | "
                f"train_auprc={train_auprc:.4f} | "

                f"val_loss={val_loss:.4f} | "
                f"val_acc={mets['accuracy']:.4f} | "
                f"val_bal_acc={mets['balanced_accuracy']:.4f} | "
                f"val_auc={val_auc:.4f} | "
                f"val_auprc={val_auprc:.4f} | "

                f"no_improve={no_improve}"
            )

            
            if writer is not None:
                # Loss
                writer.add_scalar("Loss/train_step", train_loss_epoch, epoch)
                writer.add_scalar("Loss/train_eval", train_eval_loss, epoch)
                writer.add_scalar("Loss/val", val_loss, epoch)

                # Accuracy
                writer.add_scalar("Acc/train", train_mets["accuracy"], epoch)
                writer.add_scalar("Acc/val", mets["accuracy"], epoch)

                # Balanced Accuracy
                writer.add_scalar("BalAcc/train", train_mets["balanced_accuracy"], epoch)
                writer.add_scalar("BalAcc/val", mets["balanced_accuracy"], epoch)

                # AUC / AUPRC
                writer.add_scalar("AUC/train", train_auc, epoch)
                writer.add_scalar("AUC/val", val_auc, epoch)

                writer.add_scalar("AUPRC/train", train_auprc, epoch)
                writer.add_scalar("AUPRC/val", val_auprc, epoch)

                # Validation selection metric
                writer.add_scalar("Composite/val", val_composite, epoch)

                # Training control
                writer.add_scalar("LR", cur_lr, epoch)
                writer.add_scalar("EarlyStop/no_improve", no_improve, epoch)
            log_dict = {
                "epoch": epoch,
                "lr": float(cur_lr),

                "train_loss_step": float(train_loss_epoch),
                "train_eval_loss": float(train_eval_loss),
                "train_acc": float(train_mets["accuracy"]),
                "train_bal_acc": float(train_mets["balanced_accuracy"]),
                "train_auc": float(train_auc),
                "train_auprc": float(train_auprc),

                "val_loss": float(val_loss),
                "val_acc": float(mets["accuracy"]),
                "val_bal_acc": float(mets["balanced_accuracy"]),
                "val_auc": float(val_auc),
                "val_auprc": float(val_auprc),

                "val_composite": float(val_composite),
            }

            with open(JSON_LOG_PATH, "a") as f:
                f.write(json.dumps(log_dict) + "\n")
            # 1) Best-composite checkpoint saving is retained
            if val_composite > best_composite + 1e-12:
                best_composite = val_composite
                state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                torch.save({
                    "epoch": epoch,
                    "model": state,
                    "best_composite": best_composite,
                    "val_loss": val_loss,
                    "val_auc": val_auc,
                    "val_auprc": val_auprc,
                }, best_composite_path)
                logging.info(f"  -> saved BEST_COMPOSITE to {best_composite_path}")

            # 2) Save the best validation-loss checkpoint + use validation loss for early stopping
            if val_loss < best_val_loss - 1e-12:
                best_val_loss = val_loss
                state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                torch.save({
                    "epoch": epoch,
                    "model": state,
                    "best_val_loss": best_val_loss,
                    "val_composite": val_composite,
                    "val_auc": val_auc,
                    "val_auprc": val_auprc,
                }, best_valloss_path)
                logging.info(f"  -> saved BEST_VALLOSS to {best_valloss_path}")
                no_improve = 0
            else:
                no_improve += 1

            stop_now = (no_improve >= EARLY_STOP_PATIENCE)
        else:
            stop_now = False

        if distributed:
            st = torch.tensor([1 if stop_now else 0], device=DEVICE, dtype=torch.int32)
            dist.broadcast(st, src=0)
            stop_now = bool(st.item())

        if stop_now:
            if is_main:
                logging.info("Early stopping triggered.")
            break

    # 3) last model
    if is_main:
        state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
        torch.save({
            "epoch": epoch,
            "model": state,
        }, last_model_path)
        logging.info(f"  -> saved LAST_MODEL to {last_model_path}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()

    if is_main and writer is not None:
        writer.flush()
        writer.close()

    # =========================
    # Final standalone checkpoint evaluation (main process only)
    # =========================
    if is_main:
        print("\n" + "=" * 100)
        print("Final checkpoint comparison with validation-optimal threshold (Youden J)")
        print("=" * 100)

        # standalone loaders (no DDP)
        _, valid_loader_sp, test_loader_sp, _, stats_sp = make_loaders(
            CSV_PATH,
            distributed=False,
            rank=0,
            world_size=1,
            mean_1ch=train_mean_1ch,
            std_1ch=train_std_1ch,
            target_d=TARGET_D,
        )
        
        criterion_sp = nn.BCEWithLogitsLoss(reduction="none")

        checkpoint_dict = {
            "BEST_COMPOSITE": best_composite_path,
            "BEST_VALLOSS": best_valloss_path,
            "LAST_MODEL": last_model_path,
        }

        for tag, ckpt_path in checkpoint_dict.items():
            model_sp = VolumeResNet3D(in_channels=IN_CHANNELS).to(DEVICE)
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            model_sp.load_state_dict(ckpt["model"], strict=True)

            # validation threshold-free results
            val_y, val_prob, val_loss_sp, val_auc_sp, val_auprc_sp = infer_collect_singleprocess(
                model_sp, valid_loader_sp, criterion_sp, DEVICE
            )

            # choose optimal threshold on validation
            thr = choose_best_threshold_by_youden(val_y, val_prob)

            # validation thresholded results
            val_thr_metrics = metrics_at_threshold(val_y, val_prob, thr)

            # test threshold-free + thresholded results
            test_y, test_prob, test_loss_sp, test_auc_sp, test_auprc_sp = infer_collect_singleprocess(
                model_sp, test_loader_sp, criterion_sp, DEVICE
            )
            test_thr_metrics = metrics_at_threshold(test_y, test_prob, thr)

            print_threshold_report(tag, "VAL",  val_loss_sp,  val_auc_sp,  val_auprc_sp,  val_thr_metrics)
            print_threshold_report(tag, "TEST", test_loss_sp, test_auc_sp, test_auprc_sp, test_thr_metrics)
            print("-" * 100)


if __name__ == "__main__":
    train()