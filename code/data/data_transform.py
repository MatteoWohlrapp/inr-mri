import h5py
import pathlib
import numpy as np
import torch
import fastmri
from fastmri.data import transforms as T
from fastmri.data.subsample import RandomMaskFunc
import matplotlib.pyplot as plt

SRC_ROOT = pathlib.Path("../../../dataset/fastmri/brain/singlecoil_val")
DEST_ROOT = pathlib.Path("../../../dataset/fastmri/brain/singlecoil_val_normalized")
#SRC_ROOT = pathlib.Path(r"C:\Users\jan\Documents\python_files\adlm\data\brain\singlecoil_val")
#DEST_ROOT = pathlib.Path(r"C:\Users\jan\Documents\python_files\adlm\data\brain\singlecoil_val_normalized")


def load_h5(path):
    with h5py.File(path, "r") as f:
        data = f["kspace"][()]
    return data


def load_mri_scan(path: pathlib.Path, undersampled=False):
    mri_data = load_h5(path)
    mri_data = T.to_tensor(mri_data)

    if undersampled:
        mask_func = RandomMaskFunc(center_fractions=[0.1], accelerations=[8])
        mri_data, _, _ = T.apply_mask(mri_data, mask_func)

    mri_data = fastmri.ifft2c(mri_data)
    mri_data = fastmri.complex_abs(mri_data)
    scan = mri_data.numpy()
    return scan


def normalize_scan(scan: torch.Tensor):
    scan_min = scan.min()
    scan_max = scan.max()
    normalized_scan = (scan - scan_min) / (scan_max - scan_min)
    return normalized_scan


def process_files():
    print("Processing files ...")
    path = SRC_ROOT
    print(path)
    for file in path.glob("**/*.h5"):
        print(1)
        print(file)
        fully_sampled_scan = load_mri_scan(file)
        fully_sampled_normalized_scan = normalize_scan(fully_sampled_scan)
        fully_sampled_normalize_scan = 2 * fully_sampled_normalized_scan - 1

        undersampled_scan = load_mri_scan(file, undersampled=True)
        undersampled_normalized_scan = normalize_scan(undersampled_scan)
        undersampled_normalize_scan = 2 * undersampled_normalized_scan - 1

        dest_path = DEST_ROOT / file.relative_to(SRC_ROOT)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(dest_path, "w") as f:
            f.create_dataset("fully_sampled", data=fully_sampled_normalize_scan)
            f.create_dataset("undersampled", data=undersampled_normalize_scan)
