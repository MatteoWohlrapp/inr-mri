import h5py
import pathlib
import numpy as np
import torch
import fastmri
from fastmri.data import transforms as T
import matplotlib.pyplot as plt

# hardcoded paths only for now
SRC_ROOT = pathlib.Path(r'/vol/aimspace/projects/practical_SoSe24/mri_inr/dataset/fastmri/brain')
DEST_ROOT = pathlib.Path(r'/vol/aimspace/projects/practical_SoSe24/mri_inr/dataset/fastmri/brain_norm')


def load_h5(path):
    with h5py.File(path, 'r') as f:
        data =  f['kspace'][()]
    return data

def load_mri_scan(path: pathlib.Path):
    mri_data = load_h5(path)
    mri_data = T.to_tensor(mri_data)      
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
    path = SRC_ROOT
    for file in path.glob('**/*.h5'):
        scan = load_mri_scan(file)
        normalized_scan = normalize_scan(scan)
        dest_path = DEST_ROOT / file.relative_to(SRC_ROOT)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(dest_path, 'w') as f:
            f.create_dataset('image_space', data=normalized_scan)