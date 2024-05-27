import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Callable
import fastmri
from fastmri.data import transforms as T
from fastmri.data.subsample import RandomMaskFunc
from utils.time_keeper import time_function


class MRIDataset(Dataset):
    def __init__(
        self,
        path: str,
        filter_func: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        undersampled=True,
        number_of_samples=None,
    ):
        """
        Args:
            path: Path to files
            transform: Optional callable to apply to the data (e.g., normalization, augmentation).
        """
        self.path = path
        self.files = [
            os.path.join(path, f) for f in os.listdir(path) if f.endswith(".h5")
        ]
        self.transform = transform
        self.samples = []
        self.undersampled = undersampled
        self.number_of_samples = number_of_samples

        self._prepare_dataset(number_of_samples, filter_func)

    def _prepare_dataset(
        self, number_of_samples=None, filter_func: Optional[Callable] = None
    ):
        """Prepare the dataset by listing all file paths and the number of slices per file."""
        samples = 0
        for file_path in self.files:
            if (filter_func and filter_func(file_path)) or not filter_func:
                print(f"Reading file: {file_path}")
                with h5py.File(file_path, "r") as hf:
                    num_slices = hf["kspace"].shape[0]
                    for s in range(num_slices):
                        self.samples.append((file_path, s))
                        if number_of_samples:
                            samples += 1
                            if samples >= number_of_samples:
                                return

    def __len__(self):
        return len(self.samples)

    @time_function
    def __getitem__(self, idx):
        file_path, slice_idx = self.samples[idx]
        with h5py.File(file_path, "r") as hf:
            kspace = np.asarray(hf["kspace"][slice_idx])
            kspace_tensor = T.to_tensor(
                kspace
            )  # Convert from numpy array to pytorch tensor

            if self.undersampled:
                mask_func = RandomMaskFunc(center_fractions=[0.1], accelerations=[8])
                kspace_tensor, _, _ = T.apply_mask(kspace_tensor, mask_func)

            image = fastmri.ifft2c(
                kspace_tensor
            )  # Apply Inverse Fourier Transform to get the complex image
            image_abs = fastmri.complex_abs(image)

        image_abs = image_abs.float()

        # apply transformations
        if self.transform:
            image_abs = self.transform(image_abs)

        return image_abs


class MRIDatasetTransformed(Dataset):
    def __init__(
        self,
        path: str,
        filter_func: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        undersampled=True,
        number_of_samples=None,
    ):
        """
        Args:
            path: Path to files
            transform: Optional callable to apply to the data (e.g., normalization, augmentation).
        """
        self.path = path
        self.files = [
            os.path.join(path, f) for f in os.listdir(path) if f.endswith(".h5")
        ]
        self.samples = []
        self.undersampled = undersampled
        self.number_of_samples = number_of_samples

        self._prepare_dataset(filter_func)

    def _prepare_dataset(self, filter_func: Optional[Callable] = None):
        """Prepare the dataset by listing all file paths and the number of slices per file."""
        for file_path in self.files:
            if (filter_func and filter_func(file_path)) or not filter_func:
                with h5py.File(file_path, "r") as hf:
                    num_slices = hf["undersampled"].shape[0]
                    for s in range(num_slices):
                        self.samples.append((file_path, s))
                        if self.number_of_samples > 0:
                            self.number_of_samples -= 1
                        else:
                            return

    def __len__(self):
        return len(self.samples)

    @time_function
    def __getitem__(self, idx):
        file_path, slice_idx = self.samples[idx]
        with h5py.File(file_path, "r") as hf:
            if self.undersampled:
                image_abs = np.asarray(hf["undersampled"][slice_idx])
            else:
                image_abs = np.asarray(hf["fully_sampled"][slice_idx])
            image_abs = T.to_tensor(image_abs)
        image_abs = image_abs.float()
        return image_abs
