import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Callable
import fastmri
from fastmri.data import transforms as T
from fastmri.data.subsample import RandomMaskFunc
import random

def shuffle_list_with_seed(input_list, seed = None):
    if seed:
        random.seed(seed)
    random.shuffle(input_list)
    return input_list

class MRIDataset(Dataset):
    def __init__(
        self,
        path: str,
        filter_func: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        undersampled=True,
        number_of_samples=None
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
        image_width = 320, 
        image_height = 640,
        shuffle = False
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
        #for testing
        print('Size')
        print(len(self.files))
        """
        self.files = [file for file in self.files if 'FLAIR' in str(file)]
        print(len(self.files))"""
        self.shuffle = shuffle
        self.samples = []
        self.undersampled = undersampled
        self.number_of_samples = number_of_samples
        self.image_width = image_width
        self.image_height = image_height

        self._prepare_dataset(filter_func)

        #TODO into a function?
        if self.shuffle:
            self.samples = shuffle_list_with_seed(self.samples)

    def _prepare_dataset(self, filter_func: Optional[Callable] = None):
        """Prepare the dataset by listing all file paths and the number of slices per file."""
        count = 0
        for file_path in self.files:
            if (filter_func and filter_func(file_path)) or not filter_func:
                with h5py.File(file_path, "r") as hf:
                    print(hf.keys())
                    num_slices = hf["undersampled"].shape[0]
                    if hf["undersampled"].shape[1] == self.image_height and hf["undersampled"].shape[2] == self.image_width:
                        print("Reading file: ", file_path)
                        for s in range(num_slices):
                            self.samples.append((file_path, s))
                            if count < self.number_of_samples: 
                                count += 1 
                            else: 
                                return

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, slice_idx = self.samples[idx]
        with h5py.File(file_path, "r") as hf:
            if self.undersampled:
                image_abs = np.asarray(hf["undersampled"][slice_idx])
            else:
                image_abs = np.asarray(hf["fully_sampled"][slice_idx])
            image_abs = T.to_tensor(image_abs)
        image_abs = image_abs.float()
        #change norm from [-1,1] to [0,1] for testing
        image_abs =  image_abs + 1
        image_abs = image_abs / 2
        return image_abs


class MRIDatasetTransformedInMemory(Dataset):
    def __init__(
        self,
        path: str,
        filter_func: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        undersampled=True,
        number_of_samples=None,
        image_width = 320, 
        image_height = 640
    ):
        """
        Args:
            path: Path to the data files.
            transform: Optional callable to apply to the data (e.g., normalization, augmentation).
            undersampled: Flag to indicate if the dataset should use undersampled data or fully sampled data.
            number_of_samples: Limit the number of samples to load (useful for testing or restricted environments).
        """
        self.path = path
        self.transform = transform
        self.undersampled = undersampled
        self.number_of_samples = number_of_samples
        self.image_width = image_width
        self.image_height = image_height
        self.data = []

        self._prepare_dataset(filter_func)

    def _prepare_dataset(self, filter_func: Optional[Callable] = None):
        """Prepare the dataset by loading all the data into memory."""
        files = [
            os.path.join(self.path, f)
            for f in os.listdir(self.path)
            if f.endswith(".h5")
        ]
        count = 0
        for file_path in files:
            if filter_func and not filter_func(file_path):
                continue

            with h5py.File(file_path, "r") as hf:
                dataset_key = "undersampled" if self.undersampled else "fully_sampled"
                num_slices = hf[dataset_key].shape[0]
                if hf["undersampled"].shape[1] == self.image_height and hf["undersampled"].shape[2] == self.image_width:
                    print("Reading file: ", file_path)
                    for s in range(num_slices):
                        image_abs = np.asarray(hf[dataset_key][s])
                        image_tensor = T.to_tensor(image_abs).float()
                        if self.transform:
                            image_tensor = self.transform(image_tensor)
                        self.data.append(image_tensor)
                        if count < self.number_of_samples: 
                            count += 1 
                        else: 
                            return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MRIDatasetTransformedInMemoryBoth(Dataset):
    def __init__(
        self,
        path: str,
        filter_func: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        number_of_samples=None,
        image_width=320, 
        image_height=640
    ):
        """
        Args:
            path: Path to the data files.
            transform: Optional callable to apply to the data (e.g., normalization, augmentation).
            number_of_samples: Limit the number of samples to load (useful for testing or restricted environments).
            image_width: Expected width of the images.
            image_height: Expected height of the images.
        """
        self.path = path
        self.transform = transform
        self.number_of_samples = number_of_samples
        self.image_width = image_width
        self.image_height = image_height
        self.data = []

        self._prepare_dataset(filter_func)

    def _prepare_dataset(self, filter_func: Optional[Callable] = None):
        """Prepare the dataset by loading all the data into memory."""
        files = [
            os.path.join(self.path, f)
            for f in os.listdir(self.path)
            if f.endswith(".h5")
        ]
        count = 0
        for file_path in files:
            if filter_func and not filter_func(file_path):
                continue

            with h5py.File(file_path, "r") as hf:
                num_slices = hf['undersampled'].shape[0]
                if hf['undersampled'].shape[1] == self.image_height and hf['undersampled'].shape[2] == self.image_width:
                    print("Reading file: ", file_path)
                    for s in range(num_slices):
                        image_us = np.asarray(hf['undersampled'][s])
                        image_fs = np.asarray(hf['fully_sampled'][s])
                        image_tensor_us = T.to_tensor(image_us).float()
                        image_tensor_fs = T.to_tensor(image_fs).float()
                        if self.transform:
                            image_tensor_us = self.transform(image_tensor_us)
                            image_tensor_fs = self.transform(image_tensor_fs)
                        self.data.append((image_tensor_us, image_tensor_fs))
                        if self.number_of_samples is not None and count >= self.number_of_samples:
                            return
                        count += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
