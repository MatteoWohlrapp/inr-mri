import os
import h5py
import numpy as np
import random
from fastmri.data.subsample import RandomMaskFunc
import fastmri
from fastmri.data import transforms as T

class MRIRandomSampler:
    def __init__(
        self, path, seed=42, filter_func=None, transform=None, test_files=None
    ):
        self.path = path
        self.transform = transform
        random.seed(seed)  # Seed for reproducibility
        self.files = (
            [
                os.path.join(path, f)
                for f in os.listdir(path)
                if f.endswith(".h5") and (filter_func is None or filter_func(f))
            ]
            if test_files is None
            else [os.path.join(path, f) for f in test_files]
        )

        self.available_slices = {}
        for file_path in self.files:
            with h5py.File(file_path, "r") as hf:
                num_slices = hf["kspace"].shape[0]
            self.available_slices[file_path] = list(range(num_slices))

    def get_random_sample(self):
        if not any(self.available_slices.values()):
            raise ValueError("No slices available to sample from.")

        available_files = [f for f in self.available_slices if self.available_slices[f]]
        file_path = random.choice(available_files)

        slice_idx = random.choice(self.available_slices[file_path])
        self.available_slices[file_path].remove(slice_idx)

        with h5py.File(file_path, "r") as hf:
            kspace = np.asarray(hf["kspace"][:])
            slice_idx = random.randint(0, kspace.shape[0] - 1)
            kspace_tensor = T.to_tensor(kspace[slice_idx])

        mask_func = RandomMaskFunc(center_fractions=[0.1], accelerations=[8])
        kspace_tensor_undersampled, _, _ = T.apply_mask(kspace_tensor, mask_func)

        original_image = self._kspace_to_image(kspace_tensor)
        undersampled_image = self._kspace_to_image(kspace_tensor_undersampled)

        if self.transform:
            original_image = self.transform(original_image)
            undersampled_image = self.transform(undersampled_image)

        filename = os.path.basename(file_path)
        filename = os.path.splitext(filename)[0]

        return original_image, undersampled_image, f"{filename}_slice{slice_idx}"

    def _kspace_to_image(self, kspace):
        image = fastmri.ifft2c(
            kspace
        )  # Apply Inverse Fourier Transform to get the complex image
        image_abs = fastmri.complex_abs(image)
        image_abs = image_abs.float()
        return image_abs

class MRIRandomSamplerTransformed:
    def __init__(self, path, target_height, target_width, seed=42, filter_func=None, transform=None, test_files=None):
        self.path = path
        self.target_height = target_height
        self.target_width = target_width
        self.transform = transform
        random.seed(seed)  # Seed for reproducibility
        self.files = (
            [
                os.path.join(path, f)
                for f in os.listdir(path)
                if f.endswith(".h5") and (filter_func is None or filter_func(f))
            ]
            if test_files is None
            else [os.path.join(path, f) for f in test_files]
        )

        self.available_slices = {}
        for file_path in self.files:
            with h5py.File(file_path, "r") as hf:
                num_slices = hf["undersampled"].shape[0]
            self.available_slices[file_path] = list(range(num_slices))

    def get_random_sample(self):
        while True:
            if not any(self.available_slices.values()):
                raise ValueError("No slices available to sample from.")

            available_files = [f for f in self.available_slices if self.available_slices[f]]
            file_path = random.choice(available_files)

            slice_idx = random.choice(self.available_slices[file_path])
            self.available_slices[file_path].remove(slice_idx)

            with h5py.File(file_path, "r") as hf:
                original_image = np.asarray(hf["fully_sampled"][slice_idx])
                undersampled_image = np.asarray(hf["undersampled"][slice_idx])

            # Ensure images are converted to tensors
            original_image = T.to_tensor(original_image).float()
            undersampled_image = T.to_tensor(undersampled_image).float()

            # Apply transformations if specified
            if self.transform:
                original_image = self.transform(original_image)
                undersampled_image = self.transform(undersampled_image)

            # Check if the images meet the required dimensions
            print(original_image.shape, flush=True)
            if original_image.shape[0] == self.target_height and original_image.shape[1] == self.target_width:
                filename = os.path.basename(file_path)
                filename = os.path.splitext(filename)[0]
                return original_image, undersampled_image, f"{filename}_slice{slice_idx}"
