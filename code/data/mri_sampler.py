import os
import h5py
import numpy as np
import random
from fastmri.data.subsample import RandomMaskFunc
import fastmri
from fastmri.data import transforms as T


class MRIRandomSampler:
    def __init__(self, path, seed=42, filter_func=None, transform=None, test_files=None):
        self.path = path
        self.transform = transform
        random.seed(seed)  # Seed for reproducibility
        self.files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.endswith(".h5") and (filter_func is None or filter_func(f))
        ] if test_files is None else [os.path.join(path, f) for f in test_files]

    def get_random_sample(self):
        if not self.files:
            raise ValueError("No files available to sample from.")

        file_path = random.choice(self.files)

        with h5py.File(file_path, "r") as hf:
            kspace = np.asarray(hf["kspace"][:])
            slice_idx = random.randint(0, kspace.shape[0] - 1)
            kspace_tensor = T.to_tensor(
                kspace[slice_idx]
            )

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
