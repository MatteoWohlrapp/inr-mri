from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
from data.mri_dataset import MRIDataset
from utils.overlapping_tiling import tiles
import torch 

train_dataset = MRIDataset(
    path="../../../dataset/fastmri/brain/singlecoil_train_normalized", filter_func=(lambda x: 'FLAIR' in x), undersampled=False, number_of_samples = 10
)

train_loader = DataLoader(dataset=train_dataset, batch_size=1)
patch = next(iter(train_loader))
print(f"Shape of original patch: {patch.shape}")

tiles = tiles()
patches = tiles.create_tiles(patch=patch)
print(f"Shape of patches: {patches.shape}")
patches_shape = list(patches.shape)

patches_1 = patches.reshape((patches_shape[0]*patches_shape[1]*patches_shape[2], patches_shape[3], patches_shape[4])) # -> [patches, k_h, k_w]
print(f"Shape after reshape: {patches_1.shape}")
patches_2 = patches_1.reshape((patches_shape[0], patches_shape[1], patches_shape[2], patches_shape[3], patches_shape[4]))
print(f"Shape after re-reshape: {patches_2.shape}")
reconstructed, counter = tiles.recreate_image(patches=patches_2)

print(f"Shape of reconstructed image: {reconstructed.shape}")
#print(f"Shape of normalization mask: {counter.shape}")

plt.imshow(reconstructed)
plt.show()
