from torch.utils.data import DataLoader
from dataset import MRIDataset
import matplotlib.pyplot as plt 

import torch 
import torch.nn as nn
from torchtyping import TensorType

class tiles(): 
    def __init__(self, k_h: int=32, k_w: int=32, s_h:int=16, s_w:int=16): 
        self.k_h = k_h # vertical kernel size
        self.k_w = k_w # horizontal kernel size 
        self.s_h = s_h # vertical stride size  
        self.s_w = s_w # horizontal stride size  
        
        self.b = 0 # batch size 
        self.n_h = 0 # height of MRI-images 
        self.n_w = 0 # width of MRI-images 
    
    def create_tiles(self, patch: TensorType[torch.float32, "b", "n_h", "n_w"]) -> TensorType[torch.float32, "b", "#h", "#w", "k_h", "k_w"]: 
        
        try:
            self.b, self.n_h, self.n_w = patch.shape
            assert ((self.n_h - self.k_h + self.s_h) % self.s_h) == 0, "Vertical kernel size or stride size is not compatible with image dimensions"
            assert ((self.n_w - self.k_w + self.s_w) % self.s_w) == 0, "Horizontal kernel size or stride size is not compatible with image dimensions"

            patches = patch.unfold(dimension=1 ,size=self.k_h, step=self.s_h).unfold(dimension=2, size=self.k_w, step=self.s_w)

        except AssertionError as msg: 
            print(msg)
        
        return patches 
    
    def recreate_image(self, patches: TensorType[torch.float32, "b", "#h", "#w", "k_h", "k_w"]) -> TensorType[torch.float32, "b", "n_h", "n_w"]: 
        
        fold = nn.Fold(output_size=(self.n_h, self.n_w), kernel_size=(self.k_h, self.k_w), stride=(self.s_h, self.s_w)) 
        patches = patches.contiguous().reshape(self.b, -1, self.k_h*self.k_w)
        patches = patches.permute(0, 2, 1)  
        patches = patches.contiguous().reshape(self.b, self.k_h*self.k_w, -1)
        reconstructed_sample = fold(patches).squeeze()
        
        return reconstructed_sample


train_dataset = MRIDataset(
    path='../dataset/brain/singlecoil_train', filter_func=(lambda x: 'FLAIR' in x), undersampled=False, number_of_samples = 10
)

train_loader = DataLoader(dataset=train_dataset, batch_size=4)
patch = next(iter(train_loader))
tiles = tiles()

patches = tiles.create_tiles(patch=patch)
reconstructed = tiles.recreate_image(patches=patches)

# plt.imshow(patches[0, 0, 0, :, :])
# print(patches.shape)
# plt.imshow(reconstructed[0])
plt.show()