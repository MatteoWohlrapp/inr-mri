import torch 
import torch.nn as nn
from torch.nn import functional as F
from torchtyping import TensorType

def create_tiles(patch: TensorType[torch.float32, "b", "n_h", "n_w"], k_h: int=32, k_w: int=32, s_h:int=32, s_w:int=32) -> TensorType[torch.float32, "b", "#h", "#w", "k_h", "k_w"]: 
        try:
            b, n_h, n_w = patch.shape
            assert ((n_h - k_h + s_h) % s_h) == 0, "Vertical kernel size or stride size is not compatible with image dimensions"
            assert ((n_w - k_w + s_w) % s_w) == 0, "Horizontal kernel size or stride size is not compatible with image dimensions"

            patches = patch.unfold(dimension=1 ,size=k_h, step=s_h).unfold(dimension=2, size=k_w, step=s_w)

        except AssertionError as msg: 
            print(msg)
        return b, n_h, n_w, k_h, k_w, s_h, s_w, patches
    
def recreate_image(b, n_h, n_w, k_h, k_w, s_h, s_w, patches) -> TensorType[torch.float32, "b", "n_h", "n_w"]: 
        
    fold = nn.Fold(output_size=(n_h, n_w), kernel_size=(k_h, k_w), stride=(s_h, s_w))
    patches = patches.contiguous().reshape(b, -1, k_h*k_w)
    patches = patches.permute(0, 2, 1)  
    patches = patches.contiguous().reshape(b, k_h*k_w, -1)
    reconstructed_sample = fold(patches).squeeze()
    counter = fold(torch.ones_like(patches)).squeeze().squeeze() # For possibly necessary normalization

    return reconstructed_sample, counter