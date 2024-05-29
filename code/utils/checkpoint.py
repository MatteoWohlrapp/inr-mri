import os
import torch 
import sys
from collections import OrderedDict

def load_dict(resume_path, model):
    if os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
        
        # Check if the saved model was wrapped with DataParallel, which prefixes 'module.'
        state_dict = checkpoint['state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k  # remove `module.` prefix if exists
            new_state_dict[name] = v
        
        # Load the adjusted state dict
        model.load_state_dict(new_state_dict, strict=False)
        
        del checkpoint  # cleanup to free memory
    else:
        sys.exit("=> No checkpoint found at '{}'".format(resume_path))
    return model