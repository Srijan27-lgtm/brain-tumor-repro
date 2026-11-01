import torch
import numpy as np
from torch.utils.data import Dataset

# Single Source of Truth for Labels
CLASS_TO_IDX = {
    'glioma': 0, 
    'meningioma': 1, 
    'pituitary': 3, 
    'no': 2, 
    'no_tumor': 2, 
    'notumor': 2, 
    'No Tumor': 2
}

class SartajDataset(Dataset):
    def __init__(self, hf_dataset, is_training=False):
        self.data = hf_dataset
        self.is_training = is_training
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        img = item['image']
        lbl = item['label']
        
        # 1. Robust conversion to numpy array (handles HF raw lists, PIL, etc.)
        if hasattr(img, 'convert'):
            img = np.array(img)
        elif isinstance(img, list):
            img = np.array(img, dtype=np.uint8)
            
        t = torch.tensor(img, dtype=torch.float32)
        
        # 2. Scale to [0, 1]
        if t.max() > 1.0:
            t = t / 255.0
            
        # 3. Permute to (C, H, W)
        if t.dim() == 3 and t.shape[-1] == 3:
            t = t.permute(2, 0, 1)
            
        # 4. Geometric Augmentation (Horizontal Flip) for training only
        if self.is_training and torch.rand(1) > 0.5:
            t = torch.flip(t, dims=[2])
            
        # 5. STRICT LABEL MAPPING (Crashes on unknown label to prevent silent failures)
        if isinstance(lbl, str):
            if lbl not in CLASS_TO_IDX:
                raise ValueError(f"CRITICAL: Unknown label '{lbl}' found in dataset!")
            lbl = CLASS_TO_IDX[lbl]
        else:
            lbl = int(lbl)
            
        return {'image': t, 'label': lbl}