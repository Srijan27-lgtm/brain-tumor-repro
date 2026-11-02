import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image

# Single Source of Truth for Labels (Matches both BRISC and Sartaj)
CLASS_TO_IDX = {
    'glioma': 0, 
    'meningioma': 1, 
    'pituitary': 3, 
    'no': 2, 
    'no_tumor': 2, 
    'notumor': 2, 
    'No Tumor': 2
}

class PretrainedDataset(Dataset):
    def __init__(self, hf_dataset, is_training=False):
        self.data = hf_dataset
        self.is_training = is_training
        
        # 1. Mandatory Preprocessing (Applied to ALL splits)
        # Paper: "Initially, each image was resized to a uniform dimension of 224x224 pixels."
        # Paper: "normalized using the standard ImageNet mean and standard deviation."
        base_transforms = [
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(), # Automatically scales to [0,1] and permutes to (C, H, W)
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], # CRITICAL: ImageNet Mean
                std=[0.229, 0.224, 0.225]   # CRITICAL: ImageNet Std
            )
        ]
        
        if self.is_training:
            # 2. On-the-fly Augmentations (Training ONLY)
            # Paper: "random horizontal flips, random rotations within a range of ±15 degrees, 
            #         and minor random translations... slight adjustments to brightness and contrast"
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                # RandomAffine elegantly handles BOTH rotation and minor translation
                # FIXED: Changed degrees from 10 to 15 to exactly match the paper
                transforms.RandomAffine(degrees=15, translate=(0.05, 0.05)), 
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                *base_transforms
            ])
        else:
            # 3. Validation/Test: ONLY mandatory preprocessing (NO augmentations)
            # Paper: "For the validation and test sets, no data augmentation was applied beyond 
            #         the initial, mandatory preprocessing steps of resizing and normalization."
            self.transform = transforms.Compose(base_transforms)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        img = item['image']
        lbl = item['label']
        
        # 4. Robust conversion to PIL Image (Handles HF raw lists, tensors, or actual PIL images)
        if isinstance(img, Image.Image):
            pil_img = img.convert('RGB')
        elif isinstance(img, list):
            # HF passes raw nested lists of pixels. Convert to numpy, then to PIL.
            img_array = np.array(img, dtype=np.uint8)
            pil_img = Image.fromarray(img_array).convert('RGB')
        elif isinstance(img, np.ndarray):
            pil_img = Image.fromarray(img).convert('RGB')
        else:
            raise TypeError(f"CRITICAL: Unsupported image type: {type(img)}")
            
        # 5. Apply Transforms
        t = self.transform(pil_img)
            
        # 6. STRICT LABEL MAPPING (Crashes on unknown label to prevent silent failures)
        if isinstance(lbl, str):
            if lbl not in CLASS_TO_IDX:
                raise ValueError(f"CRITICAL: Unknown label '{lbl}' found in dataset at index {idx}!")
            lbl = CLASS_TO_IDX[lbl]
        else:
            lbl = int(lbl)
            
        return {'image': t, 'label': lbl}
