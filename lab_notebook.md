# Lab Notebook: Brain Tumor Classification Reproduction

## Session: Day 1 - Data Pipeline & Preprocessing
**Date:** 2024-10-24 (Adjust to actual date)
**Goal:** Faithfully reproduce the paper's preprocessing pipeline for Table 3.

### 1. What I Tried
- Explored the BRISC 2025 dataset structure and `manifest.csv`.
- Filtered for `task == 'classification'` and `is_mask == False` to match the paper's exact 5,000 train / 1,000 test split.
- Applied the strict `timm` ImageNet pipeline step 1: `Resize(256)` using `BICUBIC` interpolation. *(Note: `CenterCrop(224)` and `Normalize` are intentionally deferred to the PyTorch DataLoader to match `timm`'s exact behavior).*
- Split the official 5,000 train images into 85% Train (4,250) and 15% Val (750) using `random_state=42` for checkpoint selection.
- Uploaded processed tensors to Hugging Face Hub to eliminate CPU bottlenecking on Vast.ai.

### 2. What I Expected
- Expected exactly 5,000 train and 1,000 test images.
- Expected the resize operation to take ~2-3 minutes on Kaggle's free CPU.

### 3. What Happened (Results)
- **Manifest Check:** Confirmed exactly 5,000 train and 1,000 test images. Classes: glioma, meningioma, no_tumor, pituitary.
- **Preprocessing:** Successfully resized 6,000 images from 512x512 to 256x256.
- **Hugging Face Upload:** Successfully uploaded to `srijan2715/brisc2025-paper-pipeline`.

### 4. Analysis / Why
- Using the official manifest prevents data leakage, a common pitfall in custom random splits.
- Resizing to 256 (not 224) is critical. If we resize directly to 224, the subsequent `CenterCrop(224)` does nothing, altering the spatial distribution of the data and violating faithful reproduction.

### 5. Next Steps
- Repeat exact same pipeline for Sartaj et al. dataset.
- Phase 2: Hand-calculate ResNet50 parameters and implement from-scratch CNN.
