# MRI Brain Tumor Classification: Model Reproduction & Evaluation + Scratch CNN

## Project Development Log and Experimental Results

**Project:** NextSwin-X Reproduction and Comparative Evaluation  
**Datasets:** BRISC and Sartaj  
**Models:** TwoStream, Swin Transformer, EfficientNetV2-B3, BiFNN, ConvNeXt-Base, NeXtSwin-X, ScratchCNN  
**Random Seeds:** 42, 43, 44  
**Reporting Period:** 23 October – 4 November 2026

---

# 1. Dataset Processing & Pipeline Fixes

**Dates:** 23/10/2026 – 25/10/2026 (till morning)

**Objective:** Process BRISC and Sartaj datasets for NextSwin-X reproduction.

## Fix 1: BRISC Aspect Ratio Issue

**What I tried:**

I used `img.resize((256, 256))` on the BRISC dataset.

**What I expected:**

I expected all BRISC images to uniformly resize to 512 × 512 and then crop cleanly.

**What actually happened:**

After inspecting `manifest.csv`, I found 659 non-square images with varying widths ranging from 216 px to 1,365 px.

My initial approach would have squashed these images, potentially altering tumor morphology.

**Fix and decision:**

I re-read the paper, which explicitly states:

> "Initially, each image was resized to a uniform dimension of 224x224 pixels."

The paper does not mention aspect ratio preservation, padding, or cropping.

To ensure methodological faithfulness, I updated the pipeline to directly resize all images to `(224, 224)` using BICUBIC interpolation.

I accepted the distortion of the 659 non-square images because this was the most faithful interpretation of the authors' exact words without introducing unverified assumptions.

**Note:** I documented this decision to explore potential post-reproduction improvements, such as better aspect-ratio handling for stress testing, once the baseline reproduction is secured.

## Fix 2: Sartaj Data Leakage and Duplicates

**What I tried:**

I assumed the official Kaggle Train/Test split for Sartaj was perfectly cleaned and used it directly.

**What I expected:**

I expected no overlap between the Training and Test sets.

**What actually happened:**

I performed SHA256 hash checks on the raw folders and found 127 images that existed in both Training and Test sets, along with internal duplicates.

**Fix and decision:**

I re-read the NextSwin paper. It explicitly states that the Sartaj dataset consists of exactly 3,264 MRI images and does not mention any data cleaning.

To ensure faithful reproduction of the paper's Table 3, I retained the official, unmodified Kaggle split and documented this transparently.

**Note:** I documented this to explore better data-cleaning pipelines after securing the reproduction baseline.

## Fix 3: Label Mapping Crash and Integrity

**What I tried:**

I uploaded the dataset to Hugging Face with string labels and a dictionary looking for `no_tumor`.

**What I expected:**

I expected a smooth upload with correct labels.

**What actually happened:**

The upload crashed with an `ArrowInvalid` error.

My string-cleaning operation converted the folder name `no_tumor` into `no`, but my dictionary still looked for `no_tumor`.

The script skipped images but continued counting them, breaking the image and label array lengths.

**Fix and decision:**

- Updated the dictionary to `{"no": 2}`.
- Added a guard clause to skip unknown labels before processing.
- Added `failed_count` tracking.
- Added an assertion to guarantee zero silent failures.
- Saved the mapping to `label_mapping.json`.
- Uploaded the mapping file to both Hugging Face repositories.

## Fix 4: Paper Preprocessing & Augmentation

**What I tried:**

I used standard timm defaults:

`Resize(256) → CenterCrop(224)`

**What I expected:**

I expected standard preprocessing to match the paper's baseline.

**What actually happened:**

After re-reading the NextSwin-X paper, I found that the authors explicitly state:

> "Initially, each image was resized to a uniform dimension of 224x224 pixels."

They do not mention an intermediate 256 resize or CenterCrop.

**Fix and decision:**

I changed the pipeline to directly resize all images to 224 × 224 using BICUBIC interpolation.

Because the images were resized to 224 × 224, the paper's on-the-fly training augmentation would now apply to the resized images rather than the original images.

**Note:** I documented this to explore better augmentation techniques after reproduction.

---

# 2. ScratchCNN Parameter Verification

**Dates:** 26/10/2026 – 30/10/2026

**Objective:** Verify every layer's parameter count in the ScratchCNN architecture before training.

**What I tried:**

I performed hand calculations for every layer's parameter count to verify the ScratchCNN architecture and ensure there was no mismatch before training.

**What I expected:**

The hand-calculated total should perfectly match the `torchinfo` verification script output.

**What actually happened:**

My initial hand calculation resulted in 98,308 parameters, which was 128 parameters fewer than the `torchinfo` result of 98,436.

**Fix and decision:**

I realized I had made a small mistake in the convolution part of Block 4 by not adding the bias (output channel) in the formula.

I corrected the hand calculation to include the bias, which perfectly matched the 98,436 total.

## Issues and Technical Notes

### Bias Issue

Missing the bias addition leads to a lower parameter count, as I experienced during the initial calculation.

### Flatten vs. Global Average Pooling (GAP)

If I had used `flatten()` directly on the `128 × 14 × 14` tensor, it would have resulted in 25,088 input features.

This would inflate the total final linear-layer parameter count to over 100,000 parameters, potentially causing overfitting.

GAP reduced this to exactly 128 features, keeping the total parameter count at 98,436.

### BatchNorm Issue

If I had not accounted for the two learnable parameters (scale and shift) in BatchNorm, parameters would have been missing from the final count.

### Backpropagation Verification

I performed hand calculations to understand the backpropagation algorithm.

I initially made mistakes in a small convolution-layer example, including incorrect derivative calculations and misapplication of the chain rule.

Upon recalculating, I found that:

- The actual output was a 2 × 2 matrix.
- The nine output pixels were the dot products of the filter as it moved across the input matrix.
- I recalculated the gradients individually.
- The product for the \(w_{11}\) weight, which was used four times, gave a total gradient of 72 for that run.

I verified the same tensor using a PyTorch script, and the result matched correctly.

---

# 3. Sanity Checks & Environment Setup

**Dates:** 30/10/2026 – 31/10/2026

## 3.1 Initial 100-Image Sanity Check

**What I tried:**

I ran a test on 100 images to perform a sanity check of the data pipeline and training loop.

**What I expected:**

I expected it to run smoothly and give 100% accuracy, representing an overfitting condition because the dataset was too small.

**What actually happened:**

The script initially produced multiple errors. After fixing them, it ran smoothly and achieved:

- Accuracy: 99%
- Loss: 0.189

### Fixes Implemented

| Issue | Fix |
|---|---|
| PIL Crash | Hugging Face stored raw pixel lists instead of PIL images, so I switched to direct tensor conversion. |
| Reshaping | Reshaped flat pixel lists into proper 3D image grids. |
| Label Casting | Explicitly cast text-string labels into integers before creating tensors. |
| Label Mapping | Mapped actual text labels, such as `glioma`, to numerical values. |
| Workers | Set `num_workers=0` to stop silent background-worker crashes in Kaggle. |
| Channel Flipping | Flipped image dimensions to put color channels first. |
| Normalization | Divided pixel values by 255 to stabilize the calculations. |

**Sanity Check Result:** 99% accuracy with 0.189 loss.

---

# 4. E2E Networks Setup & Initial Sartaj Training

## 4.1 E2E Environment Setup

**What I tried:**

I selected E2E Networks for computation after analyzing what was feasible, selected an NVIDIA A30 GPU, and started setting up the environment for full-scale training of my ScratchCNN and other models.

**What I expected:**

I expected that after some settings and verifications, training would start smoothly.

**What actually happened:**

Setting up the system took considerable time and involved multiple errors. I fixed them and started training.

### Fixes Implemented

1. The bare Ubuntu server logged in directly as root, so I removed unnecessary `sudo` commands and ran `apt` directly.
2. Hugging Face packages were not being found on the PyTorch server, so I split package installations into two separate commands.
3. Ubuntu 24.04 blocked modifications to system packages, so I used the `--ignore-installed idna` flag to force installation.
4. Terminal login kept timing out and hitting rate limits, so I temporarily hardcoded the authentication token directly into the Python script.
5. Pasting long code blocks into the nano editor caused automatic line wrapping and broke Python indentation, so I refactored the code into shorter, multiline blocks.
6. The script crashed on minor typos, such as `ln` instead of `len` and `dtypetorch` instead of `dtype`. I used the Linux `sed` command to patch the file directly in the terminal.
7. The script completed BRISC but crashed when searching for the Sartaj dataset because the repository name was slightly incorrect. I updated the script to use the exact Hugging Face repository ID.

**Security update:** The authentication token exposed during debugging was deleted and replaced with a newly generated token.

## 4.2 Sartaj Epoch-2 Mode Collapse

During the initial Sartaj training runs, the models were reaching 100% training accuracy by Epoch 2.

However, evaluation showed that they were predicting `glioma` for every single image.

**Root cause:**

A silent bug in the data pipeline was mapping all labels to index 0.

**Fixes implemented:**

- Rewrote the data transformation to explicitly map every label variation.
- Added a pre-flight check to verify the label distribution before training.
- Added an emergency stop clause so that if accuracy reaches 99.9% too early, the training run terminates instead of wasting GPU credits.

The three seeds were then retrained on E2E Networks using the corrected pipeline.

---

# 5. V2 Pipeline & Cross-Platform Validation

**Dates:** 31/10/2026 – 01/11/2026

## 5.1 Sartaj V2 Training

**What I tried:**

I retrained the Sartaj dataset using the new V2 pipeline with geometric augmentations, including flips and rotations.

**What I expected:**

The model should learn robust features without reaching 100% training accuracy through mode collapse.

**What actually happened:**

The emergency stop never triggered.

All three seeds (42, 43, 44) successfully trained to approximately 85–88% accuracy.

The augmentations prevented memorization, and explicit label mapping fixed the silent data drop.

**Fix and decision:**

I securely uploaded the validated weights using the naming convention:

`sartaj_seed{seed}_v2.pth`

The broken V1 files were permanently deleted from the active repository.

## 5.2 Evaluation Script Errors

**What I tried:**

I ran the evaluation script on Kaggle to recover metrics and compare Official vs. Clean test sets.

**What I expected:**

A smooth execution yielding the final metrics.

**What actually happened:**

The script crashed on BRISC with:

`invalid literal for int(): 'pituitary'`

Sartaj accuracy also silently dropped by approximately 20%.

**Root causes:**

1. Hugging Face was returning PIL Images, and `torch.tensor()` was corrupting them.
2. The label-mapping logic was attempting to evaluate `int()` directly on strings.

**Fixes implemented:**

- Used `np.array(img)` for image conversion.
- Corrected the label-mapping logic.
- Retained `num_workers=0`.

## 5.3 Initial ScratchCNN Results

After correcting the evaluation pipeline, the following results were obtained.

### BRISC Dataset

| Seed | Training Accuracy | Test Accuracy |
|---|---:|---:|
| 42 | 88.19% | 80.60% |
| 43 | 87.41% | 85.60% |
| 44 | 87.44% | 78.50% |

**My prediction:** Approximately 86–87% Training Accuracy and 83–84% Test Accuracy.

The prediction was reasonably close to the observed results.

The dataset contained 4,250 images, allowing the 98k-parameter model to learn features with a relatively healthy generalization gap across the three seeds.

### Sartaj Dataset

| Seed | Training Accuracy | Official Test Accuracy | Clean Test Accuracy |
|---|---:|---:|---:|
| 42 | 85.57% | 50.76% | 48.85% |
| 43 | 87.45% | 35.03% | 16.09% |
| 44 | 88.11% | 53.05% | 26.44% |

**My prediction:** 100% Training Accuracy by Epoch 2 and approximately 80% Test Accuracy.

**Actual verified results:** Training accuracy reached 85–88%, while Official Test accuracy ranged from 35–53% and Clean Test accuracy ranged from 16–48%.

The V2 augmentations successfully stopped the previous 100% training crash, but the large generalization gap remained.

Initially, I attributed this to the 98k-parameter model lacking the capacity to generalize on a small and complex dataset containing 2,439 images.

This initial explanation was subsequently reconsidered and retracted during the later Sartaj anomaly investigation.

## 5.4 Cross-Platform Verification

To ensure the low Sartaj metrics were not caused by a Kaggle execution issue, I ran the exact same evaluation script on the bare-metal E2E A30 GPU.

| Environment | Seed 42 Official Test | Seed 42 Clean Test |
|---|---:|---:|
| Kaggle | 50.76% | 48.85% |
| E2E A30 | 50.76% | 48.85% |

The metrics matched exactly across both environments.

This confirmed that the observed evaluation results were not simply a compute-environment artifact.

---

# 6. Sartaj Debugging & Pipeline Verification

**Dates:** 01/11/2026 – 02/11/2026

## 6.1 Final Seed 42 Checkpoint

**What I tried:**

I executed a clean, end-to-end script to replace the corrupted checkpoint.

**What I expected:**

The model should train smoothly and reach approximately 80% validation accuracy without the 100% Epoch-2 mode collapse.

**What actually happened:**

The model trained successfully.

- Early stopping triggered at Epoch 11.
- The best weights were obtained at Epoch 6.
- Best validation accuracy: 80.05%.

**Fix and decision:**

Saved the verified checkpoint as:

`sartaj_seed42_FINAL.pth`

Additional training artifacts were saved:

- `training_log.csv`
- `training_curves.png`

## 6.2 Checkpoint Authenticity Verification

**What I tried:**

I wrote an authenticity verification script to download the newly uploaded `sartaj_seed42_FINAL.pth` from Hugging Face and test it on the first 500 training images.

**What I expected:**

Accuracy greater than 75% to verify that the checkpoint represented the converged model rather than the previously corrupted 62.4% checkpoint.

**What actually happened:**

The script returned exactly:

**81.00% Training Accuracy**

**Fix and decision:**

This verified that the uploaded checkpoint contained the expected trained model weights and that the previous 13.79% collapse was associated with a corrupted checkpoint artifact.

The 500-image result was documented as a checkpoint verification result, not proof that every part of the pipeline was flawless.

## 6.3 Forensic Checkpoint Archiving

**What I tried:**

I wrote a script to move four broken or unusable checkpoints, including `sartaj_seed42_GOOD.pth` and V2 files, from the main repository to a new forensic archive repository named `ExtraRuns_checkpoints`.

**What I expected:**

A smooth transfer that would preserve the debugging history without cluttering the production repository.

**What actually happened:**

The script failed with a 401 Client Error due to an invalid password or missing repository access caused by a missing or invalid Kaggle environment variable.

**Fix and decision:**

Created a separate repository for experimental runs that were no longer in use or could not be used.

## 6.4 Hugging Face Token Security

During the debugging phase, hardcoded Hugging Face tokens were used in multiple scripts to bypass rate limits and environment errors.

The tokens were exposed in chat history and notebook cells, creating a security vulnerability.

**Fix and decision:**

- Deleted the exposed token.
- Generated a new token.

---

# 7. Sartaj Anomaly Investigation

**Dates:** 02/11/2026 – 03/11/2026

## 7.1 Reconsidering the Generalization Gap

**What I tried:**

I analyzed the Seed 43 metrics:

- Training Accuracy: 87.45%.
- Official Test Accuracy: 35.03% (394 images).
- Clean Test Accuracy: 16.09% (174 images).

**What I expected:**

The model should show standard overfitting, with high training accuracy and moderately lower test accuracy.

**What actually happened:**

Mathematical deduction revealed an inconsistency.

If the Official Test contains 394 images and the Clean Test contains 174 images, the difference is 220 images.

Calculating the accuracy on these 220 images yielded approximately 50%.

However, these same images were present in the training set, where the model achieved approximately 87% accuracy.

Furthermore, 16% accuracy on the Clean Test was below the 25% random-guess baseline for four classes.

This indicated that the earlier explanation based solely on insufficient model capacity was scientifically incorrect.

Low model capacity generally causes underfitting, whereas this model demonstrated high training accuracy.

Since the same architecture achieved approximately 80–85% test accuracy on BRISC, the Sartaj problem required further investigation into dataset and pipeline inconsistencies.

## 7.2 Four-Step Debugging Protocol

### Step 1: Preprocessing Mismatch Check

I wrote a script to evaluate the Training split using the exact same evaluation pipeline while keeping the model in `model.eval()` mode.

Expected interpretation:

- If accuracy drops to approximately 50%, it indicates a preprocessing mismatch.
- If accuracy remains near 87%, the evaluation pipeline is more likely consistent, requiring further investigation of the data.

I planned to align the `PretrainedDataset` transformations to ensure that evaluation applies the same normalization as the training loop, including `/255` versus ImageNet mean/std normalization.

### Step 2: Confusion Matrix Analysis

I generated confusion matrices and prediction distributions for the Sartaj test splits.

The investigation focused on identifying possible class bias, such as predicting Glioma for nearly every image.

I also planned to investigate class imbalance and implement Class-Weighted Cross Entropy Loss.

### Step 3: Label Mismatch Check

I extracted the 220 overlapping images and programmatically compared their assigned labels in the Training and Test sets.

The objective was to identify images where:

`train_label != test_label`

If the same image was assigned different classes across the splits, it would indicate contradictory ground-truth annotations in the original Kaggle dataset.

### Step 4: Perceptual Hashing Upgrade

The initial SHA256 check identified only 64 duplicates, while the mathematical deduction suggested a possible 220-image overlap.

The initial approach used strict byte-level hashing, which cannot detect visually identical images that have been resized, compressed, or had their metadata modified.

I therefore investigated upgrading the duplicate-detection procedure using:

- Perceptual Hashing (pHash).
- Structural Similarity (SSIM).

## 7.3 Retraction of the Initial Model-Capacity Explanation

After further investigation, I retracted the earlier claim that the ScratchCNN lacked sufficient capacity as the primary explanation for the low Sartaj accuracy.

**Updated findings:**

- Retracted the insufficient-model-capacity claim in the README.
- Identified a corrupted checkpoint, label-0 mapping bug, and preprocessing mismatch during the investigation.
- Updated `Sartaj_findings.md` to clarify that the 500-image 81% accuracy verifies checkpoint behavior, not that the entire pipeline is flawless.
- Regenerated the model using explicit label mapping and geometric augmentations.
- Successfully stopped the 100% Epoch-2 mode-collapse behavior.

## 7.4 ResNet50 Parameter Verification

I also hand-calculated the first ResNet50 bottleneck layer's parameter count.

The calculation matched the `torchinfo` output, and I proceeded to the next step.

---

# 8. Multi-Model Batch Training

**Dates:** 02/11/2026 – 03/11/2026

## 8.1 Initial Batch Training

**What I tried:**

I executed sequential batch training for six models on Kaggle T4 GPUs:

1. ResNet50
2. Swin Transformer
3. ConvNeXt
4. TwoStream
5. NeXtSwin-X
6. BiFNN / EfficientNet

**What I expected:**

Smooth, uninterrupted execution from start to finish.

**What actually happened:**

I encountered:

- GPU internal faults.
- Script crashes during checkpoint uploads.
- Upload failures incorrectly logged as `SUCCESS` with exit code 0.
- Redundant retraining of already completed runs.
- timm library errors for EfficientNet.

## 8.2 Pipeline Upgrades

Four critical pipeline improvements were implemented.

### 1. Checkpoint Rule

Standardized all scripts to save checkpoints based on the highest `best_val_acc`, rather than the last training epoch.

This ensured that `_best.pth` files represented the best validation performance.

### 2. Fail-Safe Uploads

Added `sys.exit(1)` on upload failure.

If the network drops, the script now halts and records a genuine failure instead of silently reporting success.

### 3. EfficientNet Library Patch

Corrected the timm model name:

`efficientnetv2_b3`

to:

`tf_efficientnetv2_b3`

This resolved the Unknown Model error.

### 4. Smart Skipping

Added an `api.file_exists` check at the beginning of the batch loop.

If a checkpoint already exists on Hugging Face, the script skips that experiment.

Additional execution improvements:

- Used `tmux` and `nohup` to prevent SSH disconnections from terminating runs.
- Split the 36 total runs into two batches of 18 to accommodate GPU availability constraints.

## 8.3 Dual-Environment Execution

I deployed a dual-environment strategy:

- Heavy model training was performed on Kaggle T4 GPUs in isolated subprocesses.
- A local CMD script was simultaneously used to bulk-upload previously trained checkpoints, including ResNet, Swin, ConvNeXt, and other models, to Hugging Face.

**Current status at this stage:**

- Approximately 90% of the 42 planned runs were successfully uploaded.
- EfficientNet training was still running in the background on Kaggle.
- The final evaluation script was being executed on downloaded checkpoints.
- The objective was to integrate all metrics and generate a custom Table 3.

---

# 9. Final Evaluation & Architectural Fixes

**Dates:** 03/11/2026 – 04/11/2026

## 9.1 Final Evaluation Script Failures

**What I tried:**

I executed the final paper-faithful evaluation script to generate the Master Results Table for the remaining models:

- Swin Transformer.
- NeXtSwin-X.
- TwoStream BRISC.
- BiFNN.

**What I expected:**

A smooth execution yielding the final Mean ± Standard Deviation metrics.

**What actually happened:**

Three distinct critical errors occurred.

### Issue 1: timm Recursion Crash

To fix the Unknown Model error (`efficientnetv2_b3`) for BiFNN, I initially attempted a runtime monkey-patch on `timm.create_model`.

This created an infinite loop in the library registry, resulting in a maximum recursion depth exceeded error.

The Kaggle kernel's RAM became unstable, and evaluation crashed for every model.

### Issue 2: Architecture Mismatches

Swin Transformer and NeXtSwin-X produced:

- Runtime size mismatch errors.
- Missing keys in `state_dict`.
- Checkpoint incompatibility.

The checkpoints had been saved using different base configurations, such as Swin-Small, while the evaluation script was initializing Swin-Base.

### Issue 3: Missing Checkpoints

The TwoStream BRISC files failed to load because the upload script had appended a `_best.pth` suffix that the evaluation paths were not checking for.

## 9.2 Fixes and Decisions

### Abandoned Runtime Patching

I realized that monkey-patching deep-learning libraries at runtime was unstable.

I wrote a script to directly edit the `models_baselines.py` source code on disk, replacing `efficientnetv2_b3` with the correct `tf_efficientnetv2_b3`.

I also performed a hard restart of the Kaggle kernel to clear the corrupted timm registry from memory.

### Tiered Checkpoint Loading

Implemented a nested try/except loading mechanism.

The evaluation script:

1. Attempts `strict=True` first.
2. Falls back to `strict=False` for minor naming inconsistencies.
3. Gracefully skips fundamental architecture mismatches with diagnostic logging.

### Path Fallbacks

Added an automatic fallback in the evaluation loop to check for the `_best.pth` suffix if the standard filename is not found.

**Outcome:**

These engineering fixes stabilized the pipeline, allowing BiFNN and TwoStream BRISC to evaluate successfully and completing the data required for the final Master Table.

## 9.3 NeXtSwin-X Feature Dimension Failure

**What I tried:**

I evaluated NeXtSwin-X, but it crashed with:

`Given normalized_shape=[1024], expected input with shape [*, 1024], but got input of size [8, 1024, 1, 1]`

**What I expected:**

The model should load and evaluate smoothly like the other baselines.

**What actually happened:**

Depending on the timm library version, the ConvNeXt backbone was outputting a four-dimensional spatial tensor, such as `[8, 1024, 1, 1]`, or sometimes NHWC format, instead of a flattened two-dimensional tensor `[Batch, Channels]`.

The internal LayerNorm could not process this representation as expected.

**Fix and decision:**

I abandoned standard `model(x)` calls.

I rewrote the forward pass to use:

`self.cnn_backbone.forward_features(x)`

This bypassed the timm internal head entirely.

I then implemented a custom `_robust_flatten()` method that dynamically detects whether the 1024 features are located at dimension 1 or dimension -1 and averages the spatial dimensions accordingly.

This guaranteed a consistent `[Batch, 1024]` output regardless of the library version, allowing all three seeds to evaluate successfully.

---

# 10. DINOv2 Advanced Analysis

## 10.1 Experimental Strategies

**What I tried:**

I implemented a multi-stage evaluation of the DINOv2 ViT-B/14 backbone.

The experiments included:

1. Linear probing.
2. MLP / ConvNeXt-style classification heads.
3. Partial fine-tuning by unfreezing the final two transformer blocks.

**What I expected:**

I expected fine-tuning to match or exceed frozen-baseline performance by allowing the backbone to adapt to medical imaging features.

**What actually happened:**

Fine-tuning the final two blocks with a low learning rate of `1e-5` yielded the best stability, preventing catastrophic forgetting while adapting to tumor morphology.

## 10.2 Interpretability Experiments

### PCA Visualization

I visualized 768-dimensional patch tokens projected into three-dimensional RGB space.

The visualizations revealed that DINOv2 naturally clusters features according to anatomical regions without explicitly using labels.

### Grad-CAM Analysis

I contrasted CNN texture bias with ViT shape bias.

- CNN visualizations focused on local tumor boundaries and textures.
- ViT visualizations attended to broader anatomical context and global anatomical displacement.

### Disagreement Analysis

I identified specific images where CNN predictions were correct but DINOv2 predictions failed.

One hypothesis was that MRI artifacts and noise confuse the ViT's global attention mechanism, while the CNN's local receptive field remains more robust.

---

# 11. Final Reproduced Results: Master Table 3

The final stabilized evaluation script was executed across all seven models, two datasets, and three random seeds.

**What I expected:**

A complete set of Mean ± Standard Deviation metrics to finalize the reproduction study and compare against the original paper.

**What actually happened:**

The evaluation pipeline completed after the architectural fixes.

The baseline models, including TwoStream, Swin, and EfficientNet, matched or exceeded the original paper's metrics.

NeXtSwin-X was successfully evaluated across all three seeds, with the observed variance reflecting the checkpoint-format inconsistencies investigated during debugging.

**Fix and decision:**

The final results were formatted into a Markdown table.

All values represent Mean ± Standard Deviation across seeds 42, 43, and 44.

## 11.1 Final Results

| Model | BRISC Accuracy | BRISC F1 | Sartaj Accuracy | Sartaj F1 |
|---|---:|---:|---:|---:|
| TwoStream (ResNet50) | 99.20 ± 0.22 | 99.20 ± 0.22 | 78.51 ± 0.43 | 75.07 ± 0.38 |
| Swin-Transformer | 99.20 ± 0.14 | 99.20 ± 0.14 | 78.34 ± 1.02 | 75.24 ± 1.40 |
| EfficientNetV2-B3 | 99.10 ± 0.22 | 99.10 ± 0.22 | 76.82 ± 1.21 | 73.67 ± 1.88 |
| BiFNN (ResNet50) | 98.53 ± 0.65 | 98.53 ± 0.66 | 75.13 ± 1.56 | 71.47 ± 1.22 |
| ConvNeXt-Base | 98.43 ± 0.05 | 98.43 ± 0.05 | 77.07 ± 0.73 | 73.33 ± 0.87 |
| NeXtSwin-X (Proposed) | 91.17 ± 3.92 | 91.04 ± 3.96 | 74.11 ± 5.39 | 70.76 ± 7.63 |
| ScratchCNN (Ours) | 83.53 ± 1.15 | 83.63 ± 1.13 | 47.21 ± 0.55 | 44.49 ± 0.71 |

---

# 12. Analysis of Reproduced Results

## 12.1 Why Our Baselines Outperformed the Paper

### Superior Pretrained Weights

We utilized modern timm defaults, which often include models pretrained on ImageNet-21K (21 million images) rather than the standard ImageNet-1K used in older papers.

This provides a richer feature initialization.

### Modern Optimization & Augmentation

The timm library continuously updates its default training recipes.

These may include:

- AdamW with optimized weight decay.
- Cosine annealing.
- Advanced augmentations such as RandAugment.

Our reproduction benefited from modern optimization configurations that may not have been available or used by the original authors.

**Result:**

Models such as TwoStream and EfficientNet gained approximately 7–10% accuracy, demonstrating the performance of the baseline architectures under the implemented training configurations.

## 12.2 timm Version Mismatch & Feature Distribution Shift

The original paper was trained using a specific, older version of the timm library.

During reproduction, newer timm versions returned four-dimensional spatial tensors, such as `[8, 1024, 1, 1]`, or NHWC representations instead of flattened two-dimensional tensors.

To prevent crashes, we engineered a workaround using `forward_features()` and a custom `_robust_flatten()` method.

While this fixed the evaluation process, manual spatial pooling may alter the feature distribution that the cross-attention mechanism was originally trained to expect.

## 12.3 Checkpoint Inconsistencies (`strict=False`)

Due to minor naming inconsistencies in saved state dictionaries across certain seeds, particularly Sartaj Seed 44, we used `strict=False` loading.

This means that a subset of layers, such as specific normalization weights, may not have been restored from the trained checkpoint and may instead have retained their initialized values.

This could degrade the fusion mechanism's effectiveness.

## 12.4 Undocumented Training Hyperparameters

Complex fusion models such as NeXtSwin-X are sensitive to hyperparameter configurations, including:

- Exact learning-rate ratios between backbone and attention head.
- Weight decay.
- Custom loss weighting.

Without the original training code, we relied on standard configurations that may not be optimal for custom cross-attention architectures.

## 12.5 Strict 224 × 224 Resizing Constraint

The paper explicitly states:

> "Initially, each image was resized to a uniform dimension of 224x224 pixels."

We strictly enforced this using direct BICUBIC resizing to `(224, 224)`.

We deliberately avoided standard practices such as `Resize(256) → CenterCrop(224)` because the paper did not mention them.

Although this ensured methodological faithfulness to the available description, it may have squashed the 659 non-square BRISC images.

This could negatively affect models that rely heavily on precise spatial geometry, such as Vision Transformers and cross-attention mechanisms.

---

# 13. Conclusion

Our reproduction study successfully validated the strong performance of the baseline architectures, often exceeding the original paper's reported results under the implemented training configurations.

The underperformance of the proposed NeXtSwin-X model was associated with documented timm library version mismatches requiring architectural workarounds, checkpoint inconsistencies, and the sensitivity of cross-attention mechanisms to feature distributions.

These observations highlight an important challenge in deep-learning reproducibility: without the exact original training environment, implementation, and hyperparameters, complex custom architectures can be difficult to reproduce precisely.

The study also involved extensive dataset processing, parameter verification, training-pipeline debugging, checkpoint validation, cross-platform testing, and advanced DINOv2 analysis.

---

# 14. GPU Usage & Computational Infrastructure

The experiments used multiple GPU environments throughout the project.

| GPU / Platform | Usage |
|---|---|
| NVIDIA Tesla T4 (Kaggle) | Primary training and evaluation |
| NVIDIA A30 (E2E Networks) | Initial trials, ScratchCNN training, and pipeline debugging |
| NVIDIA A5000 (GridShare) | DINOv2 advanced analysis |

**NVIDIA A30 (E2E Networks):**

Initially used for trials and failed runs, followed by successful experiments after debugging and pipeline corrections.

After reconsidering the overall workflow and conducting multiple rounds of debugging and training, the process worked optimally across Kaggle and GridShare.

**NVIDIA Tesla T4 (Kaggle):**

Used for primary model training and evaluation.

**NVIDIA A5000 (GridShare):**

Used for advanced DINOv2 analysis.

---

**Last Updated:** 4 November 2026
