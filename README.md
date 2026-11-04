Brain Tumor Classification: Original ScratchCNN & Reproducing Table 3
Developing an original lightweight ScratchCNN baseline and reproducing the NeXtSwin-X paper's Table 3 pre-trained models (Esfahani et al., 2026, Nature Scientific Reports).

Dataset (Preprocessed)
All images are processed directly from raw Hugging Face pixel lists into PyTorch tensors, permuted to (C, H, W), and scaled to [0, 1] to match strict Conv2d requirements.
Note: Initial attempts using standard PIL/torchvision failed due to HF's raw list storage format. A custom robust transform was engineered to handle this.

BRISC 2025
Processed: https://huggingface.co/datasets/srijan2715/brisc2025-paper-pipeline
Stats: Train=4,250 | Val=750 | Test=1,000 (Total: 6,000)
Notes: Verified 659 non-square images handled correctly. label_mapping.json included in repo.

Sartaj et al.
Processed: https://huggingface.co/datasets/srijan2715/sartaj-final-split
Stats: Train=2,444 | Val=426 | Test=394 (Total: 3,264)
Notes: Uses official unmodified split for faithful Table 3 reproduction. Additionally, SHA-256 hashing revealed 220 leaked images (55.8% of the test set), prompting a secondary "Clean" test evaluation to measure true generalization. label_mapping.json included in repo.

Model Checkpoints:
https://huggingface.co/srijan2715/scratchcnn-checkpoints

Current Status
Training Complete: ScratchCNN 50-epoch training complete across 3 statistical seeds (42, 43, 44) for both datasets.

Results:

BRISC: Achieved a healthy ~87% Train / ~81-85% Test accuracy, demonstrating successful generalization.

Sartaj: Achieved ~85-88% Train accuracy but exhibited a severe generalization gap, dropping to ~35-53% on the Official Test and ~16-48% on the Clean Test. This indicates that a 98k parameter model lacks the capacity to robustly generalize on this complex, small dataset.

Verification: All final metrics were cross-validated on both Kaggle and a bare-metal E2E A30 GPU, matching within a <0.1% margin of error.


update:  
The Sartaj evaluation pipeline has been fully debugged, verified, and stabilized. The initial 13.79% accuracy collapse was definitively isolated to a corrupted checkpoint artifact, not a model capacity issue. Following a complete regeneration (achieving 81.00% verified training accuracy), the main model registry was purged of broken weights, and the unified, reproducible training codebase data.py,train.py has been deployed.

Do check the repo:
https://huggingface.co/srijan2715/scratchcnn-checkpoints
https://huggingface.co/srijan2715/reproductioncheckpoints

Note: all the hardcoded and exposed tokens have been removed from HF and new token has been generated 
Current Status & Final Results

 Project Status: Complete & Verified
- Training Complete: ScratchCNN 50-epoch training complete across 3 statistical seeds (42, 43, 44) for both datasets.
- Pipeline Stabilization: The Sartaj evaluation pipeline has been fully debugged, verified, and stabilized. The initial severe accuracy collapse (e.g., 13.79%) was definitively isolated to a corrupted checkpoint artifact and preprocessing mismatch, *not* a model capacity issue. Following complete regeneration (achieving 81.00% verified training accuracy), the main model registry was purged of broken weights, and the unified, reproducible training codebase (`data.py`, `train.py`) has been deployed.
-Verification: All final metrics were cross-validated on  Kaggle through multiple rounds of runs and trials 

 Final Reproduced Results (Table 3)
All metrics represent Mean ± Standard Deviation across 3 random seeds (42, 43, 44).

| Model | BRISC (Accuracy / F1) | Sartaj (Accuracy / F1) |
| :--- | :---: | :---: |
| **TwoStream (ResNet50)** | **99.20 ± 0.22** / 99.20 ± 0.22 | **78.51 ± 0.43** / 75.07 ± 0.38 |
| **Swin-Transformer** | 99.20 ± 0.14 / 99.20 ± 0.14 | 78.34 ± 1.02 / 75.24 ± 1.40 |
| **EfficientNetV2-B3** | 99.10 ± 0.22 / 99.10 ± 0.22 | 76.82 ± 1.21 / 73.67 ± 1.88 |
| **BiFNN (ResNet50)** | 98.53 ± 0.65 / 98.53 ± 0.66 | 75.13 ± 1.56 / 71.47 ± 1.22 |
| **ConvNeXt-Base** | 98.43 ± 0.05 / 98.43 ± 0.05 | 77.07 ± 0.73 / 73.33 ± 0.87 |
| **NeXtSwin-X (Proposed)** | 91.17 ± 3.92 / 91.04 ± 3.96 | 74.11 ± 5.39 / 70.76 ± 7.63 |
| **ScratchCNN (Ours)** | 83.53 ± 1.15 / 83.63 ± 1.13 | 47.21 ± 0.55 / 44.49 ± 0.71 |

 DINOv2 Advanced Analysis (Part B)
Beyond Table 3 reproduction, this project includes a comprehensive interpretability study of the DINOv2 ViT-B/14 backbone:
 (Fine-tuning): Implemented linear probing, MLP/ConvNeXt-style heads, and partial fine-tuning (unfreezing the last 2 transformer blocks with `1e-5` LR), preventing catastrophic forgetting while adapting to tumor morphology.
2. Task 8 (PCA Visualization):** Projected 768D patch tokens into 3D RGB space, revealing that DINOv2 naturally clusters by anatomical regions without explicit labels.
3. Task 9 (Grad-CAM):** Contrasted CNN texture-bias (focusing on local tumor boundaries) with ViT shape-bias (attending to global anatomical displacement).
4. Task 10 (Disagreement Analysis): Identified specific images where the CNN succeeded but DINOv2 failed, hypothesizing that MRI artifacts/noise confuse the ViT's global attention, while the CNN's local receptive field remains robust.



See lab_notebook.md for daily progress, hypotheses, failures, paper citations, and detailed analysis.

For a granular chronology of Sartaj dataset debugging and pipeline forensics, see Sartaj_findings.md.

