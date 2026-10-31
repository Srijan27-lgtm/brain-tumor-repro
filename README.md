
Brain Tumor Classification: Reproducing Table 3

Developing an original lightweight ScratchCNN baseline and reproducing the NeXtSwin-X paper's Table 3 pre-trained models (Esfahani et al., 2026, Nature Scientific Reports).

Datasets-(Preprocessed)
All images are processed directly from raw Hugging Face pixel lists into PyTorch tensors, permuted to (C, H, W), and scaled to [0, 1] to match strict Conv2d requirements.

Note: Initial attempts using standard PIL/torchvision failed due to HF's raw list storage format. A custom robust transform is used to handle this.

(BRISC 2025)
Processed: srijan2715/brisc2025-paper-pipeline
Stats: Train=4,250 | Val=750 | Test=1,000 (Total: 6,000)
Notes: Verified 659 non-square images handled correctly. label_mapping.json included in repo.

(Sartaj et al.)
Processed: srijan2715/sartaj-final-split
Stats: Train=2,444 | Val=426 | Test=394 (Total: 3,264)
Notes: Uses official unmodified split for faithful Table 3 reproduction. label_mapping.json included in repo.
Engineering Journey & Key Fixes
 Status

 ScratchCNN 50-epoch training complete across 3 statistical seeds (42, 43, 44) for both datasets.
 
 BRISC achieved a healthy ~86.87% training accuracy; Sartaj reached 100% (smaller dataset, but weights are robust due to augmentation).
All 6 final .pth checkpoints securely uploaded to Hugging Face
(scratch_paper/ directory).


See lab_notebook.md for daily progress, hypotheses, failures, paper citations, and detailed analysis.
