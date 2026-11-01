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


See lab_notebook.md for daily progress, hypotheses, failures, paper citations, and detailed analysis.

For a granular chronology of Sartaj dataset debugging and pipeline forensics, see Sartaj_findings.md.
