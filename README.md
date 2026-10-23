# Brain Tumor Classification: Reproducing Table 3 & DINOv2 Extension

## Objective
Faithfully reproduce Table 3 from the NeXtSwin-X paper (Esfahani et al., 2026) and extend the analysis with DINOv2 feature extraction and ablation studies.

## Datasets (Preprocessed)
To save compute and ensure faithful reproduction of the `timm` ImageNet pipeline (`Resize(256)` -> `CenterCrop(224)`), datasets are preprocessed and hosted on Hugging Face:
- **BRISC 2025**: [srijan2715/brisc2025-paper-pipeline](https://huggingface.co/datasets/srijan2715/brisc2025-paper-pipeline)
- **Sartaj et al.**: *(Coming soon)*

## Rules Followed
- [x] Git commit at every working session with meaningful messages.
- [x] All runs logged to Weights & Biases (W&B).
- [x] Lab notebook maintained with predictions, results, and failure analysis.
- [x] Strict adherence to paper's preprocessing and hyperparameter pipeline.
