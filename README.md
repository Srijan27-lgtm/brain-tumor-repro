-->> Brain Tumor Classification: Reproducing Table 3 + DINOv2
Reproducing the NeXtSwin-X paper (Esfahani et al., 2026, Nature Scientific Reports) and extending it with DINOv2.
 
 
Datasets-(Preprocessed)
All images are directly resized to 224×224 using BICUBIC Interpolation and normalized with ImageNet mean/std, exactly matching the paper's preprocessing pipeline.
> **Note:** Older versions (`sartaj-paper-pipeline`, `sartaj-clean-v2`) are deprecated. Use the final repos linked below.


(BRISC 2025)
Processed: [srijan2715/brisc2025-paper-pipeline](https://huggingface.co/datasets/srijan2715/brisc2025-paper-pipeline)
Stats: Train=4,250 | Val=750 | Test=1,000 (Total: 6,000)
Notes: Verified 659 non-square images handled correctly. `label_mapping.json` included in repo.


(Sartaj et al.)
Processed: [srijan2715/sartaj-final-split](https://huggingface.co/datasets/srijan2715/sartaj-final-split)
Stats: Train=2,444 | Val=426 | Test=394 (Total: 3,264)
Notes: Uses official unmodified split for faithful Table 3 reproduction. `label_mapping.json` included in repo.


Current Status
 Data preprocessing and verification complete (BRISC & Sartaj)
 Datasets uploaded to Hugging Face (Public)

See `lab_notebook.md` for daily progress, hypotheses, failures, paper citations, and analysis.
  



