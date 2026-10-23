

Reproducing the NeXtSwin-X paper (Table 3) and extending it with DINOv2.

## How to reproduce the data
I am not putting raw or processed images in this Git repo to keep it clean. All heavy data is hosted on Hugging Face. 

To get the exact same processed data, just run the preprocessing notebooks in the `notebooks/` folder.

BRISC 2025
- Raw data: [Kaggle BRISC 2025](https://www.kaggle.com/datasets/briscdataset/brisc2025)
- Processed HF: [srijan2715/brisc2025-paper-pipeline](https://huggingface.co/datasets/srijan2715/brisc2025-paper-pipeline)
- Steps: Loads official manifest, resizes to 256x256 (Bicubic), splits 85/15 (seed 42).

Sartaj
- Raw data: [Kaggle Brain Tumor MRI](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
- Processed HF: (Coming soon)
- Steps: Same resize and split logic as BRISC.

## Dataset Stats

BRISC 2025
- Total: 6,000 images (5,000 train, 1,000 test)
- Classes: Glioma, Meningioma, No Tumor, Pituitary
- Dropped: 0 corrupt images
- Final splits: Train = 4,250 | Val = 750 | Test = 1,000

Sartaj
- Total: ~3,264 images
- Classes: Same 4 classes
- Dropped: (Will update after preprocessing)
- Final splits: (Will update after preprocessing)

## Logging and Tracking

- **Automated Logs:** All training runs (loss, LR, metrics, git hash) are logged automatically to Weights & Biases (W&B).
- **Results:** Final numbers for every run are saved in `results.csv` to build the final report tables.
- **My Thinking:** I use `notebooks/lab_notebook.md` to write down what I tried, what I expected, and what actually happened (including failed runs). This is separate from the automated W&B logs.

## Repo Structure

- `notebooks/`: Preprocessing scripts and my daily lab notebook.
- `src/`: Model code (from-scratch CNN, DINOv2 wrappers) and training loops.
- `scripts/`: Automation scripts for Vast.ai runs.
- `results.csv`: Master file for all experiment metrics.
