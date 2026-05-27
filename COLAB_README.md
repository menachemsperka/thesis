# Running the Thesis Project in Google Colab

This repository has been adapted to allow experiments to run on Google Colab (with CPU/GPU environments) as well as locally.

## Setup Instructions for Colab

When starting a new Colab Notebook and cloning your repository into the environment (or working from a mounted Google Drive), run the following code cell before importing and running any of the experiment scripts:

```python
import os

# 1. Set the environment variable to enable Colab-specific modifications
# This triggers the code to ignore Intel specific proxy configurations 
# and sets up optimal HuggingFace Trainer arguments for checkpointing.
os.environ["THESIS_RUN_ENV"] = "colab"

# 2. Disable Weights & Biases (optional but recommended in Colab)
# WANDB can sometimes cause execution blocking or unwanted interactions in Colab.
os.environ["WANDB_DISABLED"] = "true"

# Optional: if you're saving checkpoints to Google Drive, mount it first!
# from google.colab import drive
# drive.mount('/content/drive')
```

## What the `THESIS_RUN_ENV = "colab"` Flag Does

Setting ``THESIS_RUN_ENV = "colab"`` applies the following conditional changes throughout the codebase:

**1. Adjusts network configurations (`experiments/common.py`)**  
Removes Intel-specific HTTP and HTTPS proxies that cause issues in Colab's network environment. 

**2. Adjusts Model Training Arguments and Checkpointing (`core/th_functions.py`)**  
Configures HuggingFace `TrainingArguments` specifically for a Colab environment:
- Checkpoints are saved sequentially avoiding memory / storage crashes (`save_total_limit=3`).
- Evaluating and saving runs every 100 steps.
- Uses `load_best_model_at_end=True` alongside the F1 metric to ensure the robust best outcome from the training phase.
- Stores the final checkpoint output directly into a specified directory (`trainer_output/final_model`), making it easy to store directly onto Google Drive by setting `output_path="/content/drive/MyDrive/YOUR_FOLDER"` when invoking `train_and_evaluate_model()`. No changes are applied to local local CPU runs unless this environment flag originates in Colab.