# Comprehensive Google Colab Setup Guide

This guide provides step-by-step instructions on how to set up, clone, and run the thesis project securely and efficiently in Google Colab.

## 1. Initial Setup & Mounting Google Drive (Recommended)

Since Colab storage is ephemeral (it gets erased after your session ends), it is highly recommended to mount your Google Drive. This allows you to permanently save your repository, datasets, and generated model checkpoints.

Open a new Colab Notebook and run the following in the first cell:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Create a dedicated directory in your Drive and move into it:

```python
import os
# Create a folder for the project (change the name if you wish)
os.makedirs('/content/drive/MyDrive/thesis_project', exist_ok=True)

# Change the current working directory to the new folder
%cd /content/drive/MyDrive/thesis_project
```

## 2. Cloning the Repository Securely

If your GitHub repository is private, **do not hardcode your Personal Access Token (PAT)** in the notebook. Doing so can accidentally expose your credentials if the notebook is shared. Use Colab's interactive `getpass` module instead.

Run the following cell to securely input your GitHub PAT and clone the repository:

```python
import getpass
import os

print("Enter your GitHub Personal Access Token (PAT):")
git_token = getpass.getpass()

# TODO: Replace with your actual GitHub username and repository name
repo_owner = "your_github_username"
repo_name = "thesis_github"

# Clone the repository
repo_url = f"https://{git_token}@github.com/{repo_owner}/{repo_name}.git"
!git clone {repo_url}

# Clean up the token variable from memory for security
del git_token 
```

Navigate into the cloned repository:

```python
%cd {repo_name}
```
*(Note: If you run your Colab session again in the future, you do not need to clone again if it is on your Drive. Just run `%cd /content/drive/MyDrive/thesis_project/thesis_github`)*

## 3. Installing Dependencies

Install all the necessary Python packages specified by the project:

```python
!pip install -r requirements.txt
```

## 4. Configuring the Environment Variables (CRITICAL)

To ensure the project runs properly in Colab (such as bypassing organization proxy settings and optimizing HuggingFace Trainer checkpoints), you **must** set specific environment variables **before** importing the project's Python modules.

```python
import os

# 1. Enable Colab-specific modifications
# - Explicitly unsets any Intel/organization network proxies
# - Sets up optimal HuggingFace Trainer arguments 
#   (saves to drive, sequential checkpointing, loads best model at the end)
os.environ["THESIS_RUN_ENV"] = "colab"

# 2. Disable Weights & Biases (W&B) logging
# Recommended for Colab as W&B prompts can sometimes block cell execution
os.environ["WANDB_DISABLED"] = "true"

# 3. (Optional) Any other environment variables you want to set for your run
# os.environ["THESIS_NUM_EPOCHS"] = "5"
```

## 5. Running Experiments

Finally, you can run your experiments directly through shell commands (by prefixing with `!`) or interactively in Notebook cells.

**To run via command line:**
```python
!python run_all_experiments.py
```

**To run interactively in a Python cell:**
```python
# Assuming you want to run experiment #1
from experiments.experiment_01_regular_ner import run_experiment

# Trigger the execution
run_experiment()
```

---

### Understanding the Backstage Fixes

By setting `os.environ["THESIS_RUN_ENV"] = "colab"`, the project automatically applies the following changes:
1. **Fixes Proxy Timeouts:** Clears `HTTP_PROXY`, `HTTPS_PROXY`, and related network variables to prevent external requests to Hugging Face or datasets from timing out ([experiments/common.py](experiments/common.py)).
2. **Prevents Storage Exhaustion & Kernel Crashes:** Overrides local "temp directory" saving logic. It configures `TrainingArguments` ([core/th_functions.py](core/th_functions.py)) to limit the number of checkpoints on disk (`save_total_limit=3`), runs evaluations periodically (`eval_steps=100`, `save_steps=100`), and automatically saves and loads the best metric-based model at the end (`load_best_model_at_end=True`, `metric_for_best_model="overall_f1"`).
3. **Self-Healing Corrupted Checkpoints:** If a Colab session disconnects or is terminated while writing a checkpoint (leaving it corrupted/missing its weights file), the runner automatically detects the incomplete checkpoint, cleans it up safely, and resumes training from the highest valid checkpoint or from scratch ([core/th_functions.py](core/th_functions.py)).