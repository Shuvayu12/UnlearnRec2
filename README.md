# UnlearnRec

Research code for GNN recommendation pretraining, unlearning, and fine-tuning.

This repository remains script-first (research style), but it now has a clear execution map and a Kaggle notebook path.

## 1) Codebase Map

Core training and experiment scripts:

- `pretrain.py`: SGL pretraining
- `pretrain_lightgcn.py`: LightGCN pretraining
- `pretrain_simgcl.py`: SimGCL pretraining
- `unlearn.py`: graph unlearning stage
- `fineTune.py`: finetune unlearned model
- `main_drop.py`: pretrain-for-unlearning with edge dropping flow
- `fineTune_drop.py`: finetune for the drop-based flow
- `test_simgcl.py`: evaluate trained model
- `make_noise_dataset.py`: optional noisy/adversarial dataset generation

Shared modules:

- `Model.py`: model definitions (LightGCN, SimGCL, GraphUnlearning, etc.)
- `data_handler.py`: dataset loading, edge drop/adversarial split, dataloaders
- `params.py`: all CLI arguments
- `Utils/time_logger.py`: logging utility
- `Utils/utils.py`: losses, metrics, helper math

Experiment resources:

- `examples/pretrain_4_unlearning/`: shell command examples
- `datasets/`: train/test sparse matrices and adversarial variants
- `logs/`: past experiment logs

## 2) Unified Entry Point

Use `run_experiment.py` to avoid remembering many script names.

Examples:

```bash
python run_experiment.py --stage pretrain_simgcl -- --data gowalla --epoch 100 --batch 2048
python run_experiment.py --stage unlearn -- --data gowalla --trained_model ./checkpoints/gowalla/before_unlearning/your_model
python run_experiment.py --stage finetune -- --data gowalla --model_2_finetune ./checkpoints/gowalla/pretrain_4_unlearning/your_unlearned_model
```

Available stages:

- `pretrain_sgl`
- `pretrain_lightgcn`
- `pretrain_simgcl`
- `unlearn`
- `finetune`
- `finetune_drop`
- `main_drop`
- `test_simgcl`
- `make_noise_dataset`

## 3) Local Setup

```bash
pip install -r requirements.txt
```

Important note:

- `torch-sparse` often needs a wheel matching your installed PyTorch/CUDA stack.
- If `pip install -r requirements.txt` fails on `torch-sparse`, install it separately with the wheel index shown in the Kaggle section.

## 4) Kaggle Notebook Workflow

You have two practical options.

If you want ready-to-paste cells, use `notebooks/kaggle_quickstart_cells.py`.

### Option A: Add this repo as a Kaggle Dataset

1. Create/upload a Kaggle Dataset that contains this repository files.
2. Attach that dataset to your notebook.
3. In notebook cells:

```python
import os
os.chdir('/kaggle/input/YOUR_DATASET_NAME')
print(os.getcwd())
```

Then install dependencies and run experiments:

```python
import os
import sys
import torch

print('Python:', sys.version)
print('Torch:', torch.__version__)

os.system('pip -q install numpy scipy networkx setproctitle')
os.system('pip -q install torch-sparse -f https://data.pyg.org/whl/torch-2.6.0+cu124.html')
```

```python
os.system('python run_experiment.py --stage pretrain_simgcl -- --data gowalla --epoch 20 --batch 2048 --adversarial_attack False')
```

### Option B: Clone in notebook startup cell

```python
%cd /kaggle/working
!git clone YOUR_REPO_URL UnlearnRec
%cd /kaggle/working/UnlearnRec
!pip -q install -r requirements.txt
```

If `torch-sparse` fails:

```python
!pip -q install torch-sparse -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

## 5) Minimal End-to-End Flow

1. Pretrain base model:

```bash
python run_experiment.py --stage pretrain_simgcl -- --data gowalla --save_path ./checkpoints/gowalla/before_unlearning/pretrain_simgcl_demo
```

2. Unlearn:

```bash
python run_experiment.py --stage main_drop -- --data gowalla --trained_model ./checkpoints/gowalla/before_unlearning/pretrain_simgcl_demo --save_path ./checkpoints/gowalla/pretrain_4_unlearning/unlearn_demo
```

3. Finetune:

```bash
python run_experiment.py --stage finetune -- --data gowalla --model_2_finetune ./checkpoints/gowalla/pretrain_4_unlearning/unlearn_demo --save_path ./checkpoints/gowalla/finetuned/finetune_demo
```

## 6) Key Hyperparameters to Tune

These matter most for trade-off between recommendation quality and unlearning strength:

- `--pretrain_drop_rate`: 0.03 to 0.3
- `--batch`: 256 to 4096
- `--reg`: 1e-8 to 1e-6
- `--unlearn_wei`: 0.1 to 1.0
- `--align_wei`: 0.001 to 0.05
- `--unlearn_ssl`: 1e-4 to 1e-3

## 7) Existing Example Commands

See:

- `examples/pretrain_4_unlearning/unlearn_lightgcn_on_gowalla.sh`
- `examples/pretrain_4_unlearning/finetune_unlearned_lightgcn_on_gowalla.sh`

