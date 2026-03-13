"""
Copy-paste friendly Kaggle notebook cells for UnlearnRec.
Run each block in order in a Kaggle notebook.
"""

# Cell 1: Environment info
import os
import sys
import torch
print('Python:', sys.version)
print('Torch:', torch.__version__)

# Cell 2: Set repo location
# Option A: repository cloned into /kaggle/working/UnlearnRec
REPO_DIR = '/kaggle/working/UnlearnRec'
# Option B: repository attached as a Kaggle Dataset
# REPO_DIR = '/kaggle/input/your-unlearnrec-dataset'
os.chdir(REPO_DIR)
print('Working directory:', os.getcwd())

# Cell 3: Install dependencies
os.system('pip -q install numpy scipy networkx setproctitle')
os.system('pip -q install torch-sparse -f https://data.pyg.org/whl/torch-2.6.0+cu124.html')

# Cell 4: Short sanity run
cmd = (
    'python run_experiment.py --stage pretrain_simgcl -- '
    '--data gowalla --epoch 5 --batch 1024 --adversarial_attack False '
    '--save_path ./checkpoints/gowalla/before_unlearning/kaggle_demo_simgcl'
)
print(cmd)
os.system(cmd)

# Cell 5: Next stage examples
# os.system('python run_experiment.py --stage main_drop -- --data gowalla --trained_model ./checkpoints/gowalla/before_unlearning/kaggle_demo_simgcl --save_path ./checkpoints/gowalla/pretrain_4_unlearning/kaggle_demo_unlearn')
# os.system('python run_experiment.py --stage finetune -- --data gowalla --model_2_finetune ./checkpoints/gowalla/pretrain_4_unlearning/kaggle_demo_unlearn --save_path ./checkpoints/gowalla/finetuned/kaggle_demo_ft')
