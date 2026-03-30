"""
Kaggle notebook — UnlearnRec: run all 5 influence encoders end-to-end.
Copy this file into a Kaggle notebook (one cell per '# %%' block).

Prerequisites
─────────────
1. Enable GPU: Settings → Accelerator → GPU T4 ×2 (or P100).
2. Upload your project folder as a Kaggle **Dataset** named "unlearnrec",
   so it appears at  /kaggle/input/unlearnrec/UnlearnRec/  with all code
   and dataset pickle files inside.
   Alternatively, clone from git in Cell 2 below.
"""

# %% Cell 1 — Environment check
import os, re, subprocess, sys, torch
print("Python :", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA   :", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

# %% Cell 2 — Get the code into /kaggle/working/UnlearnRec
# ── OPTION A: repo uploaded as a Kaggle Dataset ──
INPUT_DIR = "/kaggle/input/unlearnrec/UnlearnRec"
WORK_DIR  = "/kaggle/working/UnlearnRec"

if os.path.isdir(INPUT_DIR):
    # Kaggle input is read-only, so copy to working dir
    os.system(f"cp -r {INPUT_DIR} {WORK_DIR}")
    print(f"Copied from dataset → {WORK_DIR}")
else:
    # ── OPTION B: clone from GitHub (set your repo URL) ──
    REPO_URL = "https://github.com/<YOUR_USER>/UnlearnRec.git"
    os.system(f"git clone {REPO_URL} {WORK_DIR}")
    print(f"Cloned from git → {WORK_DIR}")

os.chdir(WORK_DIR)
print("cwd:", os.getcwd())
print("contents:", os.listdir("."))

# %% Cell 3 — Install dependencies
os.system("pip install -q numpy scipy networkx setproctitle")

# torch-sparse wheel must match your PyTorch + CUDA versions.
# Kaggle P100/T4 images typically have CUDA 12.1 or 12.4.
# Adjust the URL below to match (check with `torch.version.cuda`).
cuda_tag = torch.version.cuda.replace(".", "")        # e.g. "121" or "124"
torch_tag = ".".join(torch.__version__.split(".")[:2]) # e.g. "2.6"
whl_url = f"https://data.pyg.org/whl/torch-{torch_tag}.0+cu{cuda_tag}.html"
print(f"Installing torch-sparse from: {whl_url}")
os.system(f"pip install -q torch-sparse -f {whl_url}")

# quick import test
import torch_sparse
print("torch_sparse version:", torch_sparse.__version__)

# %% Cell 4 — Verify dataset files exist
DATA = "ml1m"
data_map = {
    "ml1m":     "./datasets/ml-1m/",
    "yelp2018": "./datasets/yelp2018/",
    "yelp":     "./datasets/sparse_yelp/",
    "gowalla":  "./datasets/sparse_gowalla/",
    "amazon":   "./datasets/sparse_amazon/",
}
data_dir = data_map[DATA]
for f in ["trn_mat.pkl", "tst_mat.pkl"]:
    path = os.path.join(data_dir, f)
    assert os.path.isfile(path), f"Missing: {path}"
    print(f"  ✓ {path}")

# If adversarial attack data is used:
ADV_METHOD = "lightgcn"
adv_file = os.path.join(data_dir, f"adv_{ADV_METHOD}_mat.pkl")
if os.path.isfile(adv_file):
    print(f"  ✓ {adv_file}")
else:
    print(f"  ⚠ {adv_file} not found — set adversarial_attack=False or generate it first")

# %% Cell 5 — Step 1: Pretrain the base LightGCN model (skip if checkpoint already exists)
PRETRAIN_SAVE = f"./checkpoints/{DATA}/before_unlearning/pretrain_{DATA}_lightgcn"

if os.path.isfile(PRETRAIN_SAVE + ".mod"):
    print(f"Pretrained checkpoint already exists: {PRETRAIN_SAVE}.mod — skipping.")
else:
    os.makedirs(os.path.dirname(PRETRAIN_SAVE), exist_ok=True)
    cmd = f"""python pretrain_lightgcn.py \
        --data {DATA} \
        --model lightgcn \
        --save_path {PRETRAIN_SAVE} \
        --adversarial_attack True \
        --adv_method {ADV_METHOD} \
        --reg 0.0000001 \
        --lr 0.001 \
        --batch 4096 \
        --epoch 200 \
        --latdim 128 \
        --gnn_layer 3 \
        --gpu 0"""
    print("Running pretrain:\n", cmd)
    ret = os.system(cmd)
    assert ret == 0, f"Pretrain failed with code {ret}"
    print(f"✓ Pretrained checkpoint saved: {PRETRAIN_SAVE}.mod")

# %% Cell 6 — Step 2: Run unlearning with ALL 5 encoder types
import time

ENCODERS = ["default", "autoencoder", "attention", "hypernet", "causal"]
TRAINED_MODEL = PRETRAIN_SAVE   # path to the pretrained base model (no .mod suffix)

# Shared hyperparameters (same as the example shell scripts)
COMMON = f"""\
    --trained_model {TRAINED_MODEL} \
    --seed 1234 \
    --adversarial_attack True \
    --fineTune False \
    --adv_method {ADV_METHOD} \
    --model lightgcn \
    --data {DATA} \
    --reg 0.0000001 \
    --lr 0.001 \
    --batch 1024 \
    --epoch 256 \
    --sim_epoch 5 \
    --latdim 128 \
    --gnn_layer 3 \
    --unlearn_layer 0 \
    --bpr_wei 1 \
    --align_type v2 \
    --unlearn_type v1 \
    --unlearn_wei 1 \
    --align_wei 0.005 \
    --align_temp 1 \
    --hyper_temp 1 \
    --unlearn_ssl 0.001 \
    --pretrain_drop_rate 0.2 \
    --layer_mlp 2 \
    --perf_degrade 0.5 \
    --overall_withdraw_rate 0.1 \
    --withdraw_rate_init 1 \
    --leaky 0.99 \
    --gpu 0"""

metric_pattern = re.compile(r"^FINAL_METRICS\|(.*)$")
results = {}
for enc in ENCODERS:
    save_path = f"./checkpoints/{DATA}/pretrain_4_unlearning/encoder_{enc}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cmd = f"python main_drop.py --encoder_type {enc} --save_path {save_path} {COMMON}"

    # Extra loss-weight args for specific encoders
    if enc == "autoencoder":
        cmd += " --lambda_rec 1e-3"
    elif enc == "causal":
        cmd += " --lambda_causal 1e-3"

    print(f"\n{'='*60}")
    print(f"  ENCODER: {enc}")
    print(f"{'='*60}")
    t0 = time.time()
    proc = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    elapsed = time.time() - t0

    # Print logs so Kaggle output still shows training progress.
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr)

    status = "OK" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
    row = {
        "status": status,
        "Recall": None,
        "NDCG": None,
        "MI-BF": None,
        "MI-NG": None,
    }

    for line in reversed(proc.stdout.splitlines() if proc.stdout else []):
        m = metric_pattern.match(line.strip())
        if not m:
            continue
        parts = {}
        for kv in m.group(1).split("|"):
            if "=" in kv:
                k, v = kv.split("=", 1)
                parts[k.strip()] = v.strip()
        try:
            row["Recall"] = float(parts.get("recall"))
            row["NDCG"] = float(parts.get("ndcg"))
            row["MI-BF"] = float(parts.get("mi_bf"))
            row["MI-NG"] = float(parts.get("mi_ng"))
        except (TypeError, ValueError):
            pass
        break

    results[enc] = row
    print(f"  {status}  ({elapsed/60:.1f} min)")

print("\n\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"{'Encoder':<12} {'Status':<12} {'Recall':>10} {'NDCG':>10} {'MI-BF':>10} {'MI-NG':>10}")
for enc in ENCODERS:
    row = results[enc]
    recall = "-" if row["Recall"] is None else f"{row['Recall']:.6f}"
    ndcg = "-" if row["NDCG"] is None else f"{row['NDCG']:.6f}"
    mi_bf = "-" if row["MI-BF"] is None else f"{row['MI-BF']:.6f}"
    mi_ng = "-" if row["MI-NG"] is None else f"{row['MI-NG']:.6f}"
    print(f"{enc:<12} {row['status']:<12} {recall:>10} {ndcg:>10} {mi_bf:>10} {mi_ng:>10}")

# %% Cell 7 — Step 3 (optional): Fine-tune each unlearned model
FINETUNE_ENCODERS = ["default", "autoencoder", "attention", "hypernet", "causal"]

FT_COMMON = f"""\
    --seed 1234 \
    --adversarial_attack True \
    --fineTune True \
    --adv_method {ADV_METHOD} \
    --model lightgcn \
    --data {DATA} \
    --reg 0.0000001 \
    --lr 0.001 \
    --batch 1024 \
    --epoch 256 \
    --sim_epoch 5 \
    --latdim 128 \
    --gnn_layer 3 \
    --unlearn_layer 0 \
    --bpr_wei 1 \
    --align_type v2 \
    --unlearn_type v1 \
    --unlearn_wei 0.2 \
    --align_wei 0.01 \
    --align_temp 10 \
    --hyper_temp 1 \
    --unlearn_ssl 0.001 \
    --pretrain_drop_rate 0.2 \
    --layer_mlp 2 \
    --perf_degrade 0.5 \
    --overall_withdraw_rate 0.1 \
    --withdraw_rate_init 1 \
    --leaky 0.99 \
    --gpu 0"""

for enc in FINETUNE_ENCODERS:
    unlearned_ckpt = f"./checkpoints/{DATA}/pretrain_4_unlearning/encoder_{enc}"
    ft_save = f"./checkpoints/{DATA}/finetuned/ft_encoder_{enc}"
    os.makedirs(os.path.dirname(ft_save), exist_ok=True)

    cmd = (f"python fineTune.py "
           f"--encoder_type {enc} "
           f"--model_2_finetune {unlearned_ckpt} "
           f"--save_path {ft_save} {FT_COMMON}")

    if enc == "autoencoder":
        cmd += " --lambda_rec 1e-3"
    elif enc == "causal":
        cmd += " --lambda_causal 1e-3"

    print(f"\n{'='*60}")
    print(f"  FINETUNE ENCODER: {enc}")
    print(f"{'='*60}")
    ret = os.system(cmd)
    print("  ✓ Done" if ret == 0 else f"  ✗ Failed (code {ret})")

# %% Cell 8 — List all saved checkpoints
import glob
for ckpt in sorted(glob.glob("./checkpoints/**/*.mod", recursive=True)):
    size_mb = os.path.getsize(ckpt) / 1e6
    print(f"  {size_mb:6.1f} MB  {ckpt}")
