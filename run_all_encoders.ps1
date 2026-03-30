# run_all_encoders.ps1
# Runs main_drop.py once per encoder type, saving results to separate paths.
# Adjust --trained_model, --data, and other hyperparams to match your setup.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$encoders = @("default", "autoencoder", "attention", "hypernet", "causal")

# ---- shared arguments (edit these to match your experiment) ----
$commonArgs = @(
    "--trained_model", "./checkpoints/gowalla/before_unlearning/pretrain_gowalla_lightgcn_advlightgcn0.5_reg1e-7_lr1e-3_b4096_ep200_dim128_ly3",
    "--seed",           "1234",
    "--adversarial_attack", "True",
    "--fineTune",       "False",
    "--adv_method",     "lightgcn0.5",
    "--model",          "lightgcn",
    "--data",           "gowalla",
    "--reg",            "0.0000001",
    "--lr",             "0.001",
    "--batch",          "1024",
    "--epoch",          "256",
    "--sim_epoch",      "5",
    "--latdim",         "128",
    "--gnn_layer",      "3",
    "--unlearn_layer",  "0",
    "--bpr_wei",        "1",
    "--align_type",     "v2",
    "--unlearn_type",   "v1",
    "--unlearn_wei",    "1",
    "--align_wei",      "0.005",
    "--align_temp",     "1",
    "--hyper_temp",     "1",
    "--unlearn_ssl",    "0.001",
    "--pretrain_drop_rate", "0.2",
    "--layer_mlp",      "2",
    "--perf_degrade",   "0.5",
    "--overall_withdraw_rate", "0.1",
    "--withdraw_rate_init", "1",
    "--leaky",          "0.99"
)

foreach ($enc in $encoders) {
    $savePath = "./checkpoints/gowalla/pretrain_4_unlearning/encoder_${enc}"
    $logFile  = "./logs/gowalla/pretrain_4_unlearning/encoder_${enc}.log"

    # Ensure log directory exists
    $logDir = Split-Path $logFile
    if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }

    Write-Host "`n=========================================="
    Write-Host "  Running encoder: $enc"
    Write-Host "==========================================`n"

    $encArgs = @("--encoder_type", $enc, "--save_path", $savePath) + $commonArgs

    python main_drop.py @encArgs 2>&1 | Tee-Object -FilePath $logFile

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: encoder '$enc' failed with exit code $LASTEXITCODE"
    } else {
        Write-Host "Encoder '$enc' finished successfully."
    }
}

Write-Host "`nAll encoder runs complete."
