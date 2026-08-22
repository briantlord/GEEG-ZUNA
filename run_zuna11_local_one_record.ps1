param(
    [switch]$Run,
    [switch]$RepairMissingMetrics
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$Python = Join-Path $ProjectRoot ".zuna11_local_env\Scripts\python.exe"
$Runner = Join-Path $ProjectRoot "benchmark\metrics\run.py"
$CheckScript = Join-Path $ProjectRoot "benchmark\_check_zuna11_local.py"
$Input = Join-Path $ProjectRoot "GEEG_Raw\G001Day1Rest1.cnt"
$Cache = Join-Path $ProjectRoot "HF_cache"
$MneHome = Join-Path $ProjectRoot ".mne_local"
$ReconCache = Join-Path $ProjectRoot "results\zuna11_reconstructions_v3"
$WeightRepo = Join-Path $Cache "models--Zyphra--ZUNA1.1"
$Metrics = @("faa", "theta_beta", "frontal_midline_theta", "mu_asymmetry", "specparam_peaks")
$Output = Join-Path $ProjectRoot "results\metric_eval_G001Day1Rest1_zuna11_v4.csv"
$QcOutput = "$Output.reconstruction_qc.jsonl"
$StatusOutput = "$Output.status.jsonl"
$Log = Join-Path $ProjectRoot "results\metric_eval_G001Day1Rest1_zuna11_v4.log"
$RunManifest = Join-Path $ProjectRoot "results\run_manifest_G001Day1Rest1_v1.json"
if ($RepairMissingMetrics) {
    throw "-RepairMissingMetrics belongs to the quarantined v1 run and is not valid for corrected-v2."
}

foreach ($Path in @($Python, $Runner, $CheckScript, $Input, $WeightRepo)) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required path is missing: $Path"
    }
}

$WeightBlob = Get-ChildItem -LiteralPath (Join-Path $WeightRepo "blobs") -File |
    Where-Object Length -GT 1GB |
    Select-Object -First 1
if ($null -eq $WeightBlob) {
    throw "No ZUNA 1.1 weight blob larger than 1 GB was found under $WeightRepo"
}

# Keep all model access local and apply the Windows eager-mode compatibility settings.
$env:HF_HOME = $Cache
$env:HF_HUB_CACHE = $Cache
$env:HF_HUB_OFFLINE = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONUTF8 = "1"
$env:USE_LIBUV = "0"
$env:TORCHDYNAMO_DISABLE = "1"
$env:WANDB_MODE = "disabled"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:_MNE_FAKE_HOME_DIR = $MneHome
$env:ZUNA11_RECON_CACHE_DIR_V3 = $ReconCache
New-Item -ItemType Directory -Force -Path $MneHome | Out-Null
New-Item -ItemType Directory -Force -Path $ReconCache | Out-Null

& $Python $CheckScript
if ($LASTEXITCODE -ne 0) {
    throw "Local environment validation failed."
}

if (-not (Test-Path -LiteralPath $RunManifest)) {
    & $Python (Join-Path $ProjectRoot "benchmark\run_manifest.py") `
        --recordings $Input --methods spline zuna --out $RunManifest
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create immutable one-record run manifest."
    }
}

$Arguments = @(
    $Runner,
    "--subjects", "G001",
    "--methods", "spline", "zuna",
    "--allow-phase2-zuna"
) + @("--metrics") + $Metrics + @(
    "--zuna-version", "1.1",
    "--zuna-calibration", "median_survivor_std_zero_mean_carrier",
    "--run-manifest", $RunManifest,
    "--task-index", "0",
    "--stage0-cache-dir", (Join-Path $ProjectRoot "results\stage0_cache_v4"),
    "--out", $Output,
    "--qc-out", $QcOutput,
    "--status-out", $StatusOutput
)

Write-Host ""
Write-Host "Prepared local ZUNA 1.1 run"
Write-Host "  Recording: G001Day1Rest1.cnt"
Write-Host "  Metrics:   $($Metrics -join ', ')"
Write-Host "  Methods:   spline, ZUNA 1.1 with minimal no-ICA Stage 0"
Write-Host "  Run manifest: $RunManifest"
Write-Host "  Output:    $Output"
Write-Host "  Failures:  $StatusOutput"
Write-Host "  Log:       $Log"
Write-Host "  Reconstructions: $ReconCache"
Write-Host "  Mode:      Windows eager mode (slow)"

if (-not $Run) {
    Write-Host ""
    Write-Host "Configuration check only. Inference was NOT started."
    Write-Host "When ready, run:"
    $RepairArg = if ($RepairMissingMetrics) { " -RepairMissingMetrics" } else { "" }
    Write-Host "  powershell -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Run$RepairArg"
    exit 0
}

if (Test-Path -LiteralPath $Output) {
    Write-Host "Resuming verified units already present in: $Output"
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Output) | Out-Null
Push-Location $ProjectRoot
try {
    # Windows PowerShell wraps native stderr as ErrorRecord objects. Keep those non-terminating so
    # Python tracebacks and model diagnostics reach both the console and the log.
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python @Arguments 2>&1 | Tee-Object -FilePath $Log -Append
    $ExitCode = $LASTEXITCODE
    $ErrorActionPreference = $PreviousErrorActionPreference
}
finally {
    $ErrorActionPreference = "Stop"
    Pop-Location
}
if ($ExitCode -ne 0) {
    throw "The local metric run exited with code $ExitCode. See $Log"
}

Write-Host "Completed successfully: $Output"
