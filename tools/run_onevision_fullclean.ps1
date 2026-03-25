param(
    [string]$EnvName = "CSCI699",
    [string]$ModelName = "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    [string]$PromptFile = "",
    [string]$RunLabel = "default",
    [switch]$Detached
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$CodeRoot = Split-Path -Parent $ScriptDir
$BundleRoot = Split-Path -Parent $CodeRoot
$ProjectsRoot = Split-Path -Parent (Split-Path -Parent $BundleRoot)
$ScriptsRoot = Join-Path $CodeRoot "scripts"
$WorkRoot = Join-Path $ProjectsRoot "work\baseline_llava"
$ModelWorkRoot = Join-Path $WorkRoot "llava_onevision_7b"
$LogsRoot = Join-Path $BundleRoot "logs"
$HFHome = Join-Path $WorkRoot "hf_cache"
$PythonExe = Join-Path "D:\miniconda3\envs" "$EnvName\python.exe"

$Manifest = Join-Path $ModelWorkRoot "clean_only_6935.csv"
$SafeRunLabel = ($RunLabel -replace "[^A-Za-z0-9_-]", "_")
if ([string]::IsNullOrWhiteSpace($SafeRunLabel)) {
    $SafeRunLabel = "default"
}
$OutputJsonl = Join-Path $ModelWorkRoot "clean_only_6935_$SafeRunLabel`_preds.jsonl"
$LatestRunInfo = Join-Path $ModelWorkRoot "clean_only_6935_$SafeRunLabel`_latest_run.json"

if ([string]::IsNullOrWhiteSpace($PromptFile)) {
    $PromptFile = Join-Path $CodeRoot "prompts\binary_imminent_risk_prompt_v3_json.txt"
}

foreach ($dir in @($ModelWorkRoot, $LogsRoot, $HFHome)) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

if (-not (Test-Path $PythonExe)) {
    throw "Python not found: $PythonExe"
}

if (-not (Test-Path $Manifest)) {
    throw "Full clean manifest not found: $Manifest"
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogsRoot "llava_onevision_7b_clean6935_$SafeRunLabel`_$timestamp.log"
$LauncherStdout = Join-Path $LogsRoot "llava_onevision_7b_clean6935_$SafeRunLabel`_$timestamp.launcher.out.log"
$LauncherStderr = Join-Path $LogsRoot "llava_onevision_7b_clean6935_$SafeRunLabel`_$timestamp.launcher.err.log"

$ArgsList = @(
    (Join-Path $ScriptsRoot "06_run_vlm_attack_hf.py"),
    "--manifest", $Manifest,
    "--output-jsonl", $OutputJsonl,
    "--model-name", $ModelName,
    "--prompt-file", $PromptFile,
    "--device-map", "auto",
    "--dtype", "float16",
    "--max-new-tokens", "16",
    "--cache-dir", $HFHome,
    "--local-files-only",
    "--clear-cuda-cache",
    "--load-in-4bit",
    "--skip-existing",
    "--log-file", $LogFile
)

$runInfo = [ordered]@{
    started_at = (Get-Date).ToString("s")
    run_label = $SafeRunLabel
    manifest = $Manifest
    output_jsonl = $OutputJsonl
    log_file = $LogFile
    launcher_stdout = $LauncherStdout
    launcher_stderr = $LauncherStderr
    model_name = $ModelName
    prompt_file = $PromptFile
    detached = [bool]$Detached
}
$runInfo | ConvertTo-Json -Depth 4 | Set-Content -Path $LatestRunInfo -Encoding UTF8

if ($Detached) {
    $commandString = "& '$PSCommandPath' -EnvName '$EnvName' -ModelName '$ModelName' -PromptFile '$PromptFile'"
    $commandString += " -RunLabel '$SafeRunLabel'"
    $WrapperArgs = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-Command", $commandString
    )
    $proc = Start-Process -FilePath "powershell.exe" -ArgumentList $WrapperArgs -WorkingDirectory $ProjectsRoot -RedirectStandardOutput $LauncherStdout -RedirectStandardError $LauncherStderr -PassThru
    Write-Host "Started detached full-clean run."
    Write-Host "PID: $($proc.Id)"
    Write-Host "Log: $LogFile"
    Write-Host "Output: $OutputJsonl"
    Write-Host "Run info: $LatestRunInfo"
    exit 0
}

& $PythonExe @ArgsList
