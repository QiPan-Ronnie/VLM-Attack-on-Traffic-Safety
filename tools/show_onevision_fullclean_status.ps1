param(
    [string]$RunLabel = "default"
)

$SafeRunLabel = ($RunLabel -replace "[^A-Za-z0-9_-]", "_")
if ([string]::IsNullOrWhiteSpace($SafeRunLabel)) {
    $SafeRunLabel = "default"
}

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$CodeRoot = Split-Path -Parent $ScriptDir
$BundleRoot = Split-Path -Parent $CodeRoot
$ProjectsRoot = Split-Path -Parent (Split-Path -Parent $BundleRoot)
$WorkRoot = Join-Path $ProjectsRoot "work\baseline_llava"
$ModelWorkRoot = Join-Path $WorkRoot "llava_onevision_7b"

$LatestRunInfo = Join-Path $ModelWorkRoot "clean_only_6935_$SafeRunLabel`_latest_run.json"
$OutputJsonl = Join-Path $ModelWorkRoot "clean_only_6935_$SafeRunLabel`_preds.jsonl"

if (-not (Test-Path $LatestRunInfo)) {
    throw "Run info not found: $LatestRunInfo"
}

$info = Get-Content -Path $LatestRunInfo -Raw | ConvertFrom-Json
$logFile = $info.log_file

Write-Host "Latest log: $logFile"
Write-Host "Output JSONL: $OutputJsonl"

if (Test-Path $OutputJsonl) {
    $count = @(
        Get-Content -Path $OutputJsonl | Where-Object { $_.Trim() -ne "" }
    ).Count
    Write-Host "Completed predictions: $count"
} else {
    Write-Host "Completed predictions: 0"
}

if (Test-Path $logFile) {
    Write-Host ""
    Write-Host "--- Log tail ---"
    Get-Content -Path $logFile -Tail 20
}
