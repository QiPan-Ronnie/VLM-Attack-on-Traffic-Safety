param(
    [string]$EnvName = "CSCI699",
    [Parameter(Mandatory = $true)]
    [string]$ModelName,
    [string]$CacheDir = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$CodeRoot = Split-Path -Parent $ScriptDir
$ProjectsRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $CodeRoot))
if (-not $CacheDir) {
    $CacheDir = Join-Path $ProjectsRoot "work\baseline_llava\hf_cache"
}
New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null

$CondaCmd = Get-Command conda -ErrorAction SilentlyContinue
if (-not $CondaCmd) {
    throw "conda was not found on PATH."
}
$CondaExe = $CondaCmd.Source

$env:HF_HOME = $CacheDir
$env:HUGGINGFACE_HUB_CACHE = $CacheDir
$env:TRANSFORMERS_CACHE = $CacheDir

$DownloadCode = "from huggingface_hub import snapshot_download; snapshot_download(repo_id=r'$ModelName', cache_dir=r'$CacheDir')"
& $CondaExe run -n $EnvName --no-capture-output python -c $DownloadCode
