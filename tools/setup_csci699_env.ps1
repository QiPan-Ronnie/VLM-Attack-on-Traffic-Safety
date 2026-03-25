param(
    [string]$EnvName = "CSCI699",
    [string]$PythonVersion = "3.11",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu128",
    [switch]$InstallBitsAndBytes
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RequirementsPath = Join-Path $ScriptDir "requirements-csci699.txt"

$CondaCmd = Get-Command conda -ErrorAction SilentlyContinue
if (-not $CondaCmd) {
    throw "conda was not found on PATH. Please install Miniconda/Anaconda first."
}
$CondaExe = $CondaCmd.Source

Write-Host "Using conda: $CondaExe"
Write-Host "Creating/updating environment: $EnvName"
& $CondaExe create -n $EnvName python=$PythonVersion -y

Write-Host "Upgrading pip"
& $CondaExe run -n $EnvName --no-capture-output python -m pip install --upgrade pip

Write-Host "Installing GPU PyTorch build for CUDA 12.8"
& $CondaExe run -n $EnvName --no-capture-output python -m pip install `
    torch torchvision torchaudio `
    --index-url $TorchIndexUrl

Write-Host "Installing project requirements"
& $CondaExe run -n $EnvName --no-capture-output python -m pip install -r $RequirementsPath

if ($InstallBitsAndBytes) {
    Write-Host "Installing bitsandbytes"
    & $CondaExe run -n $EnvName --no-capture-output python -m pip install bitsandbytes
}

Write-Host "Validating imports and CUDA visibility"
$ValidationCode = "import torch, transformers, open_clip, pandas; print('torch', torch.__version__); print('transformers', transformers.__version__); print('open_clip', getattr(open_clip, '__version__', 'unknown')); print('pandas', pandas.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
& $CondaExe run -n $EnvName --no-capture-output python -c $ValidationCode
