param(
    [string]$EnvName = "CSCI699"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$CondaCmd = Get-Command conda -ErrorAction SilentlyContinue
if (-not $CondaCmd) {
    throw "conda was not found on PATH."
}
$CondaExe = $CondaCmd.Source

$ValidationCode = "import torch, transformers, open_clip, pandas; print('torch', torch.__version__); print('transformers', transformers.__version__); print('open_clip', getattr(open_clip, '__version__', 'unknown')); print('pandas', pandas.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
& $CondaExe run -n $EnvName --no-capture-output python -c $ValidationCode
