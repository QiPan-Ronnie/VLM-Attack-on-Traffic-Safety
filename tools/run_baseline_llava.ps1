param(
    [ValidateSet("smoke", "pilot")]
    [string]$Stage = "smoke",
    [string]$EnvName = "CSCI699",
    [string]$Llava0p5BModel = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    [string]$Llava7BModel = "llava-hf/llava-1.5-7b-hf",
    [switch]$RunLlava7B,
    [switch]$Use4BitFor7B
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$CodeRoot = Split-Path -Parent $ScriptDir
$BundleRoot = Split-Path -Parent $CodeRoot
$ProjectsRoot = Split-Path -Parent (Split-Path -Parent $BundleRoot)
$ScriptsRoot = Join-Path $CodeRoot "scripts"
$MasterPackRoot = Join-Path $ProjectsRoot "datasets\master_pack\master_pack"
$MasterManifest = Join-Path $MasterPackRoot "manifests\manifest.csv"
$WorkRoot = Join-Path $ProjectsRoot "work\baseline_llava"
$HFHome = Join-Path $WorkRoot "hf_cache"
$OffloadRoot = Join-Path $WorkRoot "offload"
$LogsRoot = Join-Path $BundleRoot "logs"

$CondaCmd = Get-Command conda -ErrorAction SilentlyContinue
if (-not $CondaCmd) {
    throw "conda was not found on PATH."
}
$CondaExe = $CondaCmd.Source

foreach ($dir in @($WorkRoot, $HFHome, $OffloadRoot, $LogsRoot)) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

function Invoke-CondaPython {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$ArgsList
    )
    $env:HF_HOME = $HFHome
    $env:HUGGINGFACE_HUB_CACHE = $HFHome
    $env:TRANSFORMERS_CACHE = $HFHome
    & $CondaExe run -n $EnvName --no-capture-output python @ArgsList
}

function Run-Smoke {
    $SmokeRoot = Join-Path $WorkRoot "smoke"
    $SmokeAttackRoot = Join-Path $SmokeRoot "attack_benchmark"
    $SmokeClipRoot = Join-Path $SmokeRoot "clip"
    $SmokeLlavaRoot = Join-Path $SmokeRoot "llava_0p5b"
    foreach ($dir in @($SmokeRoot, $SmokeAttackRoot, $SmokeClipRoot, $SmokeLlavaRoot)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }

    $RoiManifest = Join-Path $SmokeRoot "frame_manifest_tte_roi.csv"
    $AttackManifest = Join-Path $SmokeAttackRoot "manifest_attack.csv"
    $ClipPreds = Join-Path $SmokeClipRoot "clip_attack_preds.jsonl"
    $ClipEval = Join-Path $SmokeClipRoot "eval"
    $ClipLog = Join-Path $LogsRoot "clip_smoke.log"
    $LlavaPreds = Join-Path $SmokeLlavaRoot "llava_0p5b_preds.jsonl"
    $LlavaEval = Join-Path $SmokeLlavaRoot "eval"
    $LlavaLog = Join-Path $LogsRoot "llava_0p5b_smoke.log"

    Invoke-CondaPython @(
        (Join-Path $ScriptsRoot "03b_add_attention_roi_from_maps.py"),
        "--input-csv", $MasterManifest,
        "--output-csv", $RoiManifest,
        "--data-root", $MasterPackRoot
    )

    Invoke-CondaPython @(
        (Join-Path $ScriptsRoot "04_build_safety_attack_benchmark.py"),
        "--input-csv", $RoiManifest,
        "--data-root", $MasterPackRoot,
        "--output-dir", $SmokeAttackRoot,
        "--overlay-types", "text_watermark", "timestamp_box", "logo_patch", "semi_transparent_bar",
        "--placement-modes", "random",
        "--severities", "2", "4",
        "--variants-per-setting", "1",
        "--include-clean",
        "--labeled-only",
        "--save-ext", ".jpg",
        "--limit", "100",
        "--seed", "42"
    )

    Invoke-CondaPython @(
        (Join-Path $ScriptsRoot "05_run_clip_attack_baseline.py"),
        "--manifest", $AttackManifest,
        "--clip-model", "ViT-B-32",
        "--clip-pretrained", "laion2b_s34b_b79k",
        "--output-jsonl", $ClipPreds,
        "--max-samples", "32",
        "--device", "cuda:0",
        "--batch-size", "32",
        "--log-file", $ClipLog
    )

    Invoke-CondaPython @(
        (Join-Path $ScriptsRoot "07_eval_safety_attack.py"),
        "--manifest", $AttackManifest,
        "--predictions", $ClipPreds,
        "--output-dir", $ClipEval
    )

    Invoke-CondaPython @(
        (Join-Path $ScriptsRoot "06_run_vlm_attack_hf.py"),
        "--manifest", $AttackManifest,
        "--model-name", $Llava0p5BModel,
        "--output-jsonl", $LlavaPreds,
        "--max-samples", "8",
        "--dtype", "bfloat16",
        "--log-file", $LlavaLog
    )

    Invoke-CondaPython @(
        (Join-Path $ScriptsRoot "07_eval_safety_attack.py"),
        "--manifest", $AttackManifest,
        "--predictions", $LlavaPreds,
        "--output-dir", $LlavaEval
    )
}

function Run-Pilot {
    $PilotAttackRoot = Join-Path $WorkRoot "pilot_attack"
    $ClipRoot = Join-Path $WorkRoot "clip"
    $Llava0p5BRoot = Join-Path $WorkRoot "llava_0p5b"
    $Llava7BRoot = Join-Path $WorkRoot "llava_7b"
    foreach ($dir in @($PilotAttackRoot, $ClipRoot, $Llava0p5BRoot, $Llava7BRoot)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }

    $RoiManifest = Join-Path $PilotAttackRoot "frame_manifest_tte_roi.csv"
    $AttackManifest = Join-Path $PilotAttackRoot "manifest_attack.csv"
    $ClipPreds = Join-Path $ClipRoot "clip_attack_preds.jsonl"
    $ClipEval = Join-Path $ClipRoot "eval"
    $ClipLog = Join-Path $LogsRoot "clip_pilot.log"
    $SubsetManifest = Join-Path $Llava0p5BRoot "vlm_eval_subset.csv"
    $LlavaPreds = Join-Path $Llava0p5BRoot "llava_0p5b_preds.jsonl"
    $LlavaEval = Join-Path $Llava0p5BRoot "eval"
    $LlavaLog = Join-Path $LogsRoot "llava_0p5b_pilot.log"
    $Llava7BLog = Join-Path $LogsRoot "llava_7b_pilot.log"

    Invoke-CondaPython @(
        (Join-Path $ScriptsRoot "03b_add_attention_roi_from_maps.py"),
        "--input-csv", $MasterManifest,
        "--output-csv", $RoiManifest,
        "--data-root", $MasterPackRoot
    )

    Invoke-CondaPython @(
        (Join-Path $ScriptsRoot "04_build_safety_attack_benchmark.py"),
        "--input-csv", $RoiManifest,
        "--data-root", $MasterPackRoot,
        "--output-dir", $PilotAttackRoot,
        "--overlay-types", "text_watermark", "timestamp_box", "logo_patch", "semi_transparent_bar",
        "--placement-modes", "random",
        "--severities", "2", "4",
        "--variants-per-setting", "1",
        "--include-clean",
        "--labeled-only",
        "--save-ext", ".jpg",
        "--seed", "42"
    )

    Invoke-CondaPython @(
        (Join-Path $ScriptsRoot "05_run_clip_attack_baseline.py"),
        "--manifest", $AttackManifest,
        "--clip-model", "ViT-B-32",
        "--clip-pretrained", "laion2b_s34b_b79k",
        "--output-jsonl", $ClipPreds,
        "--device", "cuda:0",
        "--batch-size", "64",
        "--log-file", $ClipLog
    )

    Invoke-CondaPython @(
        (Join-Path $ScriptsRoot "07_eval_safety_attack.py"),
        "--manifest", $AttackManifest,
        "--predictions", $ClipPreds,
        "--output-dir", $ClipEval
    )

    Invoke-CondaPython @(
        (Join-Path $ScriptsRoot "05b_build_vlm_eval_subset.py"),
        "--manifest", $AttackManifest,
        "--output-csv", $SubsetManifest,
        "--clean-total", "100",
        "--per-condition-total", "50",
        "--placement-mode", "random",
        "--seed", "42"
    )

    Invoke-CondaPython @(
        (Join-Path $ScriptsRoot "06_run_vlm_attack_hf.py"),
        "--manifest", $SubsetManifest,
        "--model-name", $Llava0p5BModel,
        "--output-jsonl", $LlavaPreds,
        "--dtype", "bfloat16",
        "--log-file", $LlavaLog
    )

    Invoke-CondaPython @(
        (Join-Path $ScriptsRoot "07_eval_safety_attack.py"),
        "--manifest", $SubsetManifest,
        "--predictions", $LlavaPreds,
        "--output-dir", $LlavaEval
    )

    if ($RunLlava7B) {
        $SubsetManifest7B = Join-Path $Llava7BRoot "vlm_eval_subset.csv"
        Copy-Item -LiteralPath $SubsetManifest -Destination $SubsetManifest7B -Force
        $Llava7BPreds = Join-Path $Llava7BRoot "llava_7b_preds.jsonl"
        $Llava7BEval = Join-Path $Llava7BRoot "eval"
        $ArgsList = @(
            (Join-Path $ScriptsRoot "06_run_vlm_attack_hf.py"),
            "--manifest", $SubsetManifest7B,
            "--model-name", $Llava7BModel,
            "--output-jsonl", $Llava7BPreds,
            "--dtype", "float16",
            "--offload-dir", (Join-Path $OffloadRoot "llava_7b"),
            "--log-file", $Llava7BLog
        )
        if ($Use4BitFor7B) {
            $ArgsList += "--load-in-4bit"
        }
        Invoke-CondaPython -ArgsList $ArgsList

        Invoke-CondaPython @(
            (Join-Path $ScriptsRoot "07_eval_safety_attack.py"),
            "--manifest", $SubsetManifest7B,
            "--predictions", $Llava7BPreds,
            "--output-dir", $Llava7BEval
        )
    }
}

if ($Stage -eq "smoke") {
    Run-Smoke
} else {
    Run-Pilot
}



