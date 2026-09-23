param()

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$repo = Join-Path $root 'reference_0_95948'
$python = Join-Path $root 'baseline\Scripts\python.exe'
$out = Join-Path $root 'output\qwen4b_retrain_pipeline'
$status = Join-Path $out 'status.txt'
New-Item -ItemType Directory -Path $out -Force | Out-Null

$env:SSAFY_DATA = Join-Path $root 'dataset'
$env:SSAFY_MODELS = Join-Path $root 'downloads\models'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:PYTORCH_CUDA_ALLOC_CONF = 'garbage_collection_threshold:0.8,max_split_size_mb:512'

function Run-Step([string]$name, [string[]]$arguments, [string]$expected) {
    if (Test-Path -LiteralPath $expected) {
        Write-Host "Skip ${name}: $expected"
        return
    }
    Set-Content -LiteralPath $status -Value "Running: $name" -Encoding utf8
    $stdout = Join-Path $out "$name.log"
    $stderr = Join-Path $out "$name.err.log"
    $p = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repo `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
        -WindowStyle Hidden -Wait -PassThru
    if ($p.ExitCode -ne 0) {
        throw "$name failed with exit code $($p.ExitCode). Logs: $stdout and $stderr"
    }
    if (-not (Test-Path -LiteralPath $expected)) {
        throw "$name exited successfully but did not create $expected"
    }
}

function Train-Model([string]$model, [string]$tag, [int]$batch, [int]$accum) {
    $base = Join-Path $root "downloads\models\$model"
    if (-not (Test-Path -LiteralPath (Join-Path $base 'source_revision.txt'))) {
        throw "Model download incomplete: $base"
    }
    $trainScript = Join-Path $repo 'src\train_lora.py'
    $scoreScript = Join-Path $repo 'src\score.py'
    $smoke = Join-Path $repo "outputs\adapters\smoke_$tag\final\adapter_config.json"
    $final = Join-Path $repo "outputs\adapters\$tag\final\adapter_config.json"
    $scoreSmoke = Join-Path $repo "outputs\probs\smoke_score_${tag}_q4_holdout.npz"
    $holdout = Join-Path $repo "outputs\probs\${tag}_t768_q4_holdout.npz"

    Run-Step "smoke_$tag" @($trainScript, '--model', $model, '--tag', "smoke_$tag",
        '--batch', "$batch", '--accum', "$accum", '--limit', '16', '--eval-every', '0',
        '--save-every', '2', '--log-every', '1') $smoke

    $trainArgs = @($trainScript, '--model', $model, '--tag', $tag,
        '--batch', "$batch", '--accum', "$accum", '--epochs', '1', '--lr', '0.0001',
        '--rank', '16', '--alpha', '32', '--max-tokens', '768', '--min-tokens', '256',
        '--eval-every', '0', '--save-every', '50', '--log-every', '10')
    $checkpoint = Join-Path $repo "outputs\adapters\$tag\ckpt\state.pt"
    if (Test-Path -LiteralPath $checkpoint) { $trainArgs += '--resume' }
    Run-Step "train_$tag" $trainArgs $final

    Run-Step "smoke_score_$tag" @($scoreScript, '--split', 'holdout', '--model', $model,
        '--adapter', "outputs/adapters/$tag/final", '--max-tokens', '768',
        '--quant', '4bit', '--limit', '20', '--tag', "smoke_score_${tag}_q4") $scoreSmoke

    Run-Step "holdout_$tag" @($scoreScript, '--split', 'holdout', '--model', $model,
        '--adapter', "outputs/adapters/$tag/final", '--max-tokens', '768',
        '--quant', '4bit', '--tag', "${tag}_t768_q4") $holdout
}

try {
    Train-Model 'Qwen3-VL-4B-Instruct' 'local_q3vl4b_r16' 2 4
    Train-Model 'Qwen3.5-4B' 'local_q35_4b_r16' 1 8
    Set-Content -LiteralPath $status -Value 'Complete: both 4B models trained and holdout scored.' -Encoding utf8
} catch {
    Set-Content -LiteralPath $status -Value "Failed: $_" -Encoding utf8
    throw
}
