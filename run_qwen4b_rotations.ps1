param()

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$repo = Join-Path $root 'reference_0_95948'
$python = Join-Path $root 'baseline\Scripts\python.exe'
$out = Join-Path $root 'output\qwen4b_retrain_pipeline'
$status = Join-Path $out 'rotation_status.txt'
New-Item -ItemType Directory -Path $out -Force | Out-Null

$env:SSAFY_DATA = Join-Path $root 'dataset'
$env:SSAFY_MODELS = Join-Path $root 'downloads\models'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:PYTORCH_CUDA_ALLOC_CONF = 'garbage_collection_threshold:0.8,max_split_size_mb:512'

function Run-Step([string]$name, [string[]]$arguments, [string]$working, [string]$expected) {
    if (Test-Path -LiteralPath $expected) { return }
    Set-Content -LiteralPath $status -Value "Running: $name" -Encoding utf8
    $stdout = Join-Path $out "$name.log"
    $stderr = Join-Path $out "$name.err.log"
    $p = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $working `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
        -WindowStyle Hidden -Wait -PassThru
    if ($p.ExitCode -ne 0) { throw "$name failed with exit code $($p.ExitCode): $stderr" }
    if (-not (Test-Path -LiteralPath $expected)) { throw "$name did not create $expected" }
}

function Score([string]$split, [string]$model, [string]$adapterTag, [string]$baseTag, [int]$rot) {
    $tag = if ($rot) { "${baseTag}_rot$rot" } else { $baseTag }
    $expected = Join-Path $repo "outputs\probs\${tag}_${split}.npz"
    $name = "${split}_${tag}"
    Run-Step $name @((Join-Path $repo 'src\score.py'), '--split', $split,
        '--model', $model, '--adapter', "outputs/adapters/$adapterTag/final",
        '--max-tokens', '768', '--quant', '4bit', '--tag', $baseTag, '--rot', "$rot") $repo $expected
}

try {
    foreach ($item in @(
        @('Qwen3-VL-4B-Instruct', 'local_q3vl4b_r16', 'local_q3vl4b_r16_t768_q4'),
        @('Qwen3.5-4B', 'local_q35_4b_r16', 'local_q35_4b_r16_t768_q4')
    )) {
        $baseHoldout = Join-Path $repo "outputs\probs\$($item[2])_holdout.npz"
        if (-not (Test-Path -LiteralPath $baseHoldout)) { throw "Training/holdout not complete: $baseHoldout" }
        foreach ($rot in 1..3) { Score 'holdout' $item[0] $item[1] $item[2] $rot }
    }

    $gate = Join-Path $out 'rotation_gate.json'
    Run-Step 'evaluate_rotations' @((Join-Path $root 'evaluate_local_qwen4b.py')) $root $gate
    $metrics = Get-Content -LiteralPath $gate -Raw | ConvertFrom-Json
    if (-not $metrics.test_gate) {
        Set-Content -LiteralPath $status -Value "No holdout gain: $($metrics.local_four_rotated_correct)/1000 vs $($metrics.reference_four_correct)/1000" -Encoding utf8
        return
    }

    foreach ($item in @(
        @('Qwen3-VL-4B-Instruct', 'local_q3vl4b_r16', 'local_q3vl4b_r16_t768_q4'),
        @('Qwen3.5-4B', 'local_q35_4b_r16', 'local_q35_4b_r16_t768_q4')
    )) {
        foreach ($rot in 0..3) { Score 'test' $item[0] $item[1] $item[2] $rot }
    }
    $submission = Join-Path $out 'submission_local_q4_rot_t768.csv'
    Run-Step 'build_submission' @((Join-Path $root 'build_local_qwen4b_submission.py')) $root $submission
    Set-Content -LiteralPath $status -Value "Complete: holdout $($metrics.local_four_rotated_correct)/1000; submission $submission" -Encoding utf8
} catch {
    Set-Content -LiteralPath $status -Value "Failed: $_" -Encoding utf8
    throw
}
