$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$run = Join-Path $root 'output\qwen3vl8b_valid'
$adapter = Join-Path $run 'adapter'
$python = Join-Path $root 'baseline\Scripts\python.exe'
$infer = Join-Path $root 'qwen3vl_submit_from_adapter.py'
$data = Join-Path $root 'dataset'
$base = Join-Path $root 'downloads\models\Qwen3-VL-8B-Instruct'
$candidate = Join-Path $root 'output\qwen3vl8b_2epoch_candidate'
$highres = Join-Path $root 'output\qwen3vl8b_2epoch_highres'
$status = Join-Path $root 'output\qwen3vl8b_2epoch_pipeline_status.txt'

try {
    'Waiting for 2-epoch training process.' | Set-Content -LiteralPath $status -Encoding utf8
    $training = Get-Process -Id 40580 -ErrorAction SilentlyContinue
    if ($training) { Wait-Process -Id 40580 }
    if (-not (Test-Path -LiteralPath (Join-Path $adapter 'adapter_config.json')) -or
        -not (Test-Path -LiteralPath (Join-Path $run 'valid_scores.csv'))) {
        throw 'Training did not produce the final adapter and validation scores.'
    }

    '2-epoch training complete; scoring base candidate.' | Set-Content -LiteralPath $status -Encoding utf8
    & $python $infer --adapter $adapter --out $candidate --data $data --base $base --validate
    if ($LASTEXITCODE -ne 0) { throw "Base inference failed: $LASTEXITCODE" }

    'Base candidate complete; testing selective high resolution.' | Set-Content -LiteralPath $status -Encoding utf8
    & $python $infer --adapter $adapter --out $highres --data $data --base $base `
        --validate --uncertain-base-dir $candidate --uncertain-fraction 0.15 --max-visual-tokens 512
    if ($LASTEXITCODE -ne 0) { throw "High-resolution inference failed: $LASTEXITCODE" }

    $baseVal = Import-Csv -LiteralPath (Join-Path $candidate 'valid_scores.csv')
    $baseCorrect = @($baseVal | Where-Object { $_.gold -eq $_.pred }).Count
    $highresSubmission = Join-Path $highres 'submission.csv'
    if (Test-Path -LiteralPath $highresSubmission) {
        "Complete: base validation $baseCorrect/$($baseVal.Count); high-resolution candidate: $highresSubmission" |
            Set-Content -LiteralPath $status -Encoding utf8
    } else {
        "Complete: base validation $baseCorrect/$($baseVal.Count); high resolution did not improve validation. Candidate: $(Join-Path $candidate 'submission.csv')" |
            Set-Content -LiteralPath $status -Encoding utf8
    }
} catch {
    "Failed: $_" | Set-Content -LiteralPath $status -Encoding utf8
    throw
}
