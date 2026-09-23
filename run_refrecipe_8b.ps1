param(
    [ValidateSet(512, 768)]
    [int]$MaxVisualTokens = 768
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$python = Join-Path $root 'baseline\Scripts\python.exe'
$data = Join-Path $root 'dataset'
$holdout = Join-Path $root 'reference_0_95948\outputs\split\holdout.csv'
$model = Join-Path $root 'downloads\models\Qwen3-VL-8B-Instruct'
$run = Join-Path $root "output\qwen3vl8b_refrecipe_t$MaxVisualTokens"
$smoke = Join-Path $root "output\qwen3vl8b_refrecipe_t${MaxVisualTokens}_smoke"
$submission = Join-Path $root "output\qwen3vl8b_refrecipe_t${MaxVisualTokens}_submission"
$status = Join-Path $run 'pipeline_status.txt'

foreach ($required in @($python, $holdout, (Join-Path $model 'config.json'))) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Missing required file: $required" }
}
New-Item -ItemType Directory -Path $run -Force | Out-Null
$env:PYTORCH_CUDA_ALLOC_CONF = 'garbage_collection_threshold:0.8,max_split_size_mb:512'

function Run-Python([string[]]$ArgsList) {
    & $python @ArgsList
    if ($LASTEXITCODE -ne 0) { throw "Python exited with code $LASTEXITCODE" }
}

$trainCommon = @(
    (Join-Path $root 'qwen3vl_5060ti.py'), '--data', $data,
    '--mode', 'valid', '--model', $model, '--valid-ids', $holdout,
    '--skip-duplicate-filter', '--no-dev', '--reference-prompt', '--shuffle-options',
    '--max-visual-tokens', "$MaxVisualTokens", '--min-visual-tokens', '256',
    '--visual-patch-size', '28', '--epochs', '1', '--rank', '16', '--alpha', '32',
    '--learning-rate', '0.0001', '--scheduler', 'cosine', '--warmup-fraction', '0.03'
)

try {
    if (-not (Test-Path -LiteralPath (Join-Path $smoke 'adapter\adapter_config.json'))) {
        Set-Content -LiteralPath $status -Value 'Smoke test: training 16 examples.'
        Run-Python ($trainCommon + @('--out', $smoke, '--smoke', '16'))
    }

    if (-not (Test-Path -LiteralPath (Join-Path $run 'adapter\adapter_config.json'))) {
        Set-Content -LiteralPath $status -Value 'Full training in progress.'
        $trainArgs = $trainCommon + @('--out', $run, '--save-steps', '100', '--save-total-limit', '2')
        $checkpoints = @(Get-ChildItem -LiteralPath (Join-Path $run 'checkpoints') -Directory -Filter 'checkpoint-*' -ErrorAction SilentlyContinue |
            Sort-Object { [int]($_.Name -replace '^checkpoint-', '') } -Descending)
        if ($checkpoints.Count -gt 0) {
            $trainArgs += @('--resume-from', $checkpoints[0].FullName)
            Write-Host "Resuming from $($checkpoints[0].FullName)"
        }
        Run-Python $trainArgs
    }

    if (-not (Test-Path -LiteralPath (Join-Path $submission 'submission.csv'))) {
        Set-Content -LiteralPath $status -Value 'Training complete; scoring holdout and 6714 test images.'
        Run-Python @(
            (Join-Path $root 'qwen3vl_submit_from_adapter.py'),
            '--adapter', (Join-Path $run 'adapter'), '--out', $submission,
            '--data', $data, '--base', $model, '--validate',
            '--valid-split', (Join-Path $run 'valid_split.csv'),
            '--reference-prompt', '--max-visual-tokens', "$MaxVisualTokens",
            '--min-visual-tokens', '256', '--visual-patch-size', '28'
        )
    }

    $sample = Import-Csv -LiteralPath (Join-Path $data 'sample_submission.csv')
    $result = Import-Csv -LiteralPath (Join-Path $submission 'submission.csv')
    $valid = Import-Csv -LiteralPath (Join-Path $submission 'valid_scores.csv')
    if ($result.Count -ne 6714 -or $sample.Count -ne 6714 -or $valid.Count -ne 1000) {
        throw 'Unexpected validation or submission row count.'
    }
    if (($result[0].PSObject.Properties.Name -join ',') -ne 'id,answer') {
        throw 'Submission columns must be id,answer.'
    }
    for ($i = 0; $i -lt $result.Count; $i++) {
        if ($result[$i].id -ne $sample[$i].id -or $result[$i].answer -cnotmatch '^[abcd]$') {
            throw "Invalid submission row $i"
        }
    }
    $correct = @($valid | Where-Object { $_.gold -eq $_.pred }).Count
    Set-Content -LiteralPath $status -Value "Complete: holdout $correct/1000; submission $submission"
    Write-Host "Complete: holdout $correct/1000; submission $submission"
} catch {
    Set-Content -LiteralPath $status -Value "Failed: $_"
    throw
}
