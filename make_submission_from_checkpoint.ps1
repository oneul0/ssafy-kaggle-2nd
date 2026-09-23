$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$run = Join-Path $root 'output\qwen3vl8b_valid'
$final = Join-Path $run 'adapter'
if (Test-Path -LiteralPath (Join-Path $final 'adapter_config.json')) {
    $adapter = $final
} else {
    $checkpoints = @(Get-ChildItem -LiteralPath (Join-Path $run 'checkpoints') -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^checkpoint-(\d+)$' -and (Test-Path -LiteralPath (Join-Path $_.FullName 'adapter_config.json')) } |
        Sort-Object { [int]($_.Name -replace '^checkpoint-', '') })
    if (-not $checkpoints) { throw 'No completed Qwen3-VL checkpoint exists yet.' }
    $adapter = $checkpoints[-1].FullName
}
Write-Output "Using adapter: $adapter"
$python = Join-Path $root 'baseline\Scripts\python.exe'
& $python (Join-Path $root 'qwen3vl_submit_from_adapter.py') `
    --adapter $adapter `
    --data (Join-Path $root 'dataset') `
    --base (Join-Path $root 'downloads\models\Qwen3-VL-8B-Instruct') `
    --out (Join-Path $root 'output\qwen3vl8b_candidate') `
    --validate
if ($LASTEXITCODE -ne 0) { throw "Inference failed with exit code $LASTEXITCODE" }
