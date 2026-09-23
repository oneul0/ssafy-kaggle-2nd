$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$source = Join-Path $root 'dataset/dev.csv'
$target = Join-Path $root 'output/dev_pseudo_3of5.csv'
$rows = Import-Csv -LiteralPath $source
$pseudo = foreach ($row in $rows) {
    $votes = @($row.answer1, $row.answer2, $row.answer3, $row.answer4, $row.answer5) |
        Where-Object { $_ -cmatch '^[abcd]$' }
    $winner = $votes | Group-Object | Sort-Object Count -Descending | Select-Object -First 1
    if ($winner -and $winner.Count -ge 3) {
        [pscustomobject]@{
            id = $row.id; path = $row.path; question = $row.question
            a = $row.a; b = $row.b; c = $row.c; d = $row.d
            answer = $winner.Name; votes = $winner.Count; total_votes = $votes.Count
        }
    }
}
if (@($pseudo | Select-Object -ExpandProperty id -Unique).Count -ne @($pseudo).Count) {
    throw 'Duplicate dev IDs in pseudo labels.'
}
$pseudo | Export-Csv -LiteralPath $target -NoTypeInformation -Encoding utf8
Write-Output "Saved $(@($pseudo).Count) pseudo labels to $target"
