param(
    [ValidateSet(1, 3)][int]$Context = 1,
    [ValidateRange(1, 1000)][int]$Epochs = 8,
    [ValidateRange(1, 10000)][int]$StepsPerEpoch = 50,
    [ValidateSet('none', 'sqrt_inverse')][string]$ClassBalance = 'none',
    [ValidateRange(1, 1000)][int]$Patience = 5,
    [string]$RunName = 'baseline'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if ($RunName -notmatch '^[a-zA-Z0-9_-]+$') { throw 'RunName must contain only letters, numbers, underscore or dash.' }
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
function Invoke-ProjectPython {
    param([string[]]$Arguments)
    & $projectPython @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Pipeline stopped: Python exited with $LASTEXITCODE" }
}
Invoke-ProjectPython -Arguments @('scripts/download_data.py', '--patients', '24')
Invoke-ProjectPython -Arguments @('scripts/prepare_data.py')
$runPath = "runs/$RunName"
$trainArguments = @('scripts/train.py', '--out', $runPath, '--epochs', "$Epochs", '--steps-per-epoch', "$StepsPerEpoch", '--context', "$Context", '--class-balance', $ClassBalance, '--patience', "$Patience")
if (Test-Path -LiteralPath "$runPath/last.pt") { $trainArguments += @('--resume', "$runPath/last.pt") }
Invoke-ProjectPython -Arguments $trainArguments
Invoke-ProjectPython -Arguments @('scripts/evaluate.py', '--checkpoint', "$runPath/best.pt", '--split', 'validation', '--out', "runs/${RunName}_validation")
Invoke-ProjectPython -Arguments @('scripts/plot_results.py', '--run', $runPath, '--evaluation', "runs/${RunName}_validation")
Write-Output 'Pipeline complete. Run .\start_app.ps1 to open the local interface.'
