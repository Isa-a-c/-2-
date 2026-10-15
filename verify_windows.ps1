# 배포 폴더에서 실행한다. Python 없이 exe 자체로 검사한다.
$ErrorActionPreference = 'Stop'
$fourierExecutable = Join-Path $PSScriptRoot 'EngineeringMathStudio.exe'
$fourierReport = Join-Path $PSScriptRoot 'validation-report.json'
if (-not (Test-Path -LiteralPath $fourierExecutable)) { throw 'Run this script from the extracted release folder.' }
$fourierProcess = Start-Process -FilePath $fourierExecutable -ArgumentList '--self-test','validation-report.json' -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru
if (-not $fourierProcess.WaitForExit(300000)) {
    Stop-Process -Id $fourierProcess.Id
    throw 'Validation timed out after 300 seconds.'
}
if ($fourierProcess.ExitCode -ne 0) { throw "Executable failed. Check $fourierReport" }
if (-not (Test-Path -LiteralPath $fourierReport)) { throw 'No validation report was created.' }
$fourierResult = Get-Content -LiteralPath $fourierReport -Raw -Encoding UTF8 | ConvertFrom-Json
if (-not $fourierResult.passed) { throw $fourierResult.error }
$fourierResult | Format-List
Write-Host "Validation report: $fourierReport"
