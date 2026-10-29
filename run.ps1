# MLReady All-In-One Runner for PowerShell
$rootDir = $PSScriptRoot
Set-Location $rootDir

if (Test-Path "$rootDir\venv\Scripts\python.exe") {
    & "$rootDir\venv\Scripts\python.exe" "$rootDir\run_all.py"
} else {
    python "$rootDir\run_all.py"
}
