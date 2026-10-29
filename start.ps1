# MLReady One-Click Startup Script for PowerShell
$rootDir = $PSScriptRoot
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Starting MLReady SaaS Platform" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 0. Check MongoDB
$mongoRunning = $false
try {
    $tcp = Test-NetConnection -ComputerName 127.0.0.1 -Port 27017 -WarningAction SilentlyContinue -InformationLevel Quiet
    if ($tcp) { $mongoRunning = $true }
} catch {}

if (-not $mongoRunning) {
    Write-Host "[!] MongoDB is NOT detected on port 27017!" -ForegroundColor Red
    Write-Host "Attempting to start MongoDB service or mongod..." -ForegroundColor Yellow
    try {
        Start-Service MongoDB -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
        $tcp = Test-NetConnection -ComputerName 127.0.0.1 -Port 27017 -WarningAction SilentlyContinue -InformationLevel Quiet
        if ($tcp) { $mongoRunning = $true }
    } catch {}
    if (-not $mongoRunning) {
        Write-Host "Please start MongoDB or run: mongod --dbpath <data_folder>" -ForegroundColor Red
    }
} else {
    Write-Host "[OK] MongoDB is running on port 27017" -ForegroundColor Green
}

# 1. Start Backend API in a new window
Write-Host "Starting Backend API on port 8000..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$rootDir\backend'; Write-Host '--- MLReady Backend API (Port 8000) ---' -ForegroundColor Green; ..\venv\Scripts\uvicorn.exe app.main:app --reload --port 8000"

# 2. Start Background Worker in a new window
Write-Host "Starting Background Worker..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$rootDir\backend'; Write-Host '--- MLReady Background Worker ---' -ForegroundColor Green; ..\venv\Scripts\python.exe -m app.jobs.worker"

# 3. Start Frontend in a new window
Write-Host "Starting Frontend Web App on port 5173..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$rootDir\frontend'; Write-Host '--- MLReady Frontend UI (Port 5173) ---' -ForegroundColor Green; npm run dev"

Write-Host "`nAll services launched!" -ForegroundColor Green
Write-Host "Open your browser at: http://localhost:5173`n" -ForegroundColor Cyan
