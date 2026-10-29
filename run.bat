@echo off
setlocal
cd /d "%~dp0"
echo Launching MLReady All-In-One Runner...
if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe run_all.py
) else (
    python run_all.py
)
pause
