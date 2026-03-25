# Run YOLO Dual Camera App with venv

# Get the script directory
$script_dir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Change to script directory
Set-Location $script_dir

$venv_script = ".\\.venv\\Scripts\\Activate.ps1"

Write-Host "Starting YOLO Dual Camera Detection App..." -ForegroundColor Green
Write-Host ""

# Activate .venv
if (Test-Path $venv_script) {
    & $venv_script
} else {
    Write-Host "Warning: Virtual environment not found at $venv_script" -ForegroundColor Yellow
}

Write-Host "Environment activated. Starting application..." -ForegroundColor Green
Write-Host ""

# Run main.py
python main.py

Write-Host ""
Write-Host "Application closed." -ForegroundColor Yellow
Read-Host "Press Enter to exit"
