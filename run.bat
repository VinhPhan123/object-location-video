@echo off
REM YOLO Dual Camera Application Launcher

echo.
echo =========================================
echo   YOLO Dual Camera Detection & Crop
echo =========================================
echo.

REM Check if virtual environment exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    
    echo Installing dependencies...
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)

echo.
echo Starting application...
python main.py

pause
