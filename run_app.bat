@echo off
REM Run YOLO Dual Camera App with venv

cd /d "%~dp0"

echo Starting YOLO Dual Camera Detection App...
echo.

REM Activate .venv and run main.py
call ".venv\Scripts\activate.bat"

echo Environment activated. Starting application...
echo.

python main.py

echo.
echo Application closed.
pause
