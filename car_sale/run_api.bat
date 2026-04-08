@echo off
REM Car Sales Prediction API Launcher
REM This script ensures the correct working directory and runs the FastAPI app

cd /d "%~dp0"
python src\app.py
pause