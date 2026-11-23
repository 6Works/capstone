@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo   6-Stage Conversational Voice Agent Pipeline
echo   (Isolated Conda Environment: capstone_voice)
echo ========================================================

:: Detect Conda environment python
set "CONDA_ENV_PY=%USERPROFILE%\.conda\envs\capstone_voice\python.exe"
if exist "%CONDA_ENV_PY%" (
    set "PY_CMD=%CONDA_ENV_PY%"
    echo [Environment] Using Conda env: capstone_voice
) else (
    set "PY_CMD=python"
    echo [Warning] Conda env not found, falling back to system python.
)

echo.
echo Select execution mode:
echo   [1] Start Web UI (Interactive Visual 6-Stage Demo at http://localhost:8000)
echo   [2] Run Terminal Automated Demo (Simulated conversation)
echo   [3] Run Live Microphone Loop (Speak into your PC mic)
echo   [4] Run Automated Pytest Test Suite
echo ========================================================
set /p choice="Enter choice [1-4]: "

if "%choice%"=="1" (
    echo Starting FastAPI Web Server at http://localhost:8000 ...
    "%PY_CMD%" server.py
) else if "%choice%"=="2" (
    echo Running Automated Demo...
    "%PY_CMD%" main.py --mode demo
) else if "%choice%"=="3" (
    echo Starting Live Microphone Interactive Loop...
    "%PY_CMD%" main.py --mode mic
) else if "%choice%"=="4" (
    echo Running Pytest Suite...
    "%PY_CMD%" -m pytest test_pipeline.py -v
) else (
    echo Invalid choice. Defaulting to Web UI...
    "%PY_CMD%" server.py
)
pause
