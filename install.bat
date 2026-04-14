@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

echo ===================================================
echo Amin OmniVoice Installer
echo ===================================================

SET "ROOT_DIR=%~dp0"
SET "LOG_FILE=%ROOT_DIR%install_log.txt"
echo If the window closes, check that file for the error!
echo.

:: 1. Check Python Architecture (Critical for ML)
echo [1/5] Checking Python architecture...
python -c "import platform; print(f'Python Architecture: {platform.architecture()[0]}')" >> "%LOG_FILE%" 2>&1
python -c "import platform; arch = platform.architecture()[0]; exit(0 if arch == '64bit' else 1)"
if !ERRORLEVEL! neq 0 (
    echo [CRITICAL ERROR] You are using 32-bit Python. 
    echo Amin OmniVoice (and PyTorch) REQUIRE 64-bit Python.
    echo Please uninstall Python and install the "Windows installer (64-bit)" version.
    echo.
    echo Check install_log.txt for details.
    pause
    exit /b 1
)
echo Python 64-bit confirmed.

:: 2. Setup Venv
echo.
echo [2/5] Setting up virtual environment...
if exist "%ROOT_DIR%env" (
    echo Existing 'env' found.
) else (
    echo Creating fresh 'env'...
    python -m venv "%ROOT_DIR%env" >> "%LOG_FILE%" 2>&1
    if !ERRORLEVEL! neq 0 (
        echo [ERROR] Venv creation failed. Check install_log.txt
        pause
        exit /b 1
    )
)

echo.
echo [3/5] Activating environment...
call "%ROOT_DIR%env\Scripts\activate.bat" >> "%LOG_FILE%" 2>&1
if !ERRORLEVEL! neq 0 (
    echo [ERROR] Activation failed. Check install_log.txt
    pause
    exit /b 1
)

echo.
echo [4/5] Installing PyTorch with auto-detection...
python -m pip install --upgrade pip wheel >> "%LOG_FILE%" 2>&1

:: GPU Detection
set "CUDA_INDEX=cu124"
nvidia-smi --query-gpu=name --format=csv,noheader > "%ROOT_DIR%gpu.tmp" 2>nul
if !ERRORLEVEL! == 0 (
    set /p GPU_NAME=<"%ROOT_DIR%gpu.tmp"
    echo Detected GPU: !GPU_NAME!
    echo !GPU_NAME! | findstr /i "RTX.50" >nul
    if !ERRORLEVEL! == 0 (
        set "CUDA_INDEX=cu128"
        echo Target: CUDA 12.8 (Blackwell)
    ) else (
        echo Target: CUDA 12.4 (Stable)
    )
    del "%ROOT_DIR%gpu.tmp"
) else (
    echo No NVIDIA GPU detected via SMI. Using CPU/Stable fallback.
)

echo.
echo Starting massive 2GB download... this might take several minutes.
echo Be patient! Do not close the window.
python -m pip install --force-reinstall --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/%CUDA_INDEX% >> "%LOG_FILE%" 2>&1
if !ERRORLEVEL! neq 0 (
    echo [ERROR] PyTorch installation failed. 
    echo This is likely a network timeout or disk space issue.
    echo Check the END of install_log.txt for the specific error.
    pause
    exit /b 1
)

echo.
echo [5/5] Installing remaining studio dependencies...
python -m pip install -r "%ROOT_DIR%app\requirements.txt" >> "%LOG_FILE%" 2>&1
if !ERRORLEVEL! neq 0 (
    echo [ERROR] Requirements failed. Check install_log.txt
    pause
    exit /b 1
)

echo.
echo ===================================================
echo Verifying Installation...
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')" >> "%LOG_FILE%" 2>&1
echo DONE! 
echo.
echo Check the end of install_log.txt to confirm everything looks good.
echo ===================================================
pause
