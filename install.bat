@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

:: DEBUG: Force the window to stay open at the very start so we know it launched
echo DEBUG: Installer has started.
echo Root Path: %~dp0
pause

echo ===================================================
echo Amin OmniVoice Installer (Robust Edition)
echo ===================================================

SET "ROOT_DIR=%~dp0"
SET "ENV_DIR=%ROOT_DIR%env"
SET "REQ_FILE=%ROOT_DIR%app\requirements.txt"

:: Check for Python
python --version >nul 2>&1
if !ERRORLEVEL! neq 0 (
    echo [ERROR] Python is not installed or not in your PATH. 
    echo Please install Python 3.10+ and try again.
    pause
    exit /b 1
)

echo.
echo [1/4] Setting up virtual environment "env"...

:: Clean up broken env if it exists
if exist "%ENV_DIR%" (
    if not exist "%ENV_DIR%\Scripts\python.exe" (
        echo [WARNING] Existing 'env' folder is broken. Deleting...
        rd /s /q "%ENV_DIR%"
    )
)

:: Create env if missing
if not exist "%ENV_DIR%" (
    echo Creating new virtual environment in "%ENV_DIR%"...
    python -m venv "%ENV_DIR%"
    if !ERRORLEVEL! neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo Using existing environment at "%ENV_DIR%"
)

echo.
echo [2/4] Activating environment...
call "%ENV_DIR%\Scripts\activate.bat"
if !ERRORLEVEL! neq 0 (
    echo [ERROR] Failed to activate environment.
    pause
    exit /b 1
)

echo.
echo [3/4] Detecting GPU and installing PyTorch...
python -m pip install --upgrade pip wheel
if !ERRORLEVEL! neq 0 ( echo [ERROR] Pip upgrade failed. & pause & exit /b 1 )

:: Auto-detection logic (Quoted paths)
set "CUDA_INDEX=cu124"
set "GPU_FILE=%ROOT_DIR%gpu_name.txt"
nvidia-smi --query-gpu=name --format=csv,noheader > "%GPU_FILE%" 2>nul
if !ERRORLEVEL! == 0 (
    set /p GPU_NAME=<"%GPU_FILE%"
    echo Detected GPU: !GPU_NAME!
    echo !GPU_NAME! | findstr /i "RTX.50" >nul
    if !ERRORLEVEL! == 0 (
        echo [INFO] Blackwell (50-series) detected. Using CUDA 12.8...
        set "CUDA_INDEX=cu128"
    ) else (
        echo [INFO] Standard NVIDIA GPU detected. Using stable CUDA 12.4...
    )
    del "%GPU_FILE%"
) else (
    echo [WARNING] nvidia-smi not found. Defaulting to stable CUDA 12.4...
)

echo Installing PyTorch with %CUDA_INDEX%...
python -m pip install --force-reinstall --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/%CUDA_INDEX%
if !ERRORLEVEL! neq 0 ( echo [ERROR] PyTorch installation failed. & pause & exit /b 1 )

echo.
echo [4/4] Installing application dependencies...
python -m pip install -r "%REQ_FILE%"
if !ERRORLEVEL! neq 0 ( echo [ERROR] Dependencies installation failed. & pause & exit /b 1 )

echo.
echo ===================================================
echo Verifying CUDA acceleration...
python -c "import torch; print('CUDA Detection Result: ' + ('SUCCESS' if torch.cuda.is_available() else 'FAILED')); exit(0 if torch.cuda.is_available() else 1)"
if !ERRORLEVEL! neq 0 (
    echo [CRITICAL ERROR] The GPU was not detected in this environment. 
    echo Please ensure you have NVIDIA drivers installed and try running this script again.
)
echo.
echo ===================================================
echo Installation Complete! 
echo You can now launch the application using run.bat
echo ===================================================
pause
