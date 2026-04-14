@echo off
setlocal ENABLEDELAYEDEXPANSION

echo ===================================================
echo Amin OmniVoice Installer
echo ===================================================

:: Check for Python
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not installed or not in your PATH. 
    echo Please install Python 3.10+ and try again.
    pause
    exit /b 1
)

echo.
echo [1/4] Setting up virtual environment "env"...
if exist env (
    :: Check if the environment is broken (happens when moving folders)
    if not exist env\Scripts\python.exe (
        echo [WARNING] Existing 'env' folder is broken or incomplete.
        echo Deleting broken environment and starting fresh...
        rd /s /q env
    )
)

if not exist env (
    echo Creating new virtual environment...
    python -m venv env
    if !ERRORLEVEL! neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo Using existing virtual environment.
    echo (If you moved this folder, delete the 'env' folder and run this again!)
)

echo.
echo [2/4] Activating environment...
call env\Scripts\activate

echo [3/4] Detecting GPU and installing PyTorch...
python -m pip install --upgrade pip wheel
if %ERRORLEVEL% neq 0 ( echo [ERROR] Pip upgrade failed. & pause & exit /b 1 )

:: Auto-detection logic
set "CUDA_INDEX=cu124"
nvidia-smi --query-gpu=name --format=csv,noheader > gpu_name.txt 2>nul
if %ERRORLEVEL% == 0 (
    set /p GPU_NAME=<gpu_name.txt
    echo Detected GPU: !GPU_NAME!
    
    echo !GPU_NAME! | findstr /i "RTX.50" >nul
    if !ERRORLEVEL! == 0 (
        echo [INFO] Blackwell (50-series) detected. Using CUDA 12.8...
        set "CUDA_INDEX=cu128"
    ) else (
        echo [INFO] Standard NVIDIA GPU detected. Using stable CUDA 12.4...
    )
    del gpu_name.txt
) else (
    echo [WARNING] nvidia-smi not found. Defaulting to stable CUDA 12.4...
)

echo Installing PyTorch with %CUDA_INDEX%...
python -m pip install --force-reinstall --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/%CUDA_INDEX%
if %ERRORLEVEL% neq 0 ( echo [ERROR] PyTorch installation failed. & pause & exit /b 1 )

echo.
echo [4/4] Installing application dependencies...
python -m pip install -r app\requirements.txt
if %ERRORLEVEL% neq 0 ( echo [ERROR] Dependencies installation failed. & pause & exit /b 1 )

echo.
echo ===================================================
echo Verifying CUDA acceleration...
python -c "import torch; print('CUDA Detection Result: ' + ('SUCCESS' if torch.cuda.is_available() else 'FAILED')); exit(0 if torch.cuda.is_available() else 1)"
if %ERRORLEVEL% neq 0 (
    echo [CRITICAL ERROR] The GPU was not detected in this environment. 
    echo Please ensure you have NVIDIA drivers installed and try running this script again.
)
echo.
echo ===================================================
echo Installation Complete! 
echo You can now launch the application using run.bat
echo ===================================================
pause
