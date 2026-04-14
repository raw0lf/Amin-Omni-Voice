@echo off
setlocal

echo ===================================================
echo Amin OmniVoice Launcher
echo ===================================================

if not exist env\Scripts\activate (
    echo [ERROR] Virtual environment 'env' not found!
    echo Please run install.bat first to download the necessary dependencies.
    pause
    exit /b 1
)

echo Activating environment...
call env\Scripts\activate

echo.
echo Configuring parameters...
set PYTHONUNBUFFERED=1
set HF_HUB_ENABLE_HF_TRANSFER=1
set OMNIVOICE_PORT=7860
set OMNIVOICE_DEVICE=cuda

:: Set the Cache directories precisely so it finds the model you already downloaded!
:: %~dp0 dynamically resolves to the directory where this run.bat is located.
set HF_HOME=%~dp0cache\HF_HOME
set TORCH_HOME=%~dp0cache\TORCH_HOME
set GRADIO_TEMP_DIR=%~dp0cache\GRADIO_TEMP_DIR

:: Smart Offline Mode Check
if exist "%HF_HOME%\hub\models--k2-fsa--OmniVoice" (
    echo Local model cache detected. Enabling fast offline boot...
    set HF_HUB_OFFLINE=1
) else (
    echo First-time launch detected!
    echo Proceeding ONLINE to download the required models...
)

echo.
echo Launching OmniVoice Studio locally...
echo (The browser interface should automatically open shortly)
echo.

python app\app.py

pause
