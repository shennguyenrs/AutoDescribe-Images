@echo off
echo ========================================
echo   AutoDescribe Images - Web Interface
echo ========================================
echo.

cd /d "%~dp0"

echo [1/4] Checking if UV is installed...
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo UV not found. Installing UV...
    powershell -ExecutionPolicy ByPass -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install UV.
        pause
        exit /b 1
    )
    echo UV installed successfully!
    echo Please restart this script for the changes to take effect.
    pause
    exit /b 0
) else (
    echo UV is already installed.
)

echo.
echo [2/4] Updating repository...
git pull

echo.
echo [3/4] Checking dependencies...
uv sync

echo.
echo [4/4] Launching web interface...
echo.
uv run python -m image_describer --web

pause
