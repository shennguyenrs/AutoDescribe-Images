@echo off
echo ========================================
echo   AutoDescribe Images - Web Interface
echo ========================================
echo.

cd /d "%~dp0"

echo [1/3] Updating repository...
git pull

echo.
echo [2/3] Checking dependencies...
uv sync

echo.
echo [3/3] Launching web interface...
echo.
uv run python -m image_describer --web

pause
