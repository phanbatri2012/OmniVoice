@echo off
title OmniVoice FastAPI Server
cd /d "%~dp0"

echo ========================================================
echo        Khoi dong OmniVoice Worker (127.0.0.1:8011)...
echo ========================================================
echo.

echo [He thong] Dang kiem tra va don dep tien trinh cu (Port 8011)...
for /f "tokens=5" %%a in ('netstat -aon ^| find ":8011" ^| find "LISTENING"') do (
    echo [He thong] Tim thay tien trinh dang chay (PID: %%a). Dang tat...
    taskkill /F /PID %%a >nul 2>&1
)
echo [He thong] Da san sang khoi dong OmniVoice FastAPI Server...
echo.

uv run uvicorn api_server:app --host 127.0.0.1 --port 8011

echo.
pause
