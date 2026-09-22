@echo off
title OmniVoice Demo Server (Tieng Viet)
cd /d "%~dp0"

echo ========================================================
echo        Dang khoi dong OmniVoice Web Demo (Tieng Viet)...
echo ========================================================
echo.

echo [He thong] Dang kiem tra va don dep tien trinh cu (Port 8001)...
for /f "tokens=5" %%a in ('netstat -aon ^| find ":8001" ^| find "LISTENING"') do (
    echo [He thong] Tim thay tien trinh dang chay (PID: %%a). Dang tat...
    taskkill /F /PID %%a >nul 2>&1
)
echo [He thong] Da don dep xong! Dang khoi dong Giao dien...
echo.

rem Port 8001 is only the local experimental Gradio UI.  The production
rem Auto_YT worker is a separate authenticated service on 127.0.0.1:8011.
uv run omnivoice-demo --ip 127.0.0.1 --port 8001

echo.
echo Giao dien da bi dong.
pause
