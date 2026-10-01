@echo off
title Dung he thong OmniVoice
echo ========================================================
echo        Dang dung he thong OmniVoice Web Demo...
echo ========================================================
echo.

set "FOUND=0"
for /f "tokens=5" %%a in ('netstat -aon ^| find ":8001" ^| find "LISTENING"') do (
    echo [He thong] Tim thay OmniVoice Web Demo (Port 8001, PID: %%a). Dang tat...
    taskkill /F /PID %%a >nul 2>&1
    set "FOUND=1"
)
for /f "tokens=5" %%a in ('netstat -aon ^| find ":8011" ^| find "LISTENING"') do (
    echo [He thong] Tim thay OmniVoice Worker (Port 8011, PID: %%a). Dang tat...
    taskkill /F /PID %%a >nul 2>&1
    set "FOUND=1"
)

if "%FOUND%"=="0" (
    echo [He thong] Khong co he thong OmniVoice nao (Port 8001/8011) dang chay.
) else (
    echo [He thong] Da tat he thong OmniVoice thanh cong!
)

echo.
pause
