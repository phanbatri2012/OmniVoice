@echo off
title OmniVoice Demo Server
cd /d "%~dp0"

echo ========================================================
echo        Dang khoi dong OmniVoice Web Demo...
echo ========================================================
echo.

uv run omnivoice-demo --ip 127.0.0.1 --port 8001

echo.
echo Giao dien da bi dong.
pause
