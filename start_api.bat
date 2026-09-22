@echo off
title OmniVoice FastAPI Server
cd /d "%~dp0"

echo ========================================================
echo        Khoi dong OmniVoice Worker (127.0.0.1:8011)...
echo ========================================================
echo.

if "%AUTO_YT_OMNIVOICE_TOKEN%"=="" (
  echo AUTO_YT_OMNIVOICE_TOKEN is required. Start this worker from Auto_YT.
  exit /b 1
)

uv run uvicorn api_server:app --host 127.0.0.1 --port 8011

echo.
pause
