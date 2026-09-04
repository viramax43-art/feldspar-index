@echo off
:: Elevate and install/start public web UI
net session >nul 2>&1
if %errorlevel% neq 0 (
  powershell -NoProfile -Command "Start-Process '%~f0' -Verb RunAs"
  exit /b
)
cd /d "C:\work\app"
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\work\app\scripts\bootstrap_server.ps1"
if errorlevel 1 pause
