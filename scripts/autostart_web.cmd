@echo off
REM No-admin autostart: bind all interfaces on 8765
cd /d C:\work\app
if exist .venv\Scripts\python.exe (
  start "VoiceCaller" .venv\Scripts\python.exe -m app.main
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "C:\work\app\scripts\bootstrap_server.ps1"
)
