@echo off
cd /d C:\work\app
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\work\app\scripts\start_public_site.ps1"
if errorlevel 1 pause
pause
