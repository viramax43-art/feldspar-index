@echo off
chcp 65001 >nul
REM ============================================================
REM  полный перезапуск бота (чтобы подхватить код/.env)
REM ============================================================
REM  Код Python НЕ подхватывается на лету — нужен рестарт процесса.
REM  Всегда запускай ТОЛЬКО через этот файл или:
REM    .venv\Scripts\python.exe -u run_bot.py
REM  Не используй: python run_bot.py  /  python -m app.main
REM ============================================================

cd /d "%~dp0"

echo.
echo [1/2] Останавливаю все run_bot.py ...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$procs = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match 'run_bot' }; ^
   foreach ($p in $procs) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }; ^
   Start-Sleep -Seconds 2; ^
   $left = @(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match 'run_bot' }); ^
   if ($left.Count -gt 0) { Write-Host ('ERROR: still alive PID ' + ($left.ProcessId -join ', ')); exit 1 } ^
   else { Write-Host 'OK: old processes stopped.' }"

if errorlevel 1 (
  echo Не удалось остановить старый бот. Закрой окна вручную и повтори.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: нет .venv\Scripts\python.exe
  pause
  exit /b 1
)

echo [2/2] Запускаю бота (.venv^)...
echo Дождись строки: Start polling
echo Остановка: Ctrl+C в этом окне
echo.
".venv\Scripts\python.exe" -u run_bot.py
