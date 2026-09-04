#Requires -Version 5.1
# Upgrade Assistant code on the GPU host, keep VoxCPM2, run Telegram bot (no public site).
$ErrorActionPreference = "Continue"
$env:PATH = "C:\ProgramData\Anaconda3;C:\ProgramData\Anaconda3\Scripts;C:\ProgramData\Anaconda3\Library\bin;" + $env:PATH
$Root = "C:\work\app"
$venvPy = Join-Path $Root ".venv\Scripts\python.exe"
New-Item -ItemType Directory -Force -Path "$Root\logs" | Out-Null
$log = Join-Path $Root "logs\boot_bot.log"
function Log($m) {
  $line = "$(Get-Date -Format o) $m"
  Add-Content -Path $log -Value $line -Encoding UTF8
  Write-Output $line
}
function Set-DotEnv([string]$key, [string]$value) {
  $envPath = Join-Path $Root ".env"
  if (-not (Test-Path $envPath)) { Log "ERROR: .env missing"; return }
  $raw = Get-Content $envPath -Raw -Encoding UTF8
  if ($null -eq $raw) { $raw = "" }
  $pattern = "(?m)^$key=.*$"
  if ($raw -match $pattern) {
    $raw = [regex]::Replace($raw, $pattern, "$key=$value")
  } else {
    $raw = $raw.TrimEnd() + "`r`n$key=$value`r`n"
  }
  Set-Content -Path $envPath -Value $raw -Encoding UTF8
}
Log "=== boot bot ==="
Set-DotEnv "WEB_ENABLED" "false"
Set-DotEnv "TTS_ENGINE" "voxcpm2"
Set-DotEnv "DEVICE" "cuda"
Set-DotEnv "STT_DEVICE" "cpu"
Set-DotEnv "VIDEO_DUB_SEPARATION_DEVICE" "cpu"
Set-DotEnv "SILERO_DEVICE" "cpu"
Set-DotEnv "VOXCPM_MODEL_ID" "openbmb/VoxCPM2"
Set-DotEnv "TELEGRAM_DOWNLOAD_TIMEOUT_SEC" "300"
Set-DotEnv "VIDEO_DUB_MAX_SPEED" "1.15"
Set-DotEnv "VIDEO_DUB_PHRASE_GAP_SEC" "0.15"
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Get-Process cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Log "pip requirements-server.txt"
& $venvPy -m pip install -q -r "$Root\requirements-server.txt"
Log "compileall"
& $venvPy -m compileall -q "$Root\app"
if ($LASTEXITCODE -ne 0) { Log "COMPILE FAIL $LASTEXITCODE" }
$out = Join-Path $Root "logs\app.main.out.log"
Remove-Item $out -Force -ErrorAction SilentlyContinue
Log "detach app.main"
& $venvPy "$Root\scripts\detach_main.py"
Start-Sleep -Seconds 22
Log "out:"
if (Test-Path $out) { Get-Content $out -Tail 50 | ForEach-Object { Log $_ } }
Log "=== boot bot done ==="
