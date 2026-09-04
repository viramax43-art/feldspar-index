#Requires -Version 5.1
param([string]$TaskPassword = "")
$ErrorActionPreference = "Continue"
$env:PATH = "C:\ProgramData\Anaconda3;C:\ProgramData\Anaconda3\Scripts;C:\ProgramData\Anaconda3\Library\bin;" + $env:PATH
$Root = "C:\work\app"
$venvPy = Join-Path $Root ".venv\Scripts\python.exe"
$log = Join-Path $Root "logs\boot.log"
New-Item -ItemType Directory -Force -Path "$Root\logs" | Out-Null
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
Log "=== boot ==="
Set-DotEnv "WEB_ENABLED" "true"
Set-DotEnv "WEB_HOST" "0.0.0.0"
Set-DotEnv "WEB_PORT" "8765"
Set-DotEnv "WEB_PUBLIC_URL" "http://92.126.22.128"
Set-DotEnv "TTS_ENGINE" "voxcpm2"
Set-DotEnv "DEVICE" "cuda"
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Get-Process cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Log "pip eval_type_backport"
& $venvPy -m pip install -q eval-type-backport
Log "ipconfig:"
cmd /c ipconfig | Select-String "IPv4"
cmd /c "netsh advfirewall firewall add rule name=AppWeb80 dir=in action=allow protocol=TCP localport=80"
cmd /c "netsh advfirewall firewall add rule name=AppWeb8765 dir=in action=allow protocol=TCP localport=8765"
Log "compileall"
& $venvPy -m compileall -q "$Root\app"
if ($LASTEXITCODE -ne 0) { Log "COMPILE FAIL $LASTEXITCODE" }
& $venvPy "$Root\scripts\upnp_map.py" 80 8765 192.168.1.147
& $venvPy "$Root\scripts\upnp_map.py" 8765 8765 192.168.1.147
$out = Join-Path $Root "logs\app.main.out.log"
$err = Join-Path $Root "logs\app.main.err.log"
Remove-Item $out,$err -Force -ErrorAction SilentlyContinue
$runner = Join-Path $Root "logs\run_app.ps1"
@"
`$ErrorActionPreference = 'Continue'
`$env:PATH = 'C:\ProgramData\Anaconda3;C:\ProgramData\Anaconda3\Scripts;C:\ProgramData\Anaconda3\Library\bin;' + `$env:PATH
`$env:PYTHONFAULTHANDLER = '1'
`$env:PYTHONUNBUFFERED = '1'
Set-Location '$Root'
& '$venvPy' -u -m app.main *>> '$out'
"@ | Set-Content -Path $runner -Encoding UTF8
Log "start detached python"
& $venvPy "$Root\scripts\detach_main.py"
Start-Sleep -Seconds 20
Log "out:"
if (Test-Path $out) { Get-Content $out -Tail 80 | ForEach-Object { Log $_ } }
Log "listen 80/8765:"
cmd /c "netstat -ano | findstr LISTENING | findstr `":8765`""
cmd /c "netstat -ano | findstr LISTENING | findstr `":80 `""
try {
  $h = Invoke-WebRequest -Uri "http://127.0.0.1:8765/api/health" -UseBasicParsing -TimeoutSec 5
  Log ("health8765 " + $h.StatusCode + " " + $h.Content)
} catch { Log ("health8765 FAIL " + $_) }
try {
  $h = Invoke-WebRequest -Uri "http://127.0.0.1/api/health" -UseBasicParsing -TimeoutSec 5
  Log ("health80 " + $h.StatusCode + " " + $h.Content)
} catch { Log ("health80 FAIL " + $_) }
Log "=== boot done ==="
