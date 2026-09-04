#Requires -Version 5.1
# Bind Assistant web studio to 0.0.0.0 and expose it on the WAN IP (no Cloudflare domain).
$ErrorActionPreference = "Continue"
$Root = "C:\work\app"
Set-Location $Root
New-Item -ItemType Directory -Force -Path "$Root\logs", "$Root\bin" | Out-Null
$log = "$Root\logs\public_ip_start.log"
function Log($m) {
  $line = "$(Get-Date -Format o) $m"
  Add-Content -Path $log -Value $line -Encoding UTF8
  Write-Output $line
}
Log "=== start public IP ==="

$PublicUrl = "http://92.126.22.128"
$LanIp = "192.168.1.147"

function Set-DotEnv([string]$key, [string]$value) {
  $envPath = Join-Path $Root ".env"
  if (-not (Test-Path $envPath)) {
    Log "ERROR: .env missing"
    return
  }
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

Log "patch .env"
Set-DotEnv "WEB_ENABLED" "true"
Set-DotEnv "WEB_HOST" "0.0.0.0"
Set-DotEnv "WEB_PORT" "8765"
Set-DotEnv "WEB_PUBLIC_URL" $PublicUrl
Set-DotEnv "TTS_ENGINE" "voxcpm2"
Set-DotEnv "DEVICE" "cuda"
Set-DotEnv "STT_DEVICE" "cpu"
Set-DotEnv "VIDEO_DUB_SEPARATION_DEVICE" "cpu"
Set-DotEnv "SILERO_DEVICE" "cpu"
Set-DotEnv "VOXCPM_MODEL_ID" "openbmb/VoxCPM2"

# Stop previous tunnel/domain frontends
Get-Process cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Get-Process python -ErrorAction SilentlyContinue | ForEach-Object {
  try {
    $p = Get-WmiObject Win32_Process -Filter ("ProcessId=" + $_.Id) -ErrorAction SilentlyContinue
    if ($p -and $p.CommandLine -and ($p.CommandLine -match "voice_caller|app.main")) {
      Log ("stop pid " + $_.Id)
      Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
    }
  } catch {}
}

$py312 = Join-Path $env:LocalAppData "Programs\Python\Python312\python.exe"
if (-not (Test-Path $py312)) {
  Log "installing Python 3.12 (current user)"
  $inst = Join-Path $env:TEMP "python-3.12.10-amd64.exe"
  try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe" -OutFile $inst -UseBasicParsing
    Start-Process -FilePath $inst -ArgumentList "/quiet InstallAllUsers=0 PrependPath=1 Include_test=0 Include_launcher=1 Shortcuts=0" -Wait
  } catch {
    Log ("python installer failed: " + $_)
  }
}

$basePy = $null
if (Test-Path $py312) { $basePy = $py312 }
elseif (Get-Command py -ErrorAction SilentlyContinue) {
  $basePy = "py"
}
elseif (Test-Path "C:\ProgramData\Anaconda3\python.exe") {
  $basePy = "C:\ProgramData\Anaconda3\python.exe"
}
Log ("base python = " + $basePy)
if (-not $basePy) { Log "ERROR: no python"; exit 1 }

$venvPy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
  Log "creating venv"
  if ($basePy -eq "py") { & py -3.12 -m venv "$Root\.venv" 2>$null; if (-not (Test-Path $venvPy)) { & py -3 -m venv "$Root\.venv" } }
  else { & $basePy -m venv "$Root\.venv" }
}
if (-not (Test-Path $venvPy)) { Log "ERROR: venv not created"; exit 1 }

Log "pip upgrade"
& $venvPy -m pip install -U pip wheel
Log "torch CUDA"
try {
  & $venvPy -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
} catch {
  Log "installing torch cu124"
  & $venvPy -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
}
Log "requirements-server.txt"
& $venvPy -m pip install -r "$Root\requirements-server.txt"

if (-not (Test-Path "$Root\assets\tts\silero\v5_5_ru.pt") -or ((Get-Item "$Root\assets\tts\silero\v5_5_ru.pt").Length -lt 1000)) {
  Log "download silero"
  & $venvPy "$Root\scripts\download_silero.py"
}

try {
  netsh advfirewall firewall add rule name="Assistant Web 80" dir=in action=allow protocol=TCP localport=80 | Out-Null
  netsh advfirewall firewall add rule name="Assistant Web 8765" dir=in action=allow protocol=TCP localport=8765 | Out-Null
  Log "firewall rules ok"
} catch { Log ("firewall: " + $_) }

Log "UPnP 80->8765 and 8765->8765"
& $venvPy "$Root\scripts\upnp_map.py" 80 8765
& $venvPy "$Root\scripts\upnp_map.py" 8765 8765

Log "start app.main"
$out = Join-Path $Root "logs\app.main.out.log"
$err = Join-Path $Root "logs\app.main.err.log"
Start-Process -FilePath $venvPy -ArgumentList "-u","-m","app.main" -WorkingDirectory $Root -RedirectStandardOutput $out -RedirectStandardError $err -WindowStyle Hidden
Set-Content -Path "$Root\logs\PUBLIC_URL.txt" -Value $PublicUrl -Encoding ASCII
Log "PUBLIC $PublicUrl"
Log "also try ${PublicUrl}:8765 if port 80 is not forwarded"
Log "=== done ==="
