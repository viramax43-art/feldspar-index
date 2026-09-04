#Requires -Version 5.1
# Install Assistant + VoxCPM2 and expose the web UI on all interfaces (port 80 + 8765).
$ErrorActionPreference = "Stop"

function Test-Admin {
  $id = [Security.Principal.WindowsIdentity]::GetCurrent()
  $p = New-Object Security.Principal.WindowsPrincipal($id)
  return $p.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-Admin)) {
  $self = $MyInvocation.MyCommand.Path
  Start-Process powershell -Verb RunAs -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$self`""
  exit
}

$Root = "C:\work\app"
if (-not (Test-Path (Join-Path $Root "app\main.py"))) {
  $Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
}
Set-Location $Root
New-Item -ItemType Directory -Force -Path "$Root\logs" | Out-Null

function Get-PublicIp {
  try {
    return (Invoke-RestMethod -Uri "https://api.ipify.org" -TimeoutSec 8).Trim()
  } catch {
    return ""
  }
}

function Get-LanIp {
  $addr = Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" -and $_.IPAddress -notlike "26.*" } |
    Select-Object -First 1 -ExpandProperty IPAddress
  if ($addr) { return $addr }
  return "192.168.1.147"
}

$publicIp = Get-PublicIp
$lanIp = Get-LanIp
$publicUrl = if ($publicIp) { "http://$publicIp" } else { "http://$lanIp" }

function Ensure-Ffmpeg {
  if (Get-Command ffmpeg -ErrorAction SilentlyContinue) { return }
  Write-Host "Installing FFmpeg via winget..."
  winget install -e --id Gyan.FFmpeg --accept-package-agreements --accept-source-agreements --silent
  $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
}

$pyCmd = "python"
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
  if (Get-Command py -ErrorAction SilentlyContinue) { $pyCmd = "py" }
  else {
    Write-Host "Installing Python 3.12 via winget..."
    winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements --silent
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
    $pyCmd = "python"
  }
}
Ensure-Ffmpeg

if (-not (Test-Path "$Root\.venv\Scripts\python.exe")) {
  & $pyCmd -m venv .venv
}
$venvPy = "$Root\.venv\Scripts\python.exe"
& $venvPy -m pip install -U pip wheel
try {
  & $venvPy -c "import torch; assert torch.cuda.is_available()"
} catch {
  & $venvPy -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
}
& $venvPy -m pip install -r "$Root\requirements-server.txt"
if (-not (Test-Path "$Root\assets\tts\silero\v5_5_ru.pt")) {
  & $venvPy "$Root\scripts\download_silero.py"
}

$envPath = "$Root\.env"
if (Test-Path $envPath) {
  $raw = Get-Content $envPath -Raw -Encoding UTF8
  $map = @{
    "TTS_ENGINE" = "voxcpm2"
    "WEB_ENABLED" = "true"
    "WEB_HOST" = "0.0.0.0"
    "WEB_PORT" = "80"
    "WEB_PUBLIC_URL" = $publicUrl
    "STT_DEVICE" = "cpu"
    "VIDEO_DUB_SEPARATION_DEVICE" = "cpu"
    "SILERO_DEVICE" = "cpu"
    "DEVICE" = "cuda"
    "VOXCPM_MODEL_ID" = "openbmb/VoxCPM2"
  }
  foreach ($k in $map.Keys) {
    if ($raw -match "(?m)^$k=") {
      $raw = [regex]::Replace($raw, "(?m)^$k=.*$", "$k=$($map[$k])")
    } else {
      $raw = $raw.TrimEnd() + "`r`n$k=$($map[$k])`r`n"
    }
  }
  Set-Content -Path $envPath -Value $raw -Encoding UTF8
}

Get-NetFirewallRule -DisplayName "Assistant Web" -ErrorAction SilentlyContinue | Remove-NetFirewallRule -ErrorAction SilentlyContinue
New-NetFirewallRule -DisplayName "Assistant Web" -Direction Inbound -Protocol TCP -LocalPort 80,8765 -Action Allow | Out-Null
netsh advfirewall firewall add rule name="Assistant Web 80" dir=in action=allow protocol=TCP localport=80 | Out-Null
netsh advfirewall firewall add rule name="Assistant Web 8765" dir=in action=allow protocol=TCP localport=8765 | Out-Null

Get-Process python -ErrorAction SilentlyContinue | Where-Object {
  $_.Path -like "*voice_caller*"
} | Stop-Process -Force -ErrorAction SilentlyContinue

Write-Host "Starting Assistant on 0.0.0.0:80 ..."
Start-Process -FilePath $venvPy -ArgumentList "-m","app.main" -WorkingDirectory $Root
Write-Host "LAN:    http://$lanIp"
if ($publicIp) { Write-Host "Public: http://$publicIp" }
Write-Host "Radmin: http://26.83.247.236"
Write-Host "Local:  http://127.0.0.1"
pause
