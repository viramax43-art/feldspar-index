#Requires -Version 5.1
# Start Assistant + Cloudflare tunnel on the GPU host (no admin / port 8765).
$ErrorActionPreference = "Continue"
$Root = "C:\work\app"
Set-Location $Root
New-Item -ItemType Directory -Force -Path "$Root\logs", "$Root\bin" | Out-Null
$log = "$Root\logs\server_start.log"
function Log($m) {
  $line = "$(Get-Date -Format o) $m"
  Add-Content $log $line
  Write-Host $line
}

Log "=== start ==="

$pyCmd = $null
if (Get-Command python -ErrorAction SilentlyContinue) { $pyCmd = "python" }
elseif (Get-Command py -ErrorAction SilentlyContinue) { $pyCmd = "py" }
elseif (Test-Path "$env:LocalAppData\Programs\Python\Python312\python.exe") {
  $pyCmd = "$env:LocalAppData\Programs\Python\Python312\python.exe"
}
if (-not $pyCmd) {
  Log "Installing Python 3.12..."
  winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements --silent
  $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
  $pyCmd = "python"
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
  Log "Installing FFmpeg..."
  winget install -e --id Gyan.FFmpeg --accept-package-agreements --accept-source-agreements --silent
  $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
}

if (-not (Test-Path "$Root\.venv\Scripts\python.exe")) {
  Log "Creating venv..."
  & $pyCmd -m venv .venv
}
$venvPy = "$Root\.venv\Scripts\python.exe"
Log "pip upgrade"
& $venvPy -m pip install -U pip wheel
try {
  & $venvPy -c "import torch; assert torch.cuda.is_available()"
} catch {
  Log "Installing torch CUDA..."
  & $venvPy -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
}
Log "Installing requirements-server.txt"
& $venvPy -m pip install -r "$Root\requirements-server.txt"

$envPath = "$Root\.env"
if (Test-Path $envPath) {
  $raw = Get-Content $envPath -Raw -Encoding UTF8
  $map = @{
    TTS_ENGINE = "voxcpm2"
    WEB_ENABLED = "true"
    WEB_HOST = "0.0.0.0"
    WEB_PORT = "8765"
    DEVICE = "cuda"
    STT_DEVICE = "cpu"
    VIDEO_DUB_SEPARATION_DEVICE = "cpu"
    SILERO_DEVICE = "cpu"
    VIDEO_DUB_MIX_MODE = "replace"
    VIDEO_DUB_SPEECH_MASK_GAIN = "0.02"
    VIDEO_DUB_VOCAL_LEAK = "0.92"
    VIDEO_DUB_DUCK_FLOOR = "0.06"
  }
  foreach ($k in $map.Keys) {
    if ($raw -match "(?m)^$k=") {
      $raw = [regex]::Replace($raw, "(?m)^$k=.*$", "$k=$($map[$k])")
    } else {
      $raw = $raw.TrimEnd() + "`r`n$k=$($map[$k])`r`n"
    }
  }
  Set-Content $envPath $raw -Encoding UTF8
}

try {
  netsh advfirewall firewall add rule name="Assistant Web 8765" dir=in action=allow protocol=TCP localport=8765 | Out-Null
} catch {}

Get-CimInstance Win32_Process -Filter "name='python.exe'" -ErrorAction SilentlyContinue |
  Where-Object { $_.CommandLine -like "*voice_caller*app.main*" } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Get-Process cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

$cf = "$Root\bin\cloudflared.exe"
if (-not (Test-Path $cf)) {
  Log "Downloading cloudflared..."
  Invoke-WebRequest -Uri "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" -OutFile $cf -UseBasicParsing
}

Log "Starting app.main"
Start-Process -FilePath $venvPy -ArgumentList "-m", "app.main" -WorkingDirectory $Root -WindowStyle Minimized
Start-Sleep -Seconds 10

Log "Starting cloudflared"
$cfLog = "$Root\logs\cloudflared.log"
Remove-Item $cfLog -Force -ErrorAction SilentlyContinue
Start-Process -FilePath $cf -ArgumentList @("tunnel", "--url", "http://127.0.0.1:8765", "--no-autoupdate") -WorkingDirectory $Root -RedirectStandardError $cfLog -WindowStyle Minimized

$url = $null
for ($i = 0; $i -lt 90; $i++) {
  Start-Sleep -Seconds 2
  if (Test-Path $cfLog) {
    $txt = Get-Content $cfLog -Raw -ErrorAction SilentlyContinue
    if ($txt -match "https://[a-zA-Z0-9-]+\.trycloudflare\.com") {
      $url = $Matches[0]
      break
    }
  }
}

if ($url) {
  Set-Content "$Root\logs\PUBLIC_URL.txt" $url -Encoding ASCII
  Set-Content "C:\Users\artemka2\Desktop\PUBLIC_URL.txt" $url -Encoding ASCII
  if (Test-Path $envPath) {
    $raw = Get-Content $envPath -Raw -Encoding UTF8
    if ($raw -match "(?m)^WEB_PUBLIC_URL=") {
      $raw = [regex]::Replace($raw, "(?m)^WEB_PUBLIC_URL=.*$", "WEB_PUBLIC_URL=$url")
    } else {
      $raw = $raw.TrimEnd() + "`r`nWEB_PUBLIC_URL=$url`r`n"
    }
    Set-Content $envPath $raw -Encoding UTF8
  }
  Log "PUBLIC $url"
  try { Start-Process $url } catch {}
} else {
  Log "No cloudflare URL yet — check logs\cloudflared.log"
}
Log "=== done ==="
