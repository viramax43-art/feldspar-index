$ErrorActionPreference = "Continue"
$env:PATH = "C:\ProgramData\Anaconda3;C:\ProgramData\Anaconda3\Scripts;C:\ProgramData\Anaconda3\Library\bin;" + $env:PATH
$Root = "C:\work\app"
$venvPy = Join-Path $Root ".venv\Scripts\python.exe"
$log = Join-Path $Root "logs\finish_public.log"
New-Item -ItemType Directory -Force -Path (Join-Path $Root "logs") | Out-Null
function Log($m) {
  $line = "$(Get-Date -Format o) $m"
  Add-Content -Path $log -Value $line -Encoding UTF8
  Write-Output $line
}
Log "=== finish public v2 ==="
if (-not (Test-Path $venvPy)) {
  Log "create venv via Anaconda"
  & C:\ProgramData\Anaconda3\python.exe -m venv "$Root\.venv"
}
Log "pip upgrade"
$env:PIP_PROGRESS_BAR = "off"
& $venvPy -m pip install -U pip wheel -q --progress-bar off
Log "torch"
& $venvPy -c "import torch; print(torch.__version__, torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) {
  Log "install torch cu124"
  & $venvPy -m pip install --progress-bar off torch torchaudio --index-url https://download.pytorch.org/whl/cu124
}
Log "requirements-server"
& $venvPy -m pip install --progress-bar off -r "$Root\requirements-server.txt"
Log "torch check"
& $venvPy -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
Log "firewall"
cmd /c netsh advfirewall firewall add rule name=AppWeb80 dir=in action=allow protocol=TCP localport=80
cmd /c netsh advfirewall firewall add rule name=AppWeb8765 dir=in action=allow protocol=TCP localport=8765
Log "upnp"
& $venvPy "$Root\scripts\upnp_map.py" 80 8765
& $venvPy "$Root\scripts\upnp_map.py" 8765 8765
Get-Process cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Log "start app.main"
$out = Join-Path $Root "logs\app.main.out.log"
$err = Join-Path $Root "logs\app.main.err.log"
if (Test-Path $out) { Remove-Item $out -Force -ErrorAction SilentlyContinue }
if (Test-Path $err) { Remove-Item $err -Force -ErrorAction SilentlyContinue }
Start-Process -FilePath $venvPy -ArgumentList "-u","-m","app.main" -WorkingDirectory $Root -RedirectStandardOutput $out -RedirectStandardError $err -WindowStyle Hidden
Start-Sleep -Seconds 8
Log "app.main.out tail:"
if (Test-Path $out) { Get-Content $out -Tail 30 | ForEach-Object { Log $_ } }
if (Test-Path $err) { Get-Content $err -Tail 40 | ForEach-Object { Log ("ERR " + $_) } }
Log "=== finish done ==="
