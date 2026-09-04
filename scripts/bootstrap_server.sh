#!/usr/bin/env bash
# Bootstrap Voice Caller + VoxCPM2 on a Linux GPU box.
set -euo pipefail

ROOT="${VOICE_CALLER_ROOT:-$HOME/voice_caller}"
cd "$ROOT"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 python3-venv python3-pip ffmpeg git build-essential
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required" >&2
  exit 1
fi

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip wheel

if ! python -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
  python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
fi

python -m pip install -r requirements-server.txt

if [[ ! -f assets/tts/silero/v5_5_ru.pt ]]; then
  python scripts/download_silero.py || true
fi

if command -v ufw >/dev/null 2>&1; then
  sudo ufw allow 8765/tcp || true
fi
sudo iptables -I INPUT -p tcp --dport 8765 -j ACCEPT 2>/dev/null || true

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0),
          "vram_gb", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
PY

UNIT_DIR="${HOME}/.config/systemd/user"
mkdir -p "$UNIT_DIR"
cat > "$UNIT_DIR/voice-caller.service" <<EOF
[Unit]
Description=Voice Caller bot + VoxCPM2 dub studio
After=network.target

[Service]
Type=simple
WorkingDirectory=${ROOT}
Environment=CUDA_DEVICE_ORDER=PCI_BUS_ID
Environment=CUDA_VISIBLE_DEVICES=0
ExecStart=${ROOT}/.venv/bin/python -m app.main
Restart=on-failure
RestartSec=8

[Install]
WantedBy=default.target
EOF

mkdir -p "$ROOT/logs"
systemctl --user daemon-reload || true
loginctl enable-linger "$(whoami)" 2>/dev/null || true
systemctl --user enable --now voice-caller.service || {
  echo "systemd user units unavailable; starting in screen/tmux fallback"
  nohup "$ROOT/.venv/bin/python" -m app.main > "$ROOT/logs/server.log" 2>&1 &
  echo $! > "$ROOT/logs/server.pid"
}

echo "Voice Caller started. Web: see WEB_PUBLIC_URL in .env"
