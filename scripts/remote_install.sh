#!/usr/bin/env bash
set -euo pipefail

export HOME="${HOME:-/home/debian}"
cd "$HOME"

echo "[1/7] Extract project..."
rm -rf "$HOME/voicer"
mkdir -p "$HOME/voicer"
tar -xzf "$HOME/voicer_code.tar.gz" -C "$HOME/voicer"
cd "$HOME/voicer"
cp -f "$HOME/voicer.server.env" "$HOME/voicer/.env"
chmod 600 "$HOME/voicer/.env"

echo "[2/7] Ensure Miniconda..."
if [ ! -x "$HOME/miniconda3/bin/conda" ]; then
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
fi
# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

echo "[3/7] Create Python 3.11 env..."
if [ ! -d "$HOME/miniconda3/envs/voicer" ]; then
  conda create -y -n voicer python=3.11
fi
conda activate voicer
python --version

echo "[4/7] Install PyTorch CPU + deps..."
pip install -U pip wheel setuptools
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

echo "[5/7] Download Silero + certs..."
python scripts/download_silero.py
python scripts/setup_gigachat_certs.py || true

echo "[6/7] Create systemd user service helper script..."
cat > "$HOME/voicer/run_bot.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd /home/debian/voicer
source /home/debian/miniconda3/etc/profile.d/conda.sh
conda activate voicer
export PYTHONUNBUFFERED=1
exec python -m app.main
EOF
chmod +x "$HOME/voicer/run_bot.sh"

echo "[7/7] DONE"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
ls -lh assets/tts/silero/v5_5_ru.pt || true
ls -lh data/users/1327953308/ || true
