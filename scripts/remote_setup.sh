#!/usr/bin/env bash
set -euo pipefail

export HOME="${HOME:-/home/debian}"
cd "$HOME"

if [ ! -x "$HOME/miniconda3/bin/conda" ]; then
  echo "[setup] Installing Miniconda..."
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
fi

# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda --version

if [ ! -d "$HOME/miniconda3/envs/voicer" ]; then
  echo "[setup] Creating conda env voicer (Python 3.11)..."
  conda create -y -n voicer python=3.11
fi

conda activate voicer
python --version
mkdir -p "$HOME/voicer"
echo "[setup] READY"
