#!/usr/bin/env python3
"""Клонирует MockingBird в third_party/, ставит отдельный venv и качает веса.

XTTS / data/finetune не трогает.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MB_DIR = ROOT / "third_party" / "MockingBird"
VENV_DIR = ROOT / "third_party" / "mockingbird-venv"
WEIGHTS = ROOT / "data" / "mockingbird"
REPO = "https://github.com/babysor/MockingBird.git"

HF_ENCODER = "https://huggingface.co/CorentinJ/SV2TTS/resolve/main/encoder.pt"
HF_VOCODER = "https://huggingface.co/CorentinJ/SV2TTS/resolve/main/vocoder.pt"
# Английский synthesizer RTVC — запасной, если китайский mandarin.pt не скачается
HF_SYNTH = "https://huggingface.co/CorentinJ/SV2TTS/resolve/main/synthesizer.pt"


def run(cmd: list[str], **kwargs) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, **kwargs)


def find_python() -> str:
    for args in (["py", "-3.11"], ["py", "-3.10"], ["py", "-3.9"]):
        try:
            out = subprocess.check_output(args + ["-c", "import sys; print(sys.executable)"], text=True)
            exe = out.strip()
            if exe:
                print("Python для MockingBird:", exe)
                return exe
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    print("Нет Python 3.9–3.11, беру текущий:", sys.executable)
    return sys.executable


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print("уже есть", dest)
        return
    print("качаю", url, "->", dest)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    print("  ", dest.stat().st_size, "bytes")


def patch_mockingbird_compat(mb_dir: Path) -> None:
    """Совместимость с английским RTVC synthesizer.pt + WaveRNN."""
    hp = mb_dir / "models" / "synthesizer" / "hparams.py"
    if hp.exists():
        text = hp.read_text(encoding="utf-8")
        text = text.replace("hop_size = 256", "hop_size = 200")
        text = text.replace("use_gst = True", "use_gst = False")
        text = text.replace(
            'tts_cleaner_names = ["basic_cleaners"]',
            'tts_cleaner_names = ["transliteration_cleaners"]',
        )
        hp.write_text(text, encoding="utf-8")
    symbols = mb_dir / "models" / "synthesizer" / "utils" / "symbols.py"
    if symbols.exists():
        text = symbols.read_text(encoding="utf-8")
        old = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890!\\'(),-.:;? "
        new = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!'\\\"(),-.:;? "
        if old in text:
            text = text.replace(old, new)
            symbols.write_text(text, encoding="utf-8")
    fade = mb_dir / "models" / "vocoder" / "wavernn" / "models" / "fatchord_version.py"
    if fade.exists():
        src = fade.read_text(encoding="utf-8")
        old = (
            "        fade_out = np.linspace(1, 0, 20 * self.hop_length)\n"
            "        output = output[:wave_len]\n"
            "        output[-20 * self.hop_length:] *= fade_out"
        )
        new = (
            "        output = output[:wave_len]\n"
            "        fade_n = min(len(output), 20 * int(self.hop_length))\n"
            "        if fade_n > 0:\n"
            "            fade_out = np.linspace(1, 0, fade_n)\n"
            "            output[-fade_n:] *= fade_out"
        )
        if old in src:
            fade.write_text(src.replace(old, new), encoding="utf-8")
            print("патч WaveRNN fade_out")
        elif "fade_n = min(len(output)" in src:
            print("патч WaveRNN fade_out уже есть")
        else:
            print("не нашёл fade_out для патча")


def venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def main() -> None:
    MB_DIR.parent.mkdir(parents=True, exist_ok=True)
    WEIGHTS.mkdir(parents=True, exist_ok=True)

    if not (MB_DIR / ".git").exists():
        run(["git", "clone", "--depth", "1", REPO, str(MB_DIR)])
    else:
        print("клон уже есть:", MB_DIR)
    patch_mockingbird_compat(MB_DIR)

    py = find_python()
    if not venv_python().exists():
        run([py, "-m", "venv", str(VENV_DIR)])
    vpy = str(venv_python())
    run([vpy, "-m", "pip", "install", "-U", "pip", "wheel"])
    # Не ставим requirements.txt 2021 года (ломает современный pip/python).
    run(
        [
            vpy,
            "-m",
            "pip",
            "install",
            "torch",
            "torchaudio",
            "--index-url",
            "https://download.pytorch.org/whl/cu124",
        ]
    )
    run(
        [
            vpy,
            "-m",
            "pip",
            "install",
            "numpy<2",
            "scipy",
            "librosa",
            "soundfile",
            "unidecode",
            "inflect",
            "tqdm",
            "pypinyin",
            "cn2an",
            "webrtcvad-wheels",
            "pyyaml",
            "matplotlib",
        ]
    )

    download(HF_ENCODER, WEIGHTS / "encoder.pt")
    download(HF_VOCODER, WEIGHTS / "vocoder.pt")
    # Китайский synthesizer с Baidu обычно недоступен отсюда — кладём английский RTVC,
    # чтобы стек вообще завёлся. Свой mandarin.pt можно положить рядом вручную.
    mandarin = WEIGHTS / "mandarin.pt"
    if not mandarin.exists():
        download(HF_SYNTH, WEIGHTS / "synthesizer.pt")

    print("\nГотово.")
    print("  repo ", MB_DIR)
    print("  venv ", venv_python())
    print("  weights", WEIGHTS)
    print("В .env:")
    print("  TTS_ENGINE=mockingbird")
    print("Откат на XTTS: TTS_ENGINE=auto")


if __name__ == "__main__":
    main()
