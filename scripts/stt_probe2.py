"""Reproduce Whisper AV after torch/CUDA like the live bot."""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
OUT = ROOT / "logs" / "stt_probe2.txt"
OUT.parent.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    with OUT.open("a", encoding="utf-8") as fh:
        fh.write(msg.rstrip() + "\n")
    print(msg, flush=True)


def try_whisper(label: str) -> None:
    from faster_whisper import WhisperModel

    log(f"WhisperModel start ({label})")
    WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=1, num_workers=1)
    log(f"WhisperModel OK ({label})")


def main() -> int:
    OUT.write_text("", encoding="utf-8")
    kmp = os.environ.get("KMP_DUPLICATE_LIB_OK", "")
    log(f"KMP={kmp!r} skip_kmp={os.environ.get('SKIP_KMP', '')}")
    if not os.environ.get("SKIP_KMP"):
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        log("set KMP_DUPLICATE_LIB_OK=TRUE")
    try:
        import torch

        log(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log(str(torch.cuda.get_device_name(0)))
            t = torch.zeros(1, device="cuda")
            log(f"cuda tensor ok {t.device}")
    except Exception:
        log(traceback.format_exc())
        return 1
    try:
        try_whisper("after-torch")
    except Exception:
        log(traceback.format_exc())
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
