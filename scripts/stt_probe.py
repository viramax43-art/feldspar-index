"""Remote STT crash probe. Writes logs/stt_probe.txt."""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CT2_USE_EXPERIMENTAL_PACKED_GEMM", "0")

OUT = ROOT / "logs" / "stt_probe.txt"
OUT.parent.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    line = msg.rstrip() + "\n"
    with OUT.open("a", encoding="utf-8") as fh:
        fh.write(line)
    print(line, end="", flush=True)


def main() -> int:
    OUT.write_text("", encoding="utf-8")
    log(f"python {sys.version}")
    try:
        import faster_whisper
        import ctranslate2

        log(f"faster_whisper {faster_whisper.__version__}")
        log(f"ctranslate2 {ctranslate2.__version__}")
        log(f"ct2_cuda {getattr(ctranslate2, 'get_cuda_device_count', lambda: '?')()}")
    except Exception:
        log(traceback.format_exc())
        return 1
    try:
        import av

        log(f"av {getattr(av, '__version__', '?')}")
    except Exception as exc:
        log(f"av import: {exc}")
    try:
        from faster_whisper import WhisperModel

        log("loading tiny cpu/int8...")
        model = WhisperModel(
            "tiny",
            device="cpu",
            compute_type="int8",
            cpu_threads=1,
            num_workers=1,
        )
        log("tiny OK")
        del model
    except Exception:
        log("tiny FAIL")
        log(traceback.format_exc())
        return 2
    try:
        log("loading small cpu/int8...")
        model = WhisperModel(
            "small",
            device="cpu",
            compute_type="int8",
            cpu_threads=1,
            num_workers=1,
        )
        log("small OK")
        del model
    except Exception:
        log("small FAIL")
        log(traceback.format_exc())
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
