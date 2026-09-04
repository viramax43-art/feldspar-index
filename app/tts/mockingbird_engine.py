"""MockingBird TTS через отдельный Python-процесс (чужой venv, наши веса XTTS не трогаем)."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from app.tts.engine import TTSEngine

logger = logging.getLogger(__name__)


class MockingBirdEngine(TTSEngine):
    def __init__(
        self,
        root: Path,
        python_exe: Path,
        encoder: Path,
        synthesizer: Path,
        vocoder: Path,
        timeout_sec: float = 180.0,
    ) -> None:
        self.root = Path(root)
        self.python_exe = Path(python_exe)
        self.encoder = Path(encoder)
        self.synthesizer = Path(synthesizer)
        self.vocoder = Path(vocoder)
        self.timeout_sec = timeout_sec
        self._proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._sample_rate = 16000
        self._tmp = Path("data/tmp/mockingbird")
        self._stderr_tail: list[str] = []

    @property
    def name(self) -> str:
        return "mockingbird"

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def _worker_script(self) -> Path:
        # копия/оригинал воркера кладём в корень клона — так импортируется models.*
        dest = self.root / "voice_caller_worker.py"
        src = Path(__file__).resolve().parents[2] / "scripts" / "mockingbird_worker.py"
        if src.exists():
            dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        return dest

    def load(self) -> None:
        missing = [
            str(p)
            for p in (self.root, self.python_exe, self.encoder, self.synthesizer, self.vocoder)
            if not Path(p).exists()
        ]
        if missing:
            raise FileNotFoundError(
                "MockingBird не собран. Запустите:\n"
                "  python scripts/setup_mockingbird.py\n"
                "Отсутствует: " + ", ".join(missing)
            )
        worker = self._worker_script().resolve()
        python_exe = self.python_exe.resolve()
        cwd = self.root.resolve()
        env = os.environ.copy()
        env["MB_ENCODER"] = str(self.encoder.resolve())
        env["MB_SYNTHESIZER"] = str(self.synthesizer.resolve())
        env["MB_VOCODER"] = str(self.vocoder.resolve())
        env["PYTHONUNBUFFERED"] = "1"
        logger.info("Старт MockingBird worker: %s", python_exe)
        self._stderr_tail = []
        self._proc = subprocess.Popen(
            [str(python_exe), str(worker)],
            cwd=str(cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            bufsize=1,
        )
        self._start_stderr_pump()
        try:
            ready = self._readline(timeout=180.0)
        except Exception as exc:
            raise RuntimeError(f"MockingBird не загрузился: {exc}\n{self._stderr_text()}") from exc
        if not ready.get("ok"):
            err = ready.get("error") or "worker не ответил"
            raise RuntimeError(f"MockingBird не загрузился: {err}\n{self._stderr_text()}")
        self._tmp.mkdir(parents=True, exist_ok=True)
        logger.info("MockingBird worker готов")

    def warmup(self) -> None:
        with self._lock:
            self._request({"cmd": "ping"})

    def synthesize_chunk(
        self,
        text: str,
        speaker_wavs: list[Path],
        language: str,
        params: dict[str, Any],
        conditioning_cache: tuple[Any, Any] | None = None,
    ) -> tuple[Any, tuple[Any, Any] | None]:
        if not speaker_wavs:
            raise ValueError("MockingBird нужен speaker wav")
        del language  # English Tacotron; язык ответа задаётся в XTTS
        out = self._tmp / f"{uuid.uuid4().hex}.wav"
        with self._lock:
            resp = self._request(
                {
                    "cmd": "synth",
                    "text": text,
                    "speaker": str(speaker_wavs[0].resolve()),
                    "speakers": [str(p.resolve()) for p in speaker_wavs[:8]],
                    "out": str(out.resolve()),
                }
            )
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error") or "MockingBird synth failed")
        wav, sr = sf.read(str(out), always_2d=False)
        self._sample_rate = int(resp.get("sr") or sr)
        out.unlink(missing_ok=True)
        return np.asarray(wav, dtype=np.float32), conditioning_cache

    def clear_gpu_cache(self) -> None:
        return

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                self._proc.stdin.flush()
        except Exception:
            pass
        try:
            self._proc.terminate()
            self._proc.wait(timeout=8)
        except Exception:
            self._proc.kill()
        self._proc = None

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            raise RuntimeError("MockingBird worker не запущен")
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        proc.stdin.flush()
        return self._readline(self.timeout_sec)

    def _readline(self, timeout: float) -> dict[str, Any]:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        line_holder: list[str] = []

        def _read() -> None:
            line = proc.stdout.readline()
            if line:
                line_holder.append(line)

        t = threading.Thread(target=_read, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            raise TimeoutError(f"MockingBird не ответил за {timeout:.0f}с")
        if not line_holder:
            code = self._proc.poll() if self._proc else None
            raise RuntimeError(
                f"MockingBird worker закрыл stdout (exit={code})"
            )
        return json.loads(line_holder[0])

    def _start_stderr_pump(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return

        def _pump() -> None:
            assert proc.stderr is not None
            for line in proc.stderr:
                line = line.rstrip()
                if not line:
                    continue
                self._stderr_tail.append(line)
                if len(self._stderr_tail) > 80:
                    del self._stderr_tail[:-80]
                if "Gen Rate" in line or line.startswith("{|"):
                    continue
                logger.info("mockingbird | %s", line)

        threading.Thread(target=_pump, daemon=True).start()

    def _stderr_text(self) -> str:
        if not getattr(self, "_stderr_tail", None):
            return ""
        return "stderr:\n" + "\n".join(self._stderr_tail[-40:])
