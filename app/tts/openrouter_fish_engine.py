"""OpenRouter TTS: Fish Audio S2.1 (voice clone via input_references)."""

from __future__ import annotations

import base64
import io
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
import requests

from app.tts.engine import TTSEngine

logger = logging.getLogger(__name__)

OPENROUTER_SPEECH_URL = "https://openrouter.ai/api/v1/audio/speech"


class OpenRouterFishEngine(TTSEngine):
    """Cloud TTS through OpenRouter ``/api/v1/audio/speech``.

    Supports Fish Audio S2.1 Pro Free with optional zero-shot cloning from
    ``speaker_wavs`` (sent as ``input_references``).
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "fish-audio/s2.1-pro-free:free",
        sample_rate: int = 44100,
        timeout_sec: float = 120.0,
        response_format: str = "mp3",
        http_referer: str = "https://github.com/viramax43-art/feldspar-index",
        app_title: str = "feldspar-index",
        min_interval_sec: float = 3.1,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.model = model.strip()
        self._sample_rate = int(sample_rate)
        self.timeout_sec = float(timeout_sec)
        self.response_format = (response_format or "mp3").strip().lower()
        self.http_referer = http_referer
        self.app_title = app_title
        # Free tier ≈ 20 req/min — pace requests so long videos don't hit 429.
        self.min_interval_sec = max(0.0, float(min_interval_sec))
        self._last_request_ts = 0.0
        self._lock = threading.Lock()
        self._session: requests.Session | None = None

    @property
    def name(self) -> str:
        return "openrouter_fish"

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def load(self) -> None:
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY не задан")
        if self._session is None:
            self._session = requests.Session()
        logger.info(
            "OpenRouter Fish TTS готов (model=%s format=%s)",
            self.model,
            self.response_format,
        )

    def warmup(self) -> None:
        try:
            self.synthesize_chunk("Привет.", [], "ru", {}, None)
        except Exception as exc:
            logger.warning("OpenRouter Fish warmup skipped: %s", exc)

    def clear_gpu_cache(self) -> None:
        return

    def reset_session(self) -> None:
        """Drop keep-alive HTTP session so the next video starts stateless."""
        if self._session is None:
            return
        try:
            self._session.close()
        except Exception:
            pass
        self._session = None
        self._last_request_ts = 0.0

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.app_title:
            headers["X-Title"] = self.app_title
        return headers

    def _encode_reference(self, path: Path) -> dict[str, Any] | None:
        try:
            raw = Path(path).read_bytes()
        except OSError:
            logger.warning("OpenRouter Fish: cannot read ref %s", path)
            return None
        if not raw:
            return None
        # Keep refs modest for the 15 MiB decoded / 20 MiB base64 limit.
        if len(raw) > 4_000_000:
            try:
                import librosa
                import soundfile as sf

                audio, sr = librosa.load(str(path), sr=22050, mono=True, duration=12.0)
                buf = io.BytesIO()
                sf.write(buf, audio, 22050, format="WAV", subtype="PCM_16")
                raw = buf.getvalue()
            except Exception:
                logger.exception("OpenRouter Fish: ref compress failed for %s", path)
                return None
        b64 = base64.b64encode(raw).decode("ascii")
        suffix = Path(path).suffix.lower().lstrip(".") or "wav"
        mime = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "ogg": "audio/ogg",
            "flac": "audio/flac",
            "m4a": "audio/mp4",
            "webm": "audio/webm",
        }.get(suffix, "audio/wav")
        return {
            "type": "input_audio",
            "input_audio": {"data": f"data:{mime};base64,{b64}"},
        }

    def _decode_audio(self, payload: bytes) -> np.ndarray:
        if not payload:
            return np.empty(0, dtype=np.float32)
        fmt = self.response_format
        if fmt == "pcm":
            # OpenRouter PCM is typically 24 kHz mono s16le for speech models;
            # rescale to our working rate if needed.
            pcm = np.frombuffer(payload, dtype=np.int16).astype(np.float32) / 32768.0
            src_sr = 24000
            if int(src_sr) != int(self._sample_rate) and pcm.size:
                import librosa

                pcm = librosa.resample(
                    pcm, orig_sr=src_sr, target_sr=int(self._sample_rate)
                )
            return pcm.astype(np.float32)

        import soundfile as sf

        audio, file_sr = sf.read(io.BytesIO(payload), always_2d=False)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if int(file_sr) != int(self._sample_rate) and audio.size:
            import librosa

            audio = librosa.resample(
                audio, orig_sr=int(file_sr), target_sr=int(self._sample_rate)
            )
        return audio.astype(np.float32)

    def synthesize_chunk(
        self,
        text: str,
        speaker_wavs: list[Path],
        language: str,
        params: dict[str, Any],
        conditioning_cache: tuple[Any, Any] | None = None,
    ) -> tuple[np.ndarray, tuple[Any, Any] | None]:
        if self._session is None:
            self.load()
        assert self._session is not None

        spoken = (text or "").strip()
        if not spoken:
            return np.empty(0, dtype=np.float32), conditioning_cache

        from app.text.reply_lang import normalize_reply_lang

        target_lang = normalize_reply_lang(
            language or (params or {}).get("language") or "ru"
        )
        cross_lingual = bool((params or {}).get("cross_lingual"))
        # Language tags like [russian] mangled words on the free route.
        # Emotion tags are milder — use for calm/expressive only.
        allow_emotion = bool((params or {}).get("fish_emotion", True))
        short_cue = len(spoken) <= 16 and " " not in spoken.strip()
        tone = str((params or {}).get("intonation") or "").strip().lower()
        if allow_emotion and not spoken.startswith("["):
            if tone in {"calm", "soft", "whisper"}:
                spoken = f"[softly] {spoken}"
            elif tone in {"expressive", "question"} and not short_cue:
                spoken = f"[expressive] {spoken}"

        body: dict[str, Any] = {
            "model": self.model,
            "input": spoken,
            "response_format": self.response_format,
        }
        speed = (params or {}).get("speed")
        if speed is not None:
            try:
                # Fish/OpenRouter: 0.5–2.0 when supported; else ignored.
                body["speed"] = float(max(0.5, min(2.0, float(speed))))
            except (TypeError, ValueError):
                pass

        # Always clone when refs exist — voice match matters more than accent leak.
        if speaker_wavs:
            for path in speaker_wavs or []:
                encoded = self._encode_reference(Path(path))
                if encoded is not None:
                    refs: list[dict[str, Any]] = [encoded]
                    # Transcript of the REFERENCE (not the target line) helps Fish
                    # separate timbre from content — without it the model often
                    # leaks leftover words from a previous clip into the new line.
                    ref_txt = str((params or {}).get("ref_transcript") or "").strip()
                    if ref_txt:
                        refs.append({"type": "text", "text": ref_txt[:240]})
                    body["input_references"] = refs
                    break
        if cross_lingual:
            logger.info(
                "Fish cross-lingual with clone (%s -> %s)",
                (params or {}).get("ref_language") or "?",
                target_lang,
            )

        with self._lock:
            import time as _time

            last_err = ""
            response = None
            attempts = 6
            for attempt in range(attempts):
                # Proactive pacing — cheap insurance against 429 bursts.
                since = _time.monotonic() - self._last_request_ts
                if since < self.min_interval_sec:
                    _time.sleep(self.min_interval_sec - since)
                try:
                    response = self._session.post(
                        OPENROUTER_SPEECH_URL,
                        headers=self._headers(),
                        json=body,
                        timeout=self.timeout_sec,
                    )
                except requests.RequestException as exc:
                    self._last_request_ts = _time.monotonic()
                    last_err = str(exc)[:300]
                    if attempt + 1 < attempts:
                        wait_s = min(30.0, 4.0 * (attempt + 1))
                        logger.warning(
                            "OpenRouter Fish transport error (%s), retry in %.0fs (%d/%d)",
                            exc.__class__.__name__,
                            wait_s,
                            attempt + 1,
                            attempts,
                        )
                        _time.sleep(wait_s)
                        continue
                    raise RuntimeError(
                        f"OpenRouter Fish TTS transport: {last_err}"
                    ) from exc
                self._last_request_ts = _time.monotonic()
                if response.status_code < 400:
                    break
                last_err = response.text[:300]
                if response.status_code == 429 or response.status_code >= 500:
                    retry_after = response.headers.get("Retry-After")
                    try:
                        wait_s = float(retry_after) if retry_after else 0.0
                    except ValueError:
                        wait_s = 0.0
                    if wait_s <= 0:
                        wait_s = min(65.0, 6.0 * (2**attempt))
                    logger.warning(
                        "OpenRouter Fish %s rate/overload, retry in %.0fs (%d/%d)",
                        response.status_code,
                        wait_s,
                        attempt + 1,
                        attempts,
                    )
                    _time.sleep(wait_s)
                    continue
                # 4xx (quota/auth/model) — retrying won't help.
                raise RuntimeError(
                    f"OpenRouter Fish TTS HTTP {response.status_code}: {last_err[:500]}"
                )
            assert response is not None
            if response.status_code >= 400:
                detail = (response.text or last_err)[:500]
                raise RuntimeError(
                    f"OpenRouter Fish TTS HTTP {response.status_code}: {detail}"
                )
        gen_id = response.headers.get("X-Generation-Id") or ""
        logger.info(
            "OpenRouter Fish ok chars=%d gen=%s refs=%d speed=%s cross=%s lang=%s",
            len(spoken),
            gen_id[:12],
            1 if "input_references" in body else 0,
            body.get("speed"),
            int(cross_lingual),
            target_lang,
        )
        wav = self._decode_audio(response.content)
        # Cloud TTS often pads breath/silence; strip before duration fitting.
        try:
            from app.audio.postprocess import trim_silence

            trimmed = trim_silence(
                wav,
                self._sample_rate,
                frame_ms=10,
                threshold_db=-42.0,
                leading_padding_ms=8,
                trailing_padding_ms=18,
            )
            if trimmed.size >= max(8, int(0.04 * self._sample_rate)):
                wav = trimmed
        except Exception:
            pass
        return wav, conditioning_cache
