"""Сервис синтеза речи: XTTS primary + Silero fallback, phrase queue."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.audio import safe_user_path
from app.audio.postprocess import finalize_synthesis_output, merge_phrase_pcm, process_phrase
from app.config import Settings
from app.database import Database
from app.services.voice_profile import VoiceProfileService
from app.text.language import resolve_language
from app.text.preprocess import (
    get_inference_params,
    load_pronunciation_dict,
    prepare_text_for_tts,
)
from app.tts.cache import NORMALIZER_VERSION, AudioCache
from app.tts.engine import TTSEngine

logger = logging.getLogger(__name__)

CLONE_ENGINES = {"xtts", "mockingbird", "voxcpm2", "openrouter_fish"}


def _ref_fingerprint(path: Path) -> str:
    """Content hash of a clone ref — stem+size collide across videos (refs are
    cut to round durations at a fixed sample rate), which leaked cached audio
    from previously dubbed videos into new ones."""
    import hashlib

    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            while True:
                block = fh.read(1 << 16)
                if not block:
                    break
                h.update(block)
        return h.hexdigest()[:16]
    except OSError:
        try:
            return f"{path.stem}:{path.stat().st_size}"
        except OSError:
            return path.stem


class ConsentRequiredError(PermissionError):
    pass


class ProfileRequiredError(RuntimeError):
    pass


@dataclass
class SynthesisMetrics:
    engine: str
    fallback_used: bool = False
    cache_hits: int = 0
    cache_misses: int = 0
    time_to_first_phrase_ms: float = 0.0
    chunk_synth_ms: list[float] = field(default_factory=list)
    encode_ms: float = 0.0
    audio_duration_sec: float = 0.0
    total_ms: float = 0.0

    @property
    def rtf(self) -> float:
        if self.audio_duration_sec <= 0:
            return 0.0
        return self.total_ms / 1000.0 / self.audio_duration_sec


class SynthesisService:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        primary: TTSEngine,
        profile_service: VoiceProfileService,
        fallback: TTSEngine | None = None,
        audio_cache: AudioCache | None = None,
    ) -> None:
        self.settings = settings
        self.db = db
        self.primary = primary
        self.fallback = fallback
        self.profile_service = profile_service
        self.audio_cache = audio_cache or AudioCache(
            settings.audio_cache_dir,
            enabled=settings.enable_audio_cache,
        )
        self._conditioning_cache: dict[int, tuple[Any, Any]] = {}
        self._ext_conditioning_cache: dict[str, tuple[Any, Any]] = {}
        # Общий lock: XTTS (CUDA) и Silero (CPU) не гоняют inference параллельно
        self._inference_lock = asyncio.Lock()

    @property
    def engine(self) -> TTSEngine:
        """Обратная совместимость со старым API."""
        return self.primary

    async def _ensure_consent(self, user_id: int) -> None:
        if not await self.db.has_consent(user_id):
            raise ConsentRequiredError("Согласие на использование голоса не подтверждено")

    def _load_latents_from_disk(self, user_id: int) -> tuple[Any, Any] | None:
        path = self.profile_service.conditioning_cache_path(user_id)
        if not path.exists():
            return None
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
            return data["gpt_cond_latent"], data["speaker_embedding"]
        except Exception as exc:
            logger.warning("Не удалось загрузить conditioning.pt: %s", exc)
            return None

    def _save_latents_to_disk(self, user_id: int, conditioning: tuple[Any, Any]) -> None:
        path = self.profile_service.conditioning_cache_path(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            gpt, emb = conditioning
            torch.save(
                {
                    "gpt_cond_latent": gpt.detach().cpu() if hasattr(gpt, "detach") else gpt,
                    "speaker_embedding": emb.detach().cpu() if hasattr(emb, "detach") else emb,
                },
                path,
            )
        except Exception as exc:
            logger.warning("Не удалось сохранить conditioning.pt: %s", exc)

    async def _choose_engine(
        self,
        user_id: int,
        force_engine: str | None = None,
    ) -> tuple[TTSEngine, str, bool, list[Path]]:
        """
        Возвращает (engine, speaker_key, has_profile, speaker_paths).
        Не смешивает speakers внутри одного ответа.
        """
        mode = (force_engine or self.settings.tts_engine).lower()
        user = await self.db.get_user(user_id)
        has_profile = bool(user.has_voice_profile)
        speaker_paths: list[Path] = []
        if has_profile:
            speaker_paths = await self.profile_service.get_reference_paths(user_id)

        if mode == "silero":
            if self.fallback is not None and self.fallback.name == "silero":
                eng = self.fallback
            elif self.primary.name == "silero":
                eng = self.primary
            else:
                raise RuntimeError("Silero engine не сконфигурирован")
            speaker = getattr(eng, "speaker", self.settings.silero_speaker)
            return eng, f"silero:{speaker}", False, []

        if self.primary.name == "seamless_m4t" or mode == "seamless_m4t":
            if self.primary.name != "seamless_m4t":
                raise RuntimeError("SeamlessM4T не загружен")
            return self.primary, f"seamless:{self.settings.seamless_speaker_id}", False, []

        if mode in CLONE_ENGINES:
            if self.primary.name not in CLONE_ENGINES:
                raise RuntimeError(f"Движок {mode} не загружен (сейчас {self.primary.name})")
            if self.primary.name == "voxcpm2":
                return self.primary, f"voxcpm2:{user_id}", has_profile, speaker_paths
            # Fish cloud TTS works without a local voice profile (default voice).
            if self.primary.name == "openrouter_fish":
                return (
                    self.primary,
                    f"openrouter_fish:{user_id}",
                    bool(has_profile and speaker_paths),
                    speaker_paths if (has_profile and speaker_paths) else [],
                )
            if not has_profile or not speaker_paths:
                raise ProfileRequiredError(
                    "Голосовой профиль не создан. Используйте /addvoice и /finishvoice"
                )
            return self.primary, f"{self.primary.name}:{user_id}", True, speaker_paths

        # auto: клонирование, если primary умеет клонировать и есть профиль
        if self.primary.name == "seamless_m4t":
            return self.primary, f"seamless:{self.settings.seamless_speaker_id}", False, []
        if self.primary.name == "voxcpm2":
            return self.primary, f"voxcpm2:{user_id}", has_profile, speaker_paths
        if self.primary.name == "openrouter_fish":
            return (
                self.primary,
                f"openrouter_fish:{user_id}",
                bool(has_profile and speaker_paths),
                speaker_paths if (has_profile and speaker_paths) else [],
            )
        if has_profile and speaker_paths and self.primary.name in CLONE_ENGINES:
            return self.primary, f"{self.primary.name}:{user_id}", True, speaker_paths

        if self.fallback is not None:
            speaker = getattr(self.fallback, "speaker", self.settings.silero_speaker)
            return self.fallback, f"silero:{speaker}", False, []

        if self.primary.name == "silero":
            speaker = getattr(self.primary, "speaker", self.settings.silero_speaker)
            return self.primary, f"silero:{speaker}", False, []

        raise ProfileRequiredError(
            "Голосовой профиль не создан. Используйте /addvoice и /finishvoice "
            "(или настройте Silero fallback)"
        )

    async def _synth_one(
        self,
        engine: TTSEngine,
        text: str,
        speaker_paths: list[Path],
        params: dict[str, Any],
        conditioning: tuple[Any, Any] | None,
        speaker_key: str,
        metrics: SynthesisMetrics,
        language: str | None = None,
    ) -> tuple[np.ndarray, tuple[Any, Any] | None]:
        lang = language or self.settings.default_language
        cache_key = AudioCache.build_key(
            text=f"{lang}|{text}",
            engine=engine.name,
            speaker_or_profile=speaker_key,
            sample_rate=engine.sample_rate,
            normalizer_version=NORMALIZER_VERSION,
            extra=(
                f"t{params.get('temperature')}|"
                f"s{params.get('speed')}|"
                f"r{params.get('repetition_penalty')}"
            ),
        )
        # Never reuse PCM across videos / clone refs: Fish can hallucinate
        # leftover words from a previous clip if the cache key collides or
        # a ref fingerprint is stale. Video dub always hits the API fresh.
        skip_cache = (
            ":ext:" in speaker_key
            or engine.name == "openrouter_fish"
        )
        cached = None if skip_cache else self.audio_cache.get(cache_key)
        if cached is not None:
            metrics.cache_hits += 1
            return cached, conditioning

        metrics.cache_misses += 1
        t0 = time.perf_counter()
        async with self._inference_lock:
            if hasattr(engine, "synthesize_chunk_async"):
                wav, conditioning = await engine.synthesize_chunk_async(
                    text,
                    speaker_paths,
                    lang,
                    params,
                    conditioning,
                )
            else:
                loop = asyncio.get_running_loop()
                wav, conditioning = await loop.run_in_executor(
                    None,
                    lambda: engine.synthesize_chunk(
                        text,
                        speaker_paths,
                        lang,
                        params,
                        conditioning,
                    ),
                )
        elapsed = (time.perf_counter() - t0) * 1000
        metrics.chunk_synth_ms.append(elapsed)
        wav = np.asarray(wav, dtype=np.float32)
        if not skip_cache:
            self.audio_cache.put(cache_key, wav)
        return wav, conditioning

    async def _run_phrase_queue(
        self,
        engine: TTSEngine,
        chunks: list[Any],
        speaker_paths: list[Path],
        params: dict[str, Any],
        conditioning: tuple[Any, Any] | None,
        speaker_key: str,
        metrics: SynthesisMetrics,
        timeout_sec: float | None,
        language: str | None = None,
    ) -> tuple[list[np.ndarray], list[float], tuple[Any, Any] | None]:
        """Синтез фразы N+1 во время ожидания/финализации N (pipelined)."""
        audio_chunks: list[np.ndarray] = []
        pauses: list[float] = []
        if not chunks:
            return audio_chunks, pauses, conditioning

        async def run_with_timeout(coro: Any) -> Any:
            if timeout_sec and timeout_sec > 0 and engine.name in {
                "xtts",
                "voxcpm2",
                "openrouter_fish",
            }:
                return await asyncio.wait_for(coro, timeout=timeout_sec)
            return await coro

        t_start = time.perf_counter()
        pending = asyncio.create_task(
            run_with_timeout(
                self._synth_one(
                    engine,
                    chunks[0].text,
                    speaker_paths,
                    params,
                    conditioning,
                    speaker_key,
                    metrics,
                    language,
                )
            )
        )

        for idx, chunk in enumerate(chunks):
            wav, conditioning = await pending
            if metrics.time_to_first_phrase_ms <= 0:
                metrics.time_to_first_phrase_ms = (time.perf_counter() - t_start) * 1000

            next_idx = idx + 1
            if next_idx < len(chunks):
                # Синтез N+1 стартует, пока финализируем фразу N (fade/loudness на CPU)
                pending = asyncio.create_task(
                    run_with_timeout(
                        self._synth_one(
                            engine,
                            chunks[next_idx].text,
                            speaker_paths,
                            params,
                            conditioning,
                            speaker_key,
                            metrics,
                            language,
                        )
                    )
                )

            # Обрезаем встроенную тишину TTS, нормализуем, короткий fade.
            # Контролируемая пауза (chunk.pause_after) добавится при склейке.
            # Мягкий trim + без выходного denoise: иначе «проглатываются» согласные.
            gentle = ":ext:" in speaker_key
            phrase = process_phrase(
                np.asarray(wav, dtype=np.float32),
                engine.sample_rate,
                fade_ms=3.0,
                tempo=1.0 if gentle else float(params.get("speed", 1.0)),
                denoise_output=False,
                trim_threshold_db=-50.0 if gentle else -45.0,
                leading_padding_ms=16 if gentle else 25,
                trailing_padding_ms=36 if gentle else 55,
            )
            audio_chunks.append(phrase)
            pauses.append(chunk.pause_after)

        return audio_chunks, pauses, conditioning

    async def _produce_phrases(
        self,
        user_id: int,
        text: str,
        force_engine: str | None = None,
        language: str | None = None,
        intonation: str | None = None,
        speaker_wavs: list[Path] | None = None,
        ssml: str | None = None,
        speed_override: float | None = None,
        max_pause_sec: float | None = None,
        allow_fallback: bool = True,
        ref_language: str | None = None,
        ref_transcript: str | None = None,
        cross_lingual: bool = False,
    ) -> tuple[list[np.ndarray], list[float], TTSEngine, SynthesisMetrics, float, np.ndarray]:
        from app.text.ssml import apply_interior_breaks_plain, parse_ssml, strip_ssml

        await self._ensure_consent(user_id)
        self.profile_service.assert_user_access(user_id, user_id)

        ssml_prosody = None
        source = (ssml or text or "").strip()
        if "<" in source and ("speak" in source.lower() or "prosody" in source.lower()):
            plain, ssml_prosody = parse_ssml(source)
            text = plain or strip_ssml(text)
        else:
            text = strip_ssml(text)
        if ssml_prosody is not None and ssml_prosody.interior_breaks_ms:
            if max_pause_sec is None or max_pause_sec >= 0.25:
                text = apply_interior_breaks_plain(text, ssml_prosody.interior_breaks_ms)

        if len(text) > self.settings.max_text_length:
            raise ValueError(
                f"Текст слишком длинный ({len(text)}). Лимит: {self.settings.max_text_length}"
            )

        t_total = time.perf_counter()
        external_refs = [p for p in (speaker_wavs or []) if p.exists()]
        if external_refs and self.primary.name in CLONE_ENGINES:
            engine = self.primary
            # отдельный ключ кэша: клон из видео, не профиль пользователя
            ref_tag = "-".join(_ref_fingerprint(p) for p in external_refs[:4])
            speaker_key = f"{engine.name}:ext:{ref_tag}"
            has_profile = False
            speaker_paths = external_refs
            metrics = SynthesisMetrics(engine=engine.name)
        else:
            engine, speaker_key, has_profile, speaker_paths = await self._choose_engine(
                user_id, force_engine=force_engine
            )
            metrics = SynthesisMetrics(engine=engine.name)

        user = await self.db.get_user(user_id)
        user_settings = user.settings or {}
        intonation = intonation or user_settings.get(
            "intonation", self.settings.default_intonation
        )
        speed = float(user_settings.get("speed", self.settings.default_speed))
        # Старые «медленные» дефолты 0.83/0.79 давали atempo-рябь
        if abs(speed - 0.83) < 1e-9 or abs(speed - 0.79) < 1e-9:
            speed = float(self.settings.default_speed)
        temperature = user_settings.get(
            "temperature", self.settings.default_temperature
        )
        # Legacy 0.85 / 0.72 → более живой дефолт
        if abs(float(temperature) - 0.85) < 1e-9 or abs(float(temperature) - 0.72) < 1e-9:
            temperature = float(self.settings.default_temperature)
        if external_refs:
            speed = 1.0
            temperature = float(temperature or 0.82)
            temperature = max(0.78, min(0.90, temperature))
        if speed_override is not None:
            # Cloud Fish can take up to ~2.0; local clone engines stay near 1.0.
            if engine.name == "openrouter_fish":
                speed = max(0.75, min(2.0, float(speed_override)))
            else:
                speed = max(0.94, min(1.2, float(speed_override)))
        elif ssml_prosody is not None:
            if external_refs:
                speed = 1.0
            else:
                speed = ssml_prosody.clamped_rate(0.72, 1.12)
        params = get_inference_params(intonation, speed=speed, temperature=temperature)
        params["intonation"] = str(intonation or "neutral")
        # Fish speed can be up to 2.0; get_inference_params clamps to 1.3.
        if engine.name == "openrouter_fish" and speed_override is not None:
            params["speed"] = float(max(0.5, min(2.0, float(speed))))
        if external_refs:
            params["temperature"] = max(0.80, min(0.90, float(params.get("temperature", 0.82))))
            params["top_p"] = max(0.90, min(0.96, float(params.get("top_p", 0.92))))
            params["repetition_penalty"] = min(
                1.62, float(params.get("repetition_penalty", 1.75))
            )
            params["top_k"] = max(40, int(params.get("top_k", 50)))
            # Keep caller's speed for Fish so short cues can be sped into the slot.
            if engine.name != "openrouter_fish":
                params["speed"] = 1.0
            else:
                params["speed"] = float(max(0.5, min(2.0, float(speed))))
                params["fish_emotion"] = str(intonation or "").lower() in {
                    "calm",
                    "soft",
                    "whisper",
                    "expressive",
                    "question",
                }
        if engine.name == "openrouter_fish":
            params["cross_lingual"] = bool(cross_lingual)
            if (ref_transcript or "").strip():
                params["ref_transcript"] = (ref_transcript or "").strip()[:240]
            # Emotion tags for calm/ASMR even without clone refs.
            if "fish_emotion" not in params:
                params["fish_emotion"] = str(intonation or "").lower() in {
                    "calm",
                    "soft",
                    "whisper",
                    "expressive",
                    "question",
                }
        if engine.name == "voxcpm2":
            from app.text.ssml import voxcpm_style_bits

            hint = voxcpm_style_bits(
                ssml_prosody, intonation, text, stable=bool(external_refs)
            )
            if hint:
                params["vox_style"] = hint

        from app.text.reply_lang import xtts_chunk_limit, xtts_language_code

        user_lang = (user_settings or {}).get("reply_language")
        language = xtts_language_code(
            language or user_lang or resolve_language(text, self.settings.default_language),
            self.settings.default_language,
        )
        pronunciation = load_pronunciation_dict(self.settings.pronunciation_dict_path)
        lang_cap = xtts_chunk_limit(
            language, self.settings.phrase_max_chars or self.settings.max_chunk_chars
        )
        if engine.name == "voxcpm2":
            lang_cap = max(lang_cap, min(420, int(self.settings.phrase_max_chars or 420)))
        min_chars = self.settings.phrase_min_chars
        if language in {"ja", "ko"}:
            min_chars = min(min_chars, 18)
        chunks = prepare_text_for_tts(
            text,
            max_chunk_chars=lang_cap,
            min_chunk_chars=min_chars,
            soft_max_chunk_chars=min(
                self.settings.phrase_soft_max_chars or lang_cap, lang_cap
            ),
            pronunciation_dict=pronunciation,
            engine=engine.name,
            language=language,
        )
        if not chunks:
            raise ValueError("Пустой текст после обработки")
        if max_pause_sec is not None:
            cap = max(0.04, float(max_pause_sec))
            for chunk in chunks:
                chunk.pause_after = min(float(chunk.pause_after), cap)

        conditioning = None
        if has_profile and engine.name == "xtts":
            conditioning = self._conditioning_cache.get(user_id) or self._load_latents_from_disk(
                user_id
            )
        elif external_refs and engine.name == "xtts":
            conditioning = self._ext_conditioning_cache.get(speaker_key)

        timeout_sec = (
            self.settings.voxcpm_timeout_sec
            if engine.name == "voxcpm2"
            else (
                # Fish requests self-pace + retry 429 inside the engine lock;
                # a queued chunk can wait much longer than one HTTP call.
                max(
                    float(self.settings.xtts_timeout_sec),
                    float(getattr(self.settings, "openrouter_tts_timeout_sec", 120.0))
                    + 240.0,
                )
                if engine.name == "openrouter_fish"
                else self.settings.xtts_timeout_sec
            )
        )
        try:
            audio_chunks, pauses, conditioning = await self._run_phrase_queue(
                engine,
                chunks,
                speaker_paths,
                params,
                conditioning,
                speaker_key,
                metrics,
                timeout_sec=timeout_sec,
                language=language,
            )
        except (asyncio.TimeoutError, torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or (
                "out of memory" in str(exc).lower()
            )
            is_timeout = isinstance(exc, asyncio.TimeoutError)
            can_fallback = (
                allow_fallback
                and engine.name in {"xtts", "voxcpm2"}
                and self.fallback is not None
                and (is_oom or is_timeout or "cuda" in str(exc).lower())
            )
            if not can_fallback:
                if hasattr(engine, "clear_gpu_cache"):
                    engine.clear_gpu_cache()
                raise

            logger.warning(
                "%s ошибка (%s), переключаюсь на Silero fallback",
                engine.name.upper(),
                type(exc).__name__,
            )
            if hasattr(engine, "clear_gpu_cache"):
                engine.clear_gpu_cache()
            metrics.fallback_used = True
            engine = self.fallback
            assert engine is not None
            speaker_key = f"silero:{getattr(engine, 'speaker', self.settings.silero_speaker)}"
            metrics.engine = engine.name
            chunks = prepare_text_for_tts(
                text,
                max_chunk_chars=self.settings.phrase_max_chars or self.settings.max_chunk_chars,
                min_chunk_chars=self.settings.phrase_min_chars,
                soft_max_chunk_chars=self.settings.phrase_soft_max_chars,
                pronunciation_dict=pronunciation,
                engine="silero",
                language=language,
            )
            if max_pause_sec is not None:
                cap = max(0.04, float(max_pause_sec))
                for chunk in chunks:
                    chunk.pause_after = min(float(chunk.pause_after), cap)
            audio_chunks, pauses, conditioning = await self._run_phrase_queue(
                engine,
                chunks,
                [],
                params,
                None,
                speaker_key,
                metrics,
                timeout_sec=None,
                language=language,
            )
        except Exception:
            if hasattr(engine, "clear_gpu_cache"):
                engine.clear_gpu_cache()
            raise

        if conditioning is not None and metrics.engine == "xtts":
            if has_profile:
                self._conditioning_cache[user_id] = conditioning
                self._save_latents_to_disk(user_id, conditioning)
            elif external_refs:
                self._ext_conditioning_cache[speaker_key] = conditioning

        if ssml_prosody is not None:
            extra_pause = ssml_prosody.pause_after_ms / 1000.0
            if extra_pause >= 0.08 and pauses:
                pauses[-1] = max(float(pauses[-1]), extra_pause)
            gain = ssml_prosody.clamped_volume()
            if abs(gain - 1.0) > 0.03:
                g = np.float32(gain)
                audio_chunks = [np.asarray(c, dtype=np.float32) * g for c in audio_chunks]

        merged = merge_phrase_pcm(
            audio_chunks, pauses, sample_rate=engine.sample_rate, preprocess=False
        )
        if ssml_prosody is not None:
            peak = float(np.max(np.abs(merged)) or 1.0)
            if peak > 0.98:
                scale = np.float32(0.97 / peak)
                audio_chunks = [np.asarray(c, dtype=np.float32) * scale for c in audio_chunks]
                merged = np.asarray(merged, dtype=np.float32) * scale
        return audio_chunks, pauses, engine, metrics, t_total, merged

    def _log_tts_metrics(
        self,
        metrics: SynthesisMetrics,
        t_total: float,
        merged: np.ndarray,
        sample_rate: int,
        encode_ms: float = 0.0,
    ) -> None:
        metrics.encode_ms = encode_ms
        metrics.audio_duration_sec = len(merged) / float(sample_rate)
        metrics.total_ms = (time.perf_counter() - t_total) * 1000
        cache_stats = self.audio_cache.stats()
        logger.info(
            "TTS metrics | engine=%s fallback=%s ttfp=%.0fms chunks=%s "
            "encode=%.0fms audio=%.2fs total=%.0fms rtf=%.2f cache_hit=%s/%s",
            metrics.engine,
            metrics.fallback_used,
            metrics.time_to_first_phrase_ms,
            [round(x) for x in metrics.chunk_synth_ms],
            metrics.encode_ms,
            metrics.audio_duration_sec,
            metrics.total_ms,
            metrics.rtf,
            cache_stats["hits"],
            cache_stats["hits"] + cache_stats["misses"],
        )

    async def synthesize(
        self,
        user_id: int,
        text: str,
        save_wav: bool = True,
        force_engine: str | None = None,
        language: str | None = None,
        intonation: str | None = None,
        speaker_wavs: list[Path] | None = None,
        save_ogg: bool = True,
        ssml: str | None = None,
    ) -> tuple[Path | None, Path | None]:
        audio_chunks, pauses, engine, metrics, t_total, merged = await self._produce_phrases(
            user_id,
            text,
            force_engine=force_engine,
            language=language,
            intonation=intonation,
            speaker_wavs=speaker_wavs,
            ssml=ssml,
        )
        if not save_wav and not save_ogg:
            self._log_tts_metrics(metrics, t_total, merged, engine.sample_rate)
            return None, None

        out_dir = safe_user_path(self.settings.users_dir, user_id, "outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        wav_path = out_dir / f"synth_{stamp}.wav" if save_wav else None
        ogg_path = out_dir / f"synth_{stamp}.ogg" if save_ogg else None

        t_enc = time.perf_counter()
        finalize_synthesis_output(
            chunks=audio_chunks,
            pauses=pauses,
            output_wav=wav_path,
            output_ogg=ogg_path,
            sample_rate=engine.sample_rate,
            enable_ai_marker=self.settings.enable_ai_audio_marker,
            ai_marker_text=self.settings.ai_marker_text,
            prefer_pcm_ogg=ogg_path is not None,
            preprocess=False,
        )
        self._log_tts_metrics(
            metrics, t_total, merged, engine.sample_rate, (time.perf_counter() - t_enc) * 1000
        )
        return wav_path, ogg_path

    async def synthesize_pcm(
        self,
        user_id: int,
        text: str,
        force_engine: str | None = None,
        language: str | None = None,
        intonation: str | None = None,
        speaker_wavs: list[Path] | None = None,
        ssml: str | None = None,
        speed_override: float | None = None,
        max_pause_sec: float | None = None,
        allow_fallback: bool = True,
        ref_language: str | None = None,
        ref_transcript: str | None = None,
        cross_lingual: bool = False,
    ) -> tuple[np.ndarray, int]:
        """Синтез без WAV/OGG на диск — для дубляжа, чтобы не забить C:."""
        _, _, engine, metrics, t_total, merged = await self._produce_phrases(
            user_id,
            text,
            force_engine=force_engine,
            language=language,
            intonation=intonation,
            speaker_wavs=speaker_wavs,
            ssml=ssml,
            speed_override=speed_override,
            max_pause_sec=max_pause_sec,
            allow_fallback=allow_fallback,
            ref_language=ref_language,
            ref_transcript=ref_transcript,
            cross_lingual=cross_lingual,
        )
        self._log_tts_metrics(metrics, t_total, merged, engine.sample_rate)
        return np.asarray(merged, dtype=np.float32), int(engine.sample_rate)

    def invalidate_cache(self, user_id: int) -> None:
        self._conditioning_cache.pop(user_id, None)
        path = self.profile_service.conditioning_cache_path(user_id)
        path.unlink(missing_ok=True)

    def reset_after_dub(self, user_id: int) -> None:
        """Full TTS reset after a finished video: latents, clone conditioning, PCM cache."""
        self.invalidate_cache(user_id)
        self._ext_conditioning_cache.clear()
        try:
            self.audio_cache.clear()
        except Exception:
            logger.exception("TTS audio cache clear failed")
        for engine in (self.primary, self.fallback):
            reset = getattr(engine, "reset_session", None)
            if callable(reset):
                try:
                    reset()
                except Exception:
                    logger.debug("reset_session failed on %s", getattr(engine, "name", "?"))
        for engine in (self.primary, self.fallback):
            if engine is not None and hasattr(engine, "clear_gpu_cache"):
                try:
                    engine.clear_gpu_cache()
                except Exception:
                    logger.debug("clear_gpu_cache failed on %s", getattr(engine, "name", "?"), exc_info=True)
