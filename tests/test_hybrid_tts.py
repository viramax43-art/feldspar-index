"""Юнит-тесты гибридного TTS: chunker, stress, router, cache."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from app.services.synthesis import ProfileRequiredError, SynthesisMetrics, SynthesisService
from app.text.preprocess import (
    COMBINING_ACUTE,
    acute_to_silero_plus,
    expand_phones,
    prepare_text_for_tts,
    split_long_sentence,
)
from app.tts.cache import AudioCache
from app.tts.engine import TTSEngine


class FakeEngine(TTSEngine):
    def __init__(self, name: str, sample_rate: int = 24000, fail: bool = False) -> None:
        self._name = name
        self._sample_rate = sample_rate
        self.fail = fail
        self.calls: list[str] = []
        self.order: list[str] = []
        self.speaker = "xenia"

    @property
    def name(self) -> str:
        return self._name

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def load(self) -> None:
        return None

    def synthesize_chunk(
        self,
        text: str,
        speaker_wavs: list[Path],
        language: str,
        params: dict[str, Any],
        conditioning_cache: tuple[Any, Any] | None = None,
    ) -> tuple[np.ndarray, tuple[Any, Any] | None]:
        if self.fail:
            raise RuntimeError("cuda out of memory")
        self.calls.append(text)
        self.order.append(text)
        return np.zeros(2400, dtype=np.float32), conditioning_cache

    async def synthesize_chunk_async(
        self,
        text: str,
        speaker_wavs: list[Path],
        language: str,
        params: dict[str, Any],
        conditioning_cache: tuple[Any, Any] | None = None,
    ) -> tuple[np.ndarray, tuple[Any, Any] | None]:
        return self.synthesize_chunk(
            text, speaker_wavs, language, params, conditioning_cache
        )

    def clear_gpu_cache(self) -> None:
        return None


def test_phrase_chunker_bounds() -> None:
    text = (
        "Это первое предложение с достаточным числом слов для проверки. "
        "А это второе, тоже не короткое, чтобы чанкер разбил текст на фразы."
    )
    chunks = prepare_text_for_tts(text, max_chunk_chars=80, min_chunk_chars=20)
    assert chunks
    for c in chunks:
        assert len(c.text) <= 120


def test_split_long_merges_tiny() -> None:
    parts = split_long_sentence("Да. Нет, это уже длиннее двадцати четырёх.", 160, 24)
    assert parts
    assert all(isinstance(p, str) for p in parts)


def test_abbrev_and_time_safe() -> None:
    chunks = prepare_text_for_tts(
        "Купим хлеб, молоко и т.д. Встреча в 12:30.",
        max_chunk_chars=160,
    )
    joined = " ".join(c.text for c in chunks)
    assert "так далее" in joined
    assert "двенадцать" in joined


def test_phone_expansion() -> None:
    out = expand_phones("Звони +7 999 123-45-67 срочно")
    assert "девять" in out
    assert "999" not in out


def test_silero_stress_adapter() -> None:
    stressed = "догов" + "о" + COMBINING_ACUTE + "р"
    adapted = acute_to_silero_plus(stressed)
    assert adapted == "догов+ор"

    silero_chunks = prepare_text_for_tts("догово́р готов", engine="silero")
    # builtin dict may replace with acute form; silero adapter converts
    assert COMBINING_ACUTE not in silero_chunks[0].text or "+" in silero_chunks[0].text

    xtts_chunks = prepare_text_for_tts("догово́р готов", engine="xtts")
    assert COMBINING_ACUTE not in xtts_chunks[0].text
    assert "+" not in xtts_chunks[0].text


def test_cache_key_stable(tmp_path: Path) -> None:
    cache = AudioCache(tmp_path, enabled=True)
    k1 = AudioCache.build_key("привет", "silero", "silero:xenia", 24000)
    k2 = AudioCache.build_key("привет", "silero", "silero:xenia", 24000)
    k3 = AudioCache.build_key("привет", "xtts", "xtts:1", 24000)
    k4 = AudioCache.build_key("привет", "silero", "silero:xenia", 24000, extra="t0.75")
    assert k1 == k2
    assert k1 != k3
    assert k1 != k4
    audio = np.ones(100, dtype=np.float32)
    cache.put(k1, audio)
    got = cache.get(k1)
    assert got is not None
    assert np.allclose(got, audio)
    assert cache.stats()["hits"] >= 1


def test_ref_fingerprint_is_content_based(tmp_path: Path) -> None:
    from app.services.synthesis import _ref_fingerprint

    # Same filename + same size, different content (two videos) → different tags.
    a = tmp_path / "job_a" / "clone_ref_01.wav"
    b = tmp_path / "job_b" / "clone_ref_01.wav"
    a.parent.mkdir()
    b.parent.mkdir()
    payload_a = b"\x00" * 1000 + b"A"
    payload_b = b"\x00" * 1000 + b"B"
    a.write_bytes(payload_a)
    b.write_bytes(payload_b)
    assert a.stat().st_size == b.stat().st_size
    assert _ref_fingerprint(a) != _ref_fingerprint(b)
    # Same content → same tag (cache still works within one video).
    assert _ref_fingerprint(a) == _ref_fingerprint(a)
    c = tmp_path / "job_c" / "clone_ref_01.wav"
    c.parent.mkdir()
    c.write_bytes(payload_a)
    assert _ref_fingerprint(a) == _ref_fingerprint(c)
    # Missing file → graceful fallback, no crash.
    assert _ref_fingerprint(tmp_path / "nope.wav")


def test_fish_and_ext_refs_skip_pcm_cache(tmp_path: Path) -> None:
    settings = _base_settings(tmp_path)
    settings.enable_audio_cache = True
    db = MagicMock()
    profile = MagicMock()
    fish = FakeEngine("openrouter_fish")
    svc = SynthesisService(settings, db, fish, profile, fallback=None)
    metrics = SynthesisMetrics(engine="openrouter_fish")

    async def run() -> None:
        wav1, _ = await svc._synth_one(
            fish, "привет", [], {}, None, "openrouter_fish:1", metrics, language="ru"
        )
        wav2, _ = await svc._synth_one(
            fish, "привет", [], {}, None, "openrouter_fish:1", metrics, language="ru"
        )
        assert wav1.size and wav2.size
        assert metrics.cache_hits == 0
        assert fish.calls == ["привет", "привет"]

        silero = FakeEngine("silero")
        ext_metrics = SynthesisMetrics(engine="silero")
        await svc._synth_one(
            silero, "тот же текст", [], {}, None, "silero:ext:abc", ext_metrics, language="ru"
        )
        await svc._synth_one(
            silero, "тот же текст", [], {}, None, "silero:ext:abc", ext_metrics, language="ru"
        )
        assert ext_metrics.cache_hits == 0
        assert silero.calls == ["тот же текст", "тот же текст"]

    asyncio.run(run())


def test_cache_clear_wipes_ram_and_disk(tmp_path: Path) -> None:
    cache = AudioCache(tmp_path, enabled=True)
    key = AudioCache.build_key("x", "fish", "ref", 24000)
    cache.put(key, np.ones(8, dtype=np.float32))
    assert (tmp_path / f"{key}.npy").exists()
    assert cache.get(key) is not None
    cache.clear()
    assert cache.get(key) is None
    assert not (tmp_path / f"{key}.npy").exists()
    assert cache.stats()["ram_entries"] == 0


def _base_settings(tmp_path: Path) -> MagicMock:
    settings = MagicMock()
    settings.tts_engine = "auto"
    settings.silero_speaker = "xenia"
    settings.audio_cache_dir = tmp_path
    settings.enable_audio_cache = False
    settings.max_text_length = 2000
    settings.default_intonation = "neutral"
    settings.default_speed = 1.0
    settings.default_temperature = 0.75
    settings.default_language = "ru"
    settings.phrase_max_chars = 160
    settings.phrase_min_chars = 24
    settings.max_chunk_chars = 160
    settings.pronunciation_dict_path = tmp_path / "missing.json"
    settings.xtts_timeout_sec = 5.0
    settings.enable_ai_audio_marker = False
    settings.ai_marker_text = ""
    settings.users_dir = tmp_path / "users"
    settings.users_dir.mkdir(exist_ok=True)
    return settings


def test_router_xtts_vs_silero(tmp_path: Path) -> None:
    settings = _base_settings(tmp_path)
    db = MagicMock()
    db.has_consent = AsyncMock(return_value=True)

    user_with = MagicMock()
    user_with.has_voice_profile = True
    user_with.settings = {}
    user_without = MagicMock()
    user_without.has_voice_profile = False
    user_without.settings = {}

    profile = MagicMock()
    profile.assert_user_access = MagicMock()
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF")
    profile.get_reference_paths = AsyncMock(return_value=[ref])
    profile.conditioning_cache_path = MagicMock(return_value=tmp_path / "cond.pt")

    xtts = FakeEngine("xtts")
    silero = FakeEngine("silero")
    svc = SynthesisService(settings, db, xtts, profile, fallback=silero)

    async def run() -> None:
        db.get_user = AsyncMock(return_value=user_with)
        eng, _key, has_p, _ = await svc._choose_engine(1)
        assert eng.name == "xtts"
        assert has_p is True

        db.get_user = AsyncMock(return_value=user_without)
        eng, _key, has_p, _ = await svc._choose_engine(1)
        assert eng.name == "silero"
        assert has_p is False

    asyncio.run(run())


def test_router_xtts_only_requires_profile(tmp_path: Path) -> None:
    settings = _base_settings(tmp_path)
    settings.tts_engine = "xtts"
    db = MagicMock()
    user = MagicMock()
    user.has_voice_profile = False
    db.get_user = AsyncMock(return_value=user)
    profile = MagicMock()
    profile.get_reference_paths = AsyncMock(return_value=[])
    svc = SynthesisService(
        settings, db, FakeEngine("xtts"), profile, fallback=FakeEngine("silero")
    )

    with pytest.raises(ProfileRequiredError):
        asyncio.run(svc._choose_engine(1))


def test_phrase_queue_order(tmp_path: Path) -> None:
    settings = _base_settings(tmp_path)
    db = MagicMock()
    profile = MagicMock()
    eng = FakeEngine("silero")
    svc = SynthesisService(settings, db, eng, profile, fallback=None)
    from app.text.preprocess import TextChunk

    chunks = [
        TextChunk("первая", 0.2),
        TextChunk("вторая", 0.2),
        TextChunk("третья", 0.2),
    ]
    metrics = SynthesisMetrics(engine="silero")

    async def run() -> None:
        audio, pauses, _ = await svc._run_phrase_queue(
            eng,
            chunks,
            [],
            {},
            None,
            "silero:xenia",
            metrics,
            timeout_sec=None,
        )
        assert len(audio) == 3
        assert eng.order == ["первая", "вторая", "третья"]
        assert metrics.time_to_first_phrase_ms > 0

    asyncio.run(run())
