"""Сессия озвученного видео для ручного выбора «эталонной» реплики голоса.

После доставки дубляжа сохраняем: исходное видео, реплики/переводы и wav
каждой озвученной реплики. Пользователь выбирает кнопкой реплику, чей голос
понравился, — всё видео переозвучивается с клоном из ОРИГИНАЛЬНОГО окна
этой реплики (не из TTS-wav первой озвучки: Fish иначе тащит чужие слова).
Экспрессивные реплики (⚡) остаются из первой озвучки.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from app.services.transcription import TimedSegment

logger = logging.getLogger(__name__)

SESSION_FILE = "session.json"
CUES_DIR = "cues"


def session_dir(data_dir: Path, user_id: int) -> Path:
    return data_dir / "users" / str(user_id) / "last_dub"


def _segment_to_dict(seg: TimedSegment) -> dict[str, Any]:
    payload = asdict(seg)
    payload["words"] = [
        [str(word), float(start), float(end)] for word, start, end in (seg.words or [])
    ]
    return payload


def _segment_from_dict(raw: dict[str, Any]) -> TimedSegment:
    allowed = {item.name for item in fields(TimedSegment)}
    data = {key: value for key, value in raw.items() if key in allowed}
    parsed: list[tuple[str, float, float]] = []
    for item in data.get("words") or []:
        try:
            word, start, end = item
        except (TypeError, ValueError):
            continue
        parsed.append((str(word), float(start), float(end)))
    data["words"] = parsed
    return TimedSegment(**data)


def cue_wav_name(index: int) -> str:
    return f"cue_{index:03d}.wav"


def is_expressive_segment(seg: TimedSegment) -> bool:
    """Экспрессивный момент — при переозвучке эталонным голосом НЕ трогаем."""
    text = (seg.text or "").strip()
    return seg.style == "expressive" or text.endswith("!")


def save_session(
    data_dir: Path,
    user_id: int,
    *,
    source_video: Path,
    segments: list[TimedSegment],
    translated: list[str],
    lang: str,
    duration_sec: float,
    cue_audio_dir: Path | None,
) -> dict[str, Any] | None:
    """Копирует исходник + wav реплик в last_dub и пишет session.json."""
    if not source_video.exists():
        return None
    folder = session_dir(data_dir, user_id)
    clear_session(data_dir, user_id)
    folder.mkdir(parents=True, exist_ok=True)
    cues_dir = folder / CUES_DIR
    cues_dir.mkdir(parents=True, exist_ok=True)
    src_dest = folder / f"source{source_video.suffix or '.mp4'}"
    try:
        shutil.copy2(source_video, src_dest)
    except OSError:
        logger.exception("voice-pick: не скопировал исходник %s", source_video)
        shutil.rmtree(folder, ignore_errors=True)
        return None

    cues: list[dict[str, Any]] = []
    for i, seg in enumerate(segments):
        wav_name = ""
        if cue_audio_dir is not None:
            wav_src = cue_audio_dir / cue_wav_name(i)
            if wav_src.exists():
                try:
                    shutil.copy2(wav_src, cues_dir / cue_wav_name(i))
                    wav_name = cue_wav_name(i)
                except OSError:
                    logger.warning("voice-pick: wav реплики %d не скопирован", i)
        preview = (translated[i] if i < len(translated) else "") or (seg.text or "")
        cues.append(
            {
                "i": i,
                "wav": wav_name,
                "expressive": bool(is_expressive_segment(seg)),
                "start": float(seg.start),
                "preview": " ".join(str(preview).split())[:60],
            }
        )
    payload = {
        "lang": lang,
        "duration_sec": float(duration_sec),
        "source_video": src_dest.name,
        "segments": [_segment_to_dict(seg) for seg in segments],
        "translated": [str(t) for t in translated],
        "cues": cues,
    }
    tmp = folder / (SESSION_FILE + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(folder / SESSION_FILE)
    payload["_dir"] = folder
    payload["_source"] = src_dest
    logger.info(
        "voice-pick: сессия сохранена (%d реплик, %d с wav)",
        len(cues),
        sum(1 for c in cues if c["wav"]),
    )
    return payload


def load_session(data_dir: Path, user_id: int) -> dict[str, Any] | None:
    folder = session_dir(data_dir, user_id)
    path = folder / SESSION_FILE
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    src = folder / str(raw.get("source_video") or "")
    if not src.exists():
        return None
    raw["_dir"] = folder
    raw["_source"] = src
    raw["_segments"] = [
        _segment_from_dict(item)
        for item in (raw.get("segments") or [])
        if isinstance(item, dict)
    ]
    return raw


def clear_session(data_dir: Path, user_id: int) -> None:
    folder = session_dir(data_dir, user_id)
    if folder.exists():
        shutil.rmtree(folder, ignore_errors=True)


def pickable_cues(session: dict[str, Any]) -> list[dict[str, Any]]:
    """Реплики с сохранённым wav — по ним можно выбирать голос."""
    folder: Path | None = session.get("_dir")
    out: list[dict[str, Any]] = []
    for cue in session.get("cues") or []:
        wav = cue.get("wav")
        if not wav:
            continue
        if folder is not None and not (folder / CUES_DIR / str(wav)).exists():
            continue
        out.append(cue)
    return out


def expressive_reuse_paths(session: dict[str, Any]) -> dict[int, Path]:
    """Экспрессивные реплики → их wav из первой озвучки (не переозвучиваем)."""
    folder: Path = session["_dir"]
    out: dict[int, Path] = {}
    for cue in session.get("cues") or []:
        if cue.get("expressive") and cue.get("wav"):
            path = folder / CUES_DIR / str(cue["wav"])
            if path.exists():
                out[int(cue["i"])] = path
    return out


def extract_original_clone_ref(
    session: dict[str, Any],
    cue_index: int,
    *,
    max_sec: float = 6.0,
    sample_rate: int = 24000,
) -> Path | None:
    """Cut ORIGINAL speech at the chosen cue — never the dubbed TTS wav.

    Fish treats ``input_references`` as both timbre AND leftover text. Feeding
    a previous Fish clip back in is exactly how words from earlier videos
    reappear in a new line.
    """
    segs: list[TimedSegment] = list(session.get("_segments") or [])
    if cue_index < 0 or cue_index >= len(segs):
        return None
    src = session.get("_source")
    folder = session.get("_dir")
    if src is None or folder is None or not Path(src).exists():
        return None
    from app.audio import convert_to_wav
    from app.services.timeline_align import speech_window
    import numpy as np
    import soundfile as sf

    seg = segs[cue_index]
    sp0, sp1 = speech_window(seg)
    dur = min(float(max_sec), max(1.2, float(sp1) - float(sp0)))
    work = Path(folder) / "_source_mono.wav"
    out = Path(folder) / "clone_pick.wav"
    try:
        if not work.exists():
            convert_to_wav(Path(src), work, sample_rate=int(sample_rate), mono=True)
        audio, sr = sf.read(str(work), always_2d=False)
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size < int(0.4 * sr):
            return None
        a = max(0, int(float(sp0) * sr))
        b = min(audio.size, a + int(dur * sr))
        if b - a < int(0.6 * sr):
            return None
        clip = audio[a:b]
        sf.write(str(out), clip, int(sr), subtype="PCM_16")
        src_txt = (seg.text or "").strip()
        if src_txt:
            out.with_suffix(".txt").write_text(src_txt[:240], encoding="utf-8")
        logger.info(
            "voice-pick: original clone ref cue %d %.2f-%.2fs (%.2fs)",
            cue_index,
            float(sp0),
            float(sp0) + dur,
            dur,
        )
        return out
    except Exception:
        logger.exception("voice-pick: original clone ref failed cue %d", cue_index)
        return None
