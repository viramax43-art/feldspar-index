"""Disk backup for dub/chat pending state so inline buttons survive a bot restart."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from app.services.transcription import TimedSegment

logger = logging.getLogger(__name__)

_VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}


@dataclass
class PendingQuestion:
    kind: str = "chat"
    question: str = ""
    analysis_context: list[str] = field(default_factory=list)
    use_stream: bool = False
    video_path: Path | None = None
    segments: list[TimedSegment] = field(default_factory=list)
    duration_sec: float = 0.0
    await_translation: bool = False
    pasted: list[str] = field(default_factory=list)
    # Pre-analyze: wait for normal/quiet STT choice.
    await_loudness: bool = False
    # Quiet/ASMR: boost audio before Whisper.
    quiet_audio: bool = False


def pending_index_path(data_dir: Path, user_id: int) -> Path:
    folder = data_dir / "pending"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{user_id}.json"


def _segment_to_dict(seg: TimedSegment) -> dict[str, Any]:
    payload = asdict(seg)
    payload["words"] = [
        [str(word), float(start), float(end)] for word, start, end in (seg.words or [])
    ]
    return payload


def _segment_from_dict(raw: dict[str, Any]) -> TimedSegment:
    allowed = {item.name for item in fields(TimedSegment)}
    data = {key: value for key, value in raw.items() if key in allowed}
    words = data.get("words") or []
    parsed: list[tuple[str, float, float]] = []
    for item in words:
        try:
            word, start, end = item
        except (TypeError, ValueError):
            continue
        parsed.append((str(word), float(start), float(end)))
    data["words"] = parsed
    return TimedSegment(**data)


def pending_to_dict(pending: PendingQuestion) -> dict[str, Any]:
    return {
        "kind": pending.kind,
        "question": pending.question,
        "analysis_context": list(pending.analysis_context or []),
        "use_stream": bool(pending.use_stream),
        "video_path": str(pending.video_path) if pending.video_path else None,
        "segments": [_segment_to_dict(seg) for seg in pending.segments],
        "duration_sec": float(pending.duration_sec or 0.0),
        "await_translation": bool(pending.await_translation),
        "pasted": list(pending.pasted or []),
        "await_loudness": bool(pending.await_loudness),
        "quiet_audio": bool(pending.quiet_audio),
    }


def pending_from_dict(raw: dict[str, Any]) -> PendingQuestion:
    video_raw = raw.get("video_path")
    video_path = Path(video_raw) if video_raw else None
    segments = [_segment_from_dict(item) for item in (raw.get("segments") or []) if isinstance(item, dict)]
    return PendingQuestion(
        kind=str(raw.get("kind") or "chat"),
        question=str(raw.get("question") or ""),
        analysis_context=[str(item) for item in (raw.get("analysis_context") or [])],
        use_stream=bool(raw.get("use_stream")),
        video_path=video_path,
        segments=segments,
        duration_sec=float(raw.get("duration_sec") or 0.0),
        await_translation=bool(raw.get("await_translation")),
        pasted=[str(item) for item in (raw.get("pasted") or [])],
        await_loudness=bool(raw.get("await_loudness")),
        quiet_audio=bool(raw.get("quiet_audio")),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def save_pending(data_dir: Path, user_id: int, pending: PendingQuestion) -> None:
    payload = pending_to_dict(pending)
    _write_json(pending_index_path(data_dir, user_id), payload)
    if pending.kind == "video" and pending.video_path is not None:
        sidecar = pending.video_path.parent / "job.json"
        try:
            _write_json(sidecar, payload)
        except OSError:
            logger.warning("Не удалось записать sidecar %s", sidecar, exc_info=True)


def load_pending_file(path: Path) -> PendingQuestion | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    pending = pending_from_dict(raw)
    if pending.kind == "video":
        if pending.video_path is None or not pending.video_path.exists():
            return None
        if not pending.segments and not pending.await_loudness:
            return None
    return pending


def load_pending(data_dir: Path, user_id: int) -> PendingQuestion | None:
    return load_pending_file(pending_index_path(data_dir, user_id))


def clear_pending(data_dir: Path, user_id: int) -> None:
    path = pending_index_path(data_dir, user_id)
    path.unlink(missing_ok=True)
    path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)


def find_recoverable_video(
    data_dir: Path,
    user_id: int,
    *,
    max_age_sec: float = 72 * 3600,
) -> Path | None:
    tmp = data_dir / "tmp"
    if not tmp.is_dir():
        return None
    prefix = f"vid_{user_id}_"
    now = time.time()
    best: tuple[float, Path] | None = None
    for folder in tmp.iterdir():
        if not folder.is_dir() or not folder.name.startswith(prefix):
            continue
        videos = [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in _VIDEO_SUFFIXES]
        if not videos:
            continue
        src = max(videos, key=lambda path: path.stat().st_mtime)
        mtime = src.stat().st_mtime
        if now - mtime > max_age_sec:
            continue
        if best is None or mtime > best[0]:
            best = (mtime, src)
    return None if best is None else best[1]


def load_recoverable_job(data_dir: Path, user_id: int) -> PendingQuestion | None:
    pending = load_pending(data_dir, user_id)
    if pending is not None:
        return pending
    src = find_recoverable_video(data_dir, user_id)
    if src is None:
        return None
    return load_pending_file(src.parent / "job.json")
