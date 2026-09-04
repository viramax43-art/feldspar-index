"""In-memory jobs for the local dub studio."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.services.transcription import TimedSegment


@dataclass
class DubJob:
    id: str
    workdir: Path
    video_path: Path | None = None
    status: str = "queued"  # queued|analyzing|ready|translating|rendering|done|error
    message: str = ""
    error: str = ""
    progress_done: int = 0
    progress_total: int = 0
    progress_preview: str = ""
    language: str | None = None
    duration_sec: float = 0.0
    segments: list[TimedSegment] = field(default_factory=list)
    translated: list[str] = field(default_factory=list)
    result_video: Path | None = None
    result_srt: Path | None = None
    clone_sec: float = 0.0
    clone_clips: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    task: asyncio.Task[Any] | None = field(default=None, repr=False)

    def touch(self, status: str | None = None, message: str = "") -> None:
        if status:
            self.status = status
        if message:
            self.message = message
        self.updated_at = time.time()

    def to_public(self, *, lite: bool = False) -> dict[str, Any]:
        speech = sum(s.duration for s in self.segments)
        payload: dict[str, Any] = {
            "id": self.id,
            "status": self.status,
            "message": self.message,
            "error": self.error,
            "progress": {
                "done": self.progress_done,
                "total": self.progress_total,
                "preview": self.progress_preview,
            },
            "language": self.language,
            "duration_sec": round(self.duration_sec, 2),
            "speech_sec": round(speech, 2),
            "segment_count": len(self.segments),
            "clone_sec": round(self.clone_sec, 2),
            "clone_clips": self.clone_clips,
            "has_video": bool(
                self.status == "done"
                and self.result_video
                and self.result_video.exists()
            ),
            "has_srt": bool(
                self.status == "done"
                and self.result_srt
                and self.result_srt.exists()
            ),
        }
        if not lite:
            payload["segments"] = [
                {
                    "i": i,
                    "start": round(s.start, 2),
                    "end": round(s.end, 2),
                    "duration": round(s.duration, 2),
                    "style": s.style,
                    "rate": round(float(getattr(s, "rate", 1.0) or 1.0), 2),
                    "volume": round(float(getattr(s, "volume", 1.0) or 1.0), 2),
                    "text": s.text,
                    "translation": self.translated[i] if i < len(self.translated) else "",
                }
                for i, s in enumerate(self.segments)
            ]
        return payload


class JobStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, DubJob] = {}
        self._lock = asyncio.Lock()

    async def create(self) -> DubJob:
        job_id = uuid.uuid4().hex[:12]
        workdir = self.root / job_id
        workdir.mkdir(parents=True, exist_ok=True)
        job = DubJob(id=job_id, workdir=workdir)
        async with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> DubJob | None:
        return self._jobs.get(job_id)
