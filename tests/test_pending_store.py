from pathlib import Path

from app.bot.pending_store import (
    PendingQuestion,
    find_recoverable_video,
    load_pending,
    load_recoverable_job,
    pending_from_dict,
    pending_to_dict,
    save_pending,
)
from app.services.transcription import TimedSegment


def _seg() -> TimedSegment:
    return TimedSegment(
        start=1.0,
        end=2.5,
        text="hello",
        style="neutral",
        words=[("hello", 1.0, 2.5)],
    )


def test_pending_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "tmp" / "vid_7_abc" / "src.mp4"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"x")
    pending = PendingQuestion(
        kind="video",
        question="hello",
        video_path=src,
        segments=[_seg()],
        duration_sec=12.0,
        await_translation=True,
        pasted=["привет"],
    )
    save_pending(tmp_path, 7, pending)
    loaded = load_pending(tmp_path, 7)
    assert loaded is not None
    assert loaded.await_translation is True
    assert loaded.video_path == src
    assert loaded.segments[0].text == "hello"
    assert loaded.segments[0].words == [("hello", 1.0, 2.5)]
    assert loaded.pasted == ["привет"]
    again = load_recoverable_job(tmp_path, 7)
    assert again is not None
    assert again.segments[0].end == 2.5


def test_dict_ignores_unknown_segment_keys() -> None:
    pending = PendingQuestion(kind="video", segments=[_seg()], video_path=Path("x.mp4"))
    raw = pending_to_dict(pending)
    raw["segments"][0]["extra"] = 1
    restored = pending_from_dict(raw)
    assert restored.segments[0].text == "hello"


def test_find_recoverable_video_picks_newest(tmp_path: Path) -> None:
    older = tmp_path / "tmp" / "vid_3_old" / "src.mp4"
    newer = tmp_path / "tmp" / "vid_3_new" / "src.mp4"
    older.parent.mkdir(parents=True)
    newer.parent.mkdir(parents=True)
    older.write_bytes(b"a")
    newer.write_bytes(b"b")
    found = find_recoverable_video(tmp_path, 3)
    assert found == newer
    assert find_recoverable_video(tmp_path, 99) is None
