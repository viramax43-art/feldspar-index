"""Voice-pick session: сохранение/загрузка/очистка пакета для переозвучки."""

from pathlib import Path

import numpy as np
import soundfile as sf

from app.services import voice_pick
from app.services.transcription import TimedSegment


def _mk_session(tmp_path: Path, user_id: int = 42):
    data_dir = tmp_path / "data"
    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    video = src_dir / "clip.mp4"
    video.write_bytes(b"\x00" * 2048)
    cue_dir = src_dir / "cues"
    cue_dir.mkdir(exist_ok=True)
    sr = 8000
    for i in range(3):
        sf.write(
            str(cue_dir / voice_pick.cue_wav_name(i)),
            np.zeros(sr // 2, dtype=np.float32),
            sr,
        )
    segments = [
        TimedSegment(0.0, 1.0, "Привет", style="neutral"),
        TimedSegment(1.2, 2.0, "Смотри сюда!", style="expressive"),
        TimedSegment(2.5, 3.5, "Как дела?", style="question"),
    ]
    session = voice_pick.save_session(
        data_dir,
        user_id,
        source_video=video,
        segments=segments,
        translated=["Hi", "Look here!", "How are you?"],
        lang="en",
        duration_sec=4.0,
        cue_audio_dir=cue_dir,
    )
    return data_dir, session


def test_save_and_load_session(tmp_path: Path):
    data_dir, session = _mk_session(tmp_path)
    assert session is not None
    loaded = voice_pick.load_session(data_dir, 42)
    assert loaded is not None
    assert loaded["lang"] == "en"
    assert len(loaded["_segments"]) == 3
    assert loaded["_segments"][1].style == "expressive"
    assert loaded["_source"].exists()
    # wav'ы скопированы в сессию (переживают сброс tmp/outputs)
    for i in range(3):
        assert (loaded["_dir"] / "cues" / voice_pick.cue_wav_name(i)).exists()


def test_expressive_cues_marked_and_reused(tmp_path: Path):
    data_dir, session = _mk_session(tmp_path)
    cues = session["cues"]
    assert cues[0]["expressive"] is False
    assert cues[1]["expressive"] is True  # style=expressive / «!»
    assert cues[2]["expressive"] is False  # question — обычная реплика
    loaded = voice_pick.load_session(data_dir, 42)
    reuse = voice_pick.expressive_reuse_paths(loaded)
    assert list(reuse) == [1]
    assert reuse[1].exists()


def test_pickable_cues_require_wav(tmp_path: Path):
    data_dir, session = _mk_session(tmp_path)
    # у реплики 2 «потеряли» wav
    (session_dir := voice_pick.session_dir(data_dir, 42))
    (session_dir / "cues" / voice_pick.cue_wav_name(2)).unlink()
    loaded = voice_pick.load_session(data_dir, 42)
    pickable = voice_pick.pickable_cues(loaded)
    assert [c["i"] for c in pickable] == [0, 1]


def test_clear_session(tmp_path: Path):
    data_dir, session = _mk_session(tmp_path)
    assert session is not None
    voice_pick.clear_session(data_dir, 42)
    assert voice_pick.load_session(data_dir, 42) is None


def test_save_session_replaces_previous(tmp_path: Path):
    data_dir, _ = _mk_session(tmp_path)
    data_dir2, session2 = _mk_session(tmp_path)  # same user — overwrite
    assert session2 is not None
    loaded = voice_pick.load_session(data_dir, 42)
    assert loaded is not None


def test_voice_pick_keyboard_pages_and_marks():
    from app.bot.keyboards import voice_pick_keyboard

    cues = [
        {"i": i, "preview": f"реплика {i}", "expressive": i == 3, "wav": f"cue_{i:03d}.wav"}
        for i in range(20)
    ]
    kb = voice_pick_keyboard(cues, page=0)
    flat = [b for row in kb.inline_keyboard for b in row]
    pick_buttons = [b for b in flat if b.callback_data.startswith("dubvoice:pick:")]
    assert len(pick_buttons) == 8  # per page
    assert any("⚡" not in b.text for b in pick_buttons)
    # page 3 (last, 4 items)
    kb3 = voice_pick_keyboard(cues, page=2, chosen=17)
    flat3 = [b for row in kb3.inline_keyboard for b in row]
    pick3 = [b for b in flat3 if b.callback_data.startswith("dubvoice:pick:")]
    assert len(pick3) == 4
    chosen_btn = [b for b in pick3 if "✅" in b.text]
    assert len(chosen_btn) == 1
    assert chosen_btn[0].callback_data.startswith("dubvoice:pick:17:")
    # keep button exists
    assert any(b.callback_data == "dubvoice:keep" for b in flat3)
    # callback data within Telegram 64-byte limit
    assert all(len(b.callback_data.encode()) <= 64 for b in flat3)


def test_extract_original_clone_ref_from_source_not_dub(tmp_path: Path):
    data_dir, _ = _mk_session(tmp_path)
    loaded = voice_pick.load_session(data_dir, 42)
    assert loaded is not None
    sr = 24000
    audio = np.ones(sr * 4, dtype=np.float32) * 0.1
    audio[:sr] = 0.9
    sf.write(str(loaded["_dir"] / "_source_mono.wav"), audio, sr)
    # First cue is 0–1s on the original track (loud), not the silent dubbed wav.
    ref = voice_pick.extract_original_clone_ref(loaded, 0, sample_rate=sr)
    assert ref is not None and ref.exists()
    clip, csr = sf.read(str(ref))
    assert int(csr) == sr
    assert float(np.mean(clip)) > 0.6
    assert ref.with_suffix(".txt").read_text(encoding="utf-8") == "Привет"
    # Later cue sits on the quiet part of the original.
    ref2 = voice_pick.extract_original_clone_ref(loaded, 2, sample_rate=sr)
    assert ref2 is not None
    clip2, _ = sf.read(str(ref2))
    assert float(np.mean(np.abs(clip2))) < 0.3
    assert voice_pick.extract_original_clone_ref(loaded, 99) is None


def test_lock_placements_do_not_cascade():
    from app.services.video_dub import clamp_clip_to_slot, lock_placements_to_speech

    segs = [
        TimedSegment(0.0, 1.0, "a", words=[("a", 0.0, 1.0)]),
        TimedSegment(5.0, 6.0, "b", words=[("b", 5.0, 6.0)]),
        TimedSegment(10.0, 12.0, "c", words=[("c", 10.0, 12.0)]),
    ]
    # Long clips that would shove cue 2/3 far right in silence-borrow layout.
    places = lock_placements_to_speech(segs, [4.0, 9.0, 5.0], 30.0)
    assert abs(places[0][0] - 0.0) < 1e-6
    assert abs(places[1][0] - 5.0) < 1e-6
    assert abs(places[2][0] - 10.0) < 1e-6
    # Clamp runaway 40s clip down to the slot budget.
    wav = np.ones(int(40 * 16000), dtype=np.float32) * 0.2
    out = clamp_clip_to_slot(
        wav, 16000, speech_dur=2.0, hard_cap=2.5, abs_cap_sec=14.0
    )
    assert out.size / 16000 < 8.0

