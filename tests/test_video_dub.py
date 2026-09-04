import numpy as np
import pytest

from app.services.transcription import TimedSegment, merge_timed_segments, split_long_timed_segments
from app.audio.separation import (
    build_language_swap_bed,
    mask_speech_regions,
    mix_dub_tracks,
    preserve_background_events,
    subtract_vocal_leak,
)
from app.services.video_dub import (
    _assert_disk_space,
    _pause_hints_from_words,
    align_speech_to_words,
    compact_repetitions_to_budget,
    fit_wav_to_duration,
    format_cue_sheet,
    format_srt_timestamp,
    format_translate_pack,
    inflate_interior_pauses,
    layout_dub_placements,
    layout_complete_speech_placements,
    anchor_placements_to_segments,
    center_align_digit_placements,
    match_clip_to_source_duration,
    match_voice_level_to_source,
    merge_translations,
    parse_translation_json,
    parse_user_translation,
    segment_time_budget,
    select_clone_segments,
    write_srt,
)
from app.text.language import leftover_source_language
from app.tts.voxcpm_engine import wrap_voxcpm_text
from app.text.reply_lang import xtts_chunk_limit


def test_xtts_ja_limit_below_tokenizer():
    assert xtts_chunk_limit("ja") <= 71
    assert xtts_chunk_limit("ru") <= 182


def test_split_long_timed_segments_by_word_gaps():
    words = [
        ("hello", 0.0, 0.4),
        ("there", 0.45, 0.9),
        ("my", 1.4, 1.6),
        ("friends", 1.65, 2.2),
        ("today", 2.8, 3.3),
        ("we", 3.35, 3.5),
        ("talk", 3.55, 4.0),
        ("about", 4.05, 4.5),
        ("code", 4.55, 5.0),
        ("again", 5.5, 6.2),
        ("and", 6.25, 6.4),
        ("more", 6.45, 7.0),
    ]
    segs = [
        TimedSegment(0.0, 7.0, " ".join(w[0] for w in words), words=words),
    ]
    out = split_long_timed_segments(segs, max_sec=3.0, min_gap=0.25)
    assert len(out) >= 3
    assert out[0].start < out[-1].start
    assert all(s.duration <= 3.6 for s in out)


def test_parse_translation_json_extracts_array():
    raw = 'конечно:\n[{"i":0,"text":"Hello"},{"i":1,"text":"World"}]\n'
    assert parse_translation_json(raw, 2) == ["Hello", "World"]
    assert parse_translation_json("nope", 1) is None


def test_parse_translation_json_markdown_and_strings():
    raw = '```json\n["Hi", "There"]\n```'
    assert parse_translation_json(raw, 2) == ["Hi", "There"]
    partial = '[{"i":0,"text":"A"},{"i":1,"text":"B"'
    assert parse_translation_json(partial, 2) == ["A", "B"]


def test_parse_translation_json_survives_extra_and_missing_quotes():
    doubled = (
        '[{"i":0,"text":"Адрес клиента"},'
        '{"i":1,"text":""Так ты видишь всё это."},'
        '{"i":2,"text":""Для меня главное"}]'
    )
    got = parse_translation_json(doubled, 3)
    assert got is not None
    assert "Адрес клиента" in got[0]
    assert "видишь" in got[1]
    assert "главное" in got[2]

    missing = (
        '[{"i":0,"text":"стоит ли снова из-за миль?"},'
        '{"i":1,"text":"принять этот заказ у Walmart?",{"i":2,"text":"этот Walmart дальше"}]'
    )
    got = parse_translation_json(missing, 3)
    assert got is not None
    assert "миль" in got[0]
    assert "Walmart" in got[1]
    assert "дальше" in got[2]


def test_format_cue_sheet_includes_timings_and_translation():
    segs = [
        TimedSegment(0.0, 1.2, "Привет?", style="question", rate=1.12, volume=1.2),
        TimedSegment(1.2, 3.0, "Да.", style="calm"),
    ]
    pages = format_cue_sheet(
        segs, translations=["Hi?", "Yes."], title="Перевод", media_duration=10.0
    )
    text = "\n".join(pages)
    assert "00.0–01.2" in text
    assert "Привет?" in text
    assert "Hi?" in text
    assert "❓" in text
    assert "×1.12" in text
    assert "120%" in text
    assert "покрытие" in text


def test_translate_pack_roundtrip():
    segs = [
        TimedSegment(0.0, 1.0, "Привет мир"),
        TimedSegment(1.0, 2.5, "Как дела?"),
        TimedSegment(2.5, 4.0, "Пока"),
    ]
    pack = format_translate_pack(segs)
    assert '<c i="1">Привет мир</c>' in pack
    assert '<c i="2">Как дела?</c>' in pack
    # DeepL returns same tags with translated bodies.
    deepl = (
        '<c i="1">Hello world</c>\n\n'
        '<c i="2">How are you?</c>\n\n'
        '<c i="3">Bye</c>'
    )
    assert parse_user_translation(deepl, 3) == [
        "Hello world",
        "How are you?",
        "Bye",
    ]
    # Legacy numbered format still works.
    pasted = "01. Hello world\n02. How are you?\n03. Bye"
    assert parse_user_translation(pasted, 3) == [
        "Hello world",
        "How are you?",
        "Bye",
    ]
    plain = "Hello world\nHow are you?\nBye"
    assert parse_user_translation(plain, 3) == [
        "Hello world",
        "How are you?",
        "Bye",
    ]
    partial = parse_user_translation("02. How are you?", 3)
    assert partial is not None
    merged = merge_translations(["", "", ""], partial)
    assert merged[1] == "How are you?"
    assert merged[0] == ""
    # Multi-message plain chunks (no numbers) fill next empty slots.
    chunk1 = parse_user_translation(
        "Hello world\nHow are you?", 3, already_filled=["", "", ""]
    )
    assert chunk1 == ["Hello world", "How are you?", ""]
    mid = merge_translations(["", "", ""], chunk1)
    chunk2 = parse_user_translation("Bye", 3, already_filled=mid)
    assert chunk2 == ["", "", "Bye"]
    assert merge_translations(mid, chunk2) == [
        "Hello world",
        "How are you?",
        "Bye",
    ]
    # Second numbered chunk merges with first without wiping.
    first = parse_user_translation("01. A\n02. B", 4)
    second = parse_user_translation("03. C\n04. D", 4, already_filled=first)
    assert merge_translations(first or [], second or []) == ["A", "B", "C", "D"]
    # Partial DeepL tags.
    part_tags = parse_user_translation(
        '<c i="3">Bye</c>', 3, already_filled=["Hello world", "How are you?", ""]
    )
    assert part_tags == ["", "", "Bye"]
    srt = (
        "1\n00:00:00,000 --> 00:00:01,000\nHello world\n\n"
        "2\n00:00:01,000 --> 00:00:02,500\nHow are you?\n\n"
        "3\n00:00:02,500 --> 00:00:04,000\nBye\n"
    )
    assert parse_user_translation(srt, 3) == [
        "Hello world",
        "How are you?",
        "Bye",
    ]


def test_parse_ignores_cue_sheet_meta_when_pasted():
    """User mistakenly pastes the timing cue sheet (or DeepL of it)."""
    raw = (
        "❗️ 00.9–02.0 (1.1с) 131% expressive\n"
        "So,\n"
        "❗️ 02.1–03.3 (1.2с) ×0.78 113% expressive\n"
        "I see you\n"
        "• 09.8–11.9 (2.1с) ×1.24 neutral\n"
        "on the floor.\n"
    )
    got = parse_user_translation(raw, 3)
    assert got == ["So,", "I see you", "on the floor."]


def test_format_translate_pack_escapes_xml():
    segs = [TimedSegment(0.0, 1.0, 'A <B> & "C"')]
    pack = format_translate_pack(segs)
    assert "&lt;B&gt;" in pack
    assert "&amp;" in pack
    assert parse_user_translation(pack, 1) == ['A <B> & "C"']


def test_merge_timed_segments_dedupes_overlap():
    segs = [
        TimedSegment(0.0, 2.0, "Привет", style="neutral"),
        TimedSegment(1.5, 3.0, "Привет", style="neutral"),
        TimedSegment(3.1, 4.0, "Мир", style="neutral"),
    ]
    merged = merge_timed_segments(segs)
    assert len(merged) == 2
    assert merged[0].end >= 2.9
    assert merged[1].text == "Мир"


def test_select_clone_segments_prefers_loud_speech():
    segs = [
        TimedSegment(0.0, 0.5, "а", style="neutral", rms=0.2),
        TimedSegment(1.0, 4.0, "Длинная фраза", style="neutral", rms=0.05),
        TimedSegment(5.0, 8.5, "Громкая фраза", style="expressive", rms=0.15),
        TimedSegment(9.0, 12.0, "Ещё", style="neutral", rms=0.08),
    ]
    picked = select_clone_segments(segs, max_sec=10.0, max_clips=2, min_clip_sec=1.2)
    assert len(picked) == 2
    assert all(p.duration >= 1.2 for p in picked)
    # Best-scored (loudest) clip first — Fish/XTTS consume only the first ref.
    assert abs(picked[0].start - 5.0) < 1e-6


def test_select_clone_segments_whisper_bonus_when_requested():
    segs = [
        TimedSegment(1.0, 4.0, "тихо", style="calm", rms=0.04),
        TimedSegment(5.0, 8.0, "громко", style="expressive", rms=0.15),
    ]
    picked = select_clone_segments(
        segs, max_sec=4.0, max_clips=1, min_clip_sec=1.2, prefer_whisper=True
    )
    assert len(picked) == 1
    assert abs(picked[0].start - 1.0) < 1e-6


def test_fit_wav_to_duration_matches_slot():
    sr = 16000
    wav = np.ones(sr, dtype=np.float32) * 0.2
    # чуть длиннее слота → мягкий speed + trim
    fitted = fit_wav_to_duration(wav, sr, 0.85, min_speed=0.92, max_speed=1.12)
    assert abs(len(fitted) / sr - 0.85) < 0.04


def test_fit_does_not_slow_down_to_fill_slot():
    sr = 16000
    wav = np.ones(sr // 2, dtype=np.float32) * 0.2  # 0.5с
    fitted = fit_wav_to_duration(
        wav, sr, 2.0, min_speed=0.92, max_speed=1.12, fill_short=False
    )
    # натуральный темп, без растягивания на 2с
    assert abs(len(fitted) / sr - 0.5) < 0.05


def test_fit_long_clip_trims_after_mild_speedup():
    sr = 16000
    wav = np.ones(sr * 4, dtype=np.float32) * 0.1  # 4с в слот 0.8с
    fitted = fit_wav_to_duration(
        wav, sr, 0.8, min_speed=0.92, max_speed=1.12, fill_short=False
    )
    assert abs(len(fitted) / sr - 0.8) < 0.05


def test_fit_allows_small_overflow_without_trim():
    sr = 16000
    wav = np.ones(int(sr * 2.5), dtype=np.float32) * 0.1
    fitted = fit_wav_to_duration(
        wav,
        sr,
        1.0,
        min_speed=1.0,
        max_speed=1.0,
        fill_short=False,
        allow_overflow_sec=2.0,
    )
    assert abs(len(fitted) / sr - 2.5) < 0.05


def test_fit_does_not_cut_speech_when_trim_tail_off():
    sr = 16000
    wav = np.ones(int(sr * 1.4), dtype=np.float32) * 0.2
    fitted = fit_wav_to_duration(
        wav,
        sr,
        1.0,
        min_speed=1.0,
        max_speed=1.0,
        fill_short=False,
        allow_overflow_sec=0.06,
        trim_tail=False,
    )
    assert len(fitted) / sr > 1.2


def test_fit_stretch_short_slows_to_original_speech():
    sr = 16000
    wav = np.ones(sr // 2, dtype=np.float32) * 0.2  # 0.5с
    fitted = fit_wav_to_duration(
        wav,
        sr,
        0.85,
        min_speed=0.6,
        max_speed=1.0,
        fill_short=False,
        stretch_short=True,
    )
    assert len(fitted) / sr > 0.7
    assert len(fitted) / sr < 0.95


def test_pause_hints_from_words():
    hints = _pause_hints_from_words([("a", 0.0, 0.2), ("b", 0.9, 1.1)])
    assert hints and abs(hints[0] - 0.7) < 1e-6
    assert _pause_hints_from_words([("a", 0.0, 0.2), ("b", 0.22, 0.4)]) == []


def test_inflate_interior_pauses_only_small_gaps():
    sr = 16000
    speech = np.ones(sr, dtype=np.float32) * 0.3
    gap = np.zeros(int(0.08 * sr), dtype=np.float32)
    wav = np.concatenate([speech, gap, speech])
    out = inflate_interior_pauses(wav, sr, 0.5)
    assert out.size / sr > wav.size / sr + 0.03
    assert out.size / sr < wav.size / sr + 0.6


def test_match_clip_fits_source_slot():
    sr = 16000
    left = np.ones(sr, dtype=np.float32) * 0.25
    gap = np.zeros(int(0.07 * sr), dtype=np.float32)
    right = np.ones(sr, dtype=np.float32) * 0.25
    wav = np.concatenate([left, gap, right])
    out = match_clip_to_source_duration(
        wav,
        sr,
        1.6,
        min_speed=0.94,
        max_speed=1.08,
        overflow_sec=0.22,
    )
    dur = out.size / float(sr)
    assert 1.45 <= dur <= 1.95


def test_align_speech_lands_on_original_words():
    sr = 16000
    left = np.ones(int(0.3 * sr), dtype=np.float32) * 0.3
    hole = np.zeros(int(0.08 * sr), dtype=np.float32)
    right = np.ones(int(0.3 * sr), dtype=np.float32) * 0.3
    wav = np.concatenate([left, hole, right])
    out = align_speech_to_words(
        wav,
        sr,
        words=[("a", 0.0, 0.4), ("b", 1.2, 1.6)],
        ref_start=0.0,
        ref_end=1.6,
        min_speed=0.85,
        max_speed=1.15,
        max_gap_sec=0.09,
    )
    second = int(1.2 * sr)
    assert float(np.mean(np.abs(out[: int(0.05 * sr)]))) > 0.1
    assert float(np.mean(np.abs(out[second : second + int(0.05 * sr)]))) > 0.1


def test_complete_speech_layout_never_shortens_or_overlaps():
    segs = [
        TimedSegment(0.5, 1.0, "first"),
        TimedSegment(1.1, 1.5, "second"),
        TimedSegment(2.0, 2.4, "third"),
    ]
    durations = [1.2, 1.0, 0.8]
    places = layout_complete_speech_placements(
        segs, durations, 4.0, gap_sec=0.1
    )
    for placed, duration in zip(places, durations):
        assert abs((placed[1] - placed[0]) - duration) < 1e-6
    assert places[0][0] == pytest.approx(0.5)
    assert places[1][0] >= places[0][1] + 0.1 - 1e-6
    assert places[2][0] >= places[1][1] + 0.1 - 1e-6


def test_complete_speech_layout_anchors_to_speech_window():
    segs = [
        TimedSegment(
            0.0,
            2.0,
            "hi there",
            words=[("hi", 0.5, 0.8), ("there", 0.9, 1.3)],
        )
    ]
    places = layout_complete_speech_placements(segs, [0.4], 3.0, gap_sec=0.1)
    assert places[0][0] == pytest.approx(0.5)
    assert places[0][1] == pytest.approx(0.9)


def test_complete_speech_layout_uses_leading_room_at_media_end():
    segs = [TimedSegment(1.0, 1.5, "complete phrase")]
    places = layout_complete_speech_placements(segs, [2.5], 3.0, gap_sec=0.1)
    assert places[0][0] == pytest.approx(0.5)
    assert places[0][1] == pytest.approx(3.0)


def test_complete_speech_layout_may_extend_media_instead_of_cutting():
    segs = [
        TimedSegment(0.0, 0.5, "first"),
        TimedSegment(0.5, 1.0, "second"),
    ]
    places = layout_complete_speech_placements(
        segs, [1.2, 1.2], 2.0, gap_sec=0.1
    )
    assert places[-1][1] > 2.0
    assert all(
        abs((end - start) - duration) < 1e-6
        for (start, end), duration in zip(places, [1.2, 1.2])
    )


def test_complete_speech_layout_pulls_back_into_leading_room():
    # Tail overflows media: cues shift left into the pauses before them.
    segs = [
        TimedSegment(0.0, 0.8, "one", words=[("one", 0.0, 0.8)]),
        TimedSegment(2.0, 2.6, "two", words=[("two", 2.0, 2.6)]),
        TimedSegment(4.0, 4.5, "three", words=[("three", 4.0, 4.5)]),
    ]
    durations = [1.2, 1.2, 1.2]
    places = layout_complete_speech_placements(
        segs, durations, 4.0, gap_sec=0.1, max_early_sec=1.5
    )
    assert places[-1][1] <= 4.0 + 1e-6
    for (start, end), duration in zip(places, durations):
        assert abs((end - start) - duration) < 1e-6
    for k in range(1, len(places)):
        assert places[k][0] >= places[k - 1][1] + 0.1 - 1e-6
    # middle cue moved into the room after the first one
    assert places[1][0] < 2.0


def test_complete_speech_layout_pullback_respects_early_cap():
    segs = [
        TimedSegment(5.0, 5.5, "only", words=[("only", 5.0, 5.5)]),
    ]
    places = layout_complete_speech_placements(
        segs, [3.0], 6.0, gap_sec=0.1, max_early_sec=1.0
    )
    # 5.0 + 3.0 = 8.0 > 6.0 → pull left, but not more than 1.0s before onset
    assert places[0][0] == pytest.approx(4.0)
    assert places[0][1] == pytest.approx(7.0)  # residual overflow stays


def test_anchor_placements_on_segment_start():
    segs = [
        TimedSegment(0.5, 2.2, "a"),
        TimedSegment(2.3, 4.0, "b"),
    ]
    places = anchor_placements_to_segments(segs, [1.7, 1.5], gap_sec=0.1)
    assert abs(places[0][0] - 0.5) < 0.02
    assert abs(places[1][0] - 2.3) < 0.02
    # не сдвигаем следующую — обрезаем предыдущую
    assert places[0][1] <= places[1][0] + 1e-3


def test_anchor_never_delays_next_cue():
    segs = [
        TimedSegment(0.0, 1.0, "a", words=[("a", 0.0, 1.0)]),
        TimedSegment(1.2, 2.0, "b", words=[("b", 1.2, 2.0)]),
    ]
    places = anchor_placements_to_segments(segs, [3.0, 0.5], gap_sec=0.1)
    assert abs(places[1][0] - 1.2) < 0.02
    assert places[0][1] <= 1.2 - 0.05


def test_center_align_does_not_cascade_delay_later_digits():
    segs = [
        TimedSegment(0.4, 1.0, "5", words=[("5", 0.4, 1.0)]),
        TimedSegment(1.5, 2.0, "4", words=[("4", 1.5, 2.0)]),
        TimedSegment(2.5, 3.0, "3", words=[("3", 2.5, 3.0)]),
        TimedSegment(3.5, 4.0, "2", words=[("2", 3.5, 4.0)]),
        TimedSegment(4.5, 5.0, "1", words=[("1", 4.5, 5.0)]),
        TimedSegment(5.5, 6.2, "0", words=[("0", 5.5, 6.2)]),
    ]
    # Long TTS on early digits must not push «один»/«ноль» seconds late.
    durs = [1.6, 0.9, 1.3, 2.0, 2.5, 2.6]
    places = center_align_digit_placements(
        segs, durs, media_duration=12.0, preroll_first_sec=0.25, min_gap_sec=0.1
    )
    # Last two should stay near original speech centers (~4.75 and ~5.85).
    assert places[4][0] < 5.3
    assert places[5][0] < 6.5
    assert places[5][0] - 5.5 < 1.2


def test_cue_sync_budget_includes_pause():
    from app.services.video_dub import cue_sync_budget

    segs = [
        TimedSegment(0.0, 1.0, "a", words=[("a", 0.5, 1.0)]),
        TimedSegment(3.0, 4.0, "b", words=[("b", 3.0, 3.5)]),
    ]
    sp0, speech_dur, pause_room, hard_cap = cue_sync_budget(segs, 0, 10.0, gap_sec=0.1)
    assert abs(sp0 - 0.5) < 1e-6
    assert abs(speech_dur - 0.5) < 1e-6
    assert hard_cap >= 2.3  # до 3.0 − gap
    assert pause_room >= 1.8


def test_match_clip_keeps_speech_with_pause_overflow():
    sr = 16000
    # 2.0s speech into 1.0s articulation + 1.2s pause room
    wav = np.ones(int(2.0 * sr), dtype=np.float32) * 0.2
    out = match_clip_to_source_duration(
        wav,
        sr,
        1.0,
        min_speed=0.95,
        max_speed=1.35,
        overflow_sec=1.2,
        trim_tail=False,
    )
    dur = out.size / float(sr)
    assert dur <= 2.25
    assert dur >= 1.0
    # не должно схлопнуться до articulation-only
    assert dur >= 1.45


def test_layout_keeps_full_clip_using_slack_and_pause():
    segs = [
        TimedSegment(0.0, 1.0, "a"),
        TimedSegment(4.0, 5.0, "b"),
    ]
    places = layout_dub_placements(segs, [2.6, 0.7], 10.0, slack_sec=2.0, gap_sec=0.15)
    assert abs((places[0][1] - places[0][0]) - 2.6) < 1e-3
    assert abs(places[0][0] - 0.0) < 0.05
    assert places[0][1] <= places[1][0] - 0.14


def test_layout_delays_next_instead_of_cutting():
    segs = [
        TimedSegment(0.0, 1.0, "a"),
        TimedSegment(1.0, 2.0, "b"),
    ]
    places = layout_dub_placements(segs, [2.4, 0.8], 10.0, slack_sec=2.0, gap_sec=0.15)
    assert (places[0][1] - places[0][0]) >= 2.3
    assert places[1][0] >= places[0][1] - 1e-3
    assert places[1][0] <= 1.0 + 2.0 + 0.45
    assert abs((places[1][1] - places[1][0]) - 0.8) < 0.05


def test_layout_does_not_cut_previous_tail():
    segs = [
        TimedSegment(0.0, 1.0, "a"),
        TimedSegment(1.05, 2.0, "b"),
    ]
    places = layout_dub_placements(segs, [1.6, 0.7], 10.0, slack_sec=2.0, gap_sec=0.1)
    assert abs((places[0][1] - places[0][0]) - 1.6) < 0.03
    assert places[1][0] >= places[0][1] - 1e-3


def test_layout_starts_early_to_keep_clip_at_media_end():
    segs = [TimedSegment(8.0, 9.0, "a")]
    places = layout_dub_placements(segs, [4.0], 10.0, slack_sec=2.0, gap_sec=0.15)
    assert places[0][1] <= 10.0 + 1e-3
    assert abs((places[0][1] - places[0][0]) - 4.0) < 0.05
    assert abs(places[0][0] - 6.0) < 0.05


def test_layout_respects_slack_and_does_not_overlap():
    segs = [
        TimedSegment(0.0, 1.0, "a"),
        TimedSegment(1.0, 2.0, "b"),
        TimedSegment(2.0, 3.0, "c"),
    ]
    places = layout_dub_placements(
        segs, [3.0, 3.0, 3.0], 6.0, slack_sec=2.0, gap_sec=0.1
    )
    for i, seg in enumerate(segs):
        t0, t1 = places[i]
        assert t0 >= seg.start - 2.0 - 1e-3
        assert t1 <= 6.0 + 1e-3
    assert (places[0][1] - places[0][0]) >= 2.9
    for i in range(len(places) - 1):
        assert places[i][1] <= places[i + 1][0] + 1e-3
    for i in range(len(places) - 1):
        assert places[i][1] <= places[i + 1][0] + 1e-3


def test_segment_time_budget_uses_gap():
    segs = [
        TimedSegment(0.0, 1.0, "a"),
        TimedSegment(1.5, 2.5, "b"),
    ]
    budget = segment_time_budget(segs, 0, 10.0)
    # до следующей фразы минус gap, можно занять паузу
    assert abs(budget - 1.38) < 0.02


def test_segment_time_budget_keeps_small_gap_when_abutting():
    segs = [
        TimedSegment(0.0, 1.25, "a"),
        TimedSegment(1.25, 2.5, "b"),
    ]
    budget = segment_time_budget(segs, 0, 10.0)
    assert abs(budget - 1.13) < 0.02


def test_long_translation_uses_gap_until_next_cue():
    segs = [
        TimedSegment(5.1, 12.8, "ingredients"),
        TimedSegment(13.4, 22.2, "step one"),
    ]
    budget = segment_time_budget(segs, 0, 165.0)
    # 13.4 − 5.1 − gap, не режем по original end 12.8
    assert budget >= 8.1
    assert budget <= 8.3


def test_fit_speeds_up_small_overflow_instead_of_cutting():
    sr = 16000
    wav = np.ones(int(sr * 2.0), dtype=np.float32) * 0.2
    fitted = fit_wav_to_duration(wav, sr, 1.85, min_speed=0.94, max_speed=1.15)
    assert abs(len(fitted) / sr - 1.85) < 0.08


def test_compact_repetitions_collapses_immediate_duplicates():
    out = compact_repetitions_to_budget(
        "да, да, да, да", budget_sec=0.3, language="ru"
    )
    assert out == "да"


def test_compact_repetitions_reduces_long_loops_by_three():
    src = "uh " * 12
    out = compact_repetitions_to_budget(src.strip(), budget_sec=0.35, language="en")
    assert out.split() == ["uh"] * 4


def test_compact_repetitions_stops_when_close_to_budget():
    src = "hello hello hello world world world"
    out = compact_repetitions_to_budget(src, budget_sec=0.45, language="en")
    # Keep meaning tokens, but shorter than source.
    assert len(out.split()) <= len(src.split())
    assert "hello" in out and "world" in out


def test_write_srt(tmp_path):
    segs = [
        TimedSegment(0.0, 1.25, "Привет", style="neutral"),
        TimedSegment(1.25, 2.5, "Мир", style="neutral"),
    ]
    path = write_srt(tmp_path / "out.srt", segs, ["Hello", "World"])
    body = path.read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:01,250" in body
    assert "Hello" in body
    assert format_srt_timestamp(3661.5) == "01:01:01,500"


def test_mask_speech_regions_attenuates_speech_slots():
    sr = 16000
    audio = np.ones(sr * 4, dtype=np.float32)
    segs = [TimedSegment(1.0, 2.0, "речь")]
    masked = mask_speech_regions(audio, sr, segs, speech_gain=0.01, fade_ms=10, pad_ms=0)
    assert masked[0] == 1.0
    assert masked[int(1.5 * sr)] < 0.05
    assert masked[-1] == 1.0


def test_mask_speech_regions_hard_mute_kills_residual():
    sr = 16000
    audio = np.ones(sr * 3, dtype=np.float32) * 0.8
    segs = [TimedSegment(0.5, 1.5, "речь")]
    masked = mask_speech_regions(audio, sr, segs, speech_gain=0.0, fade_ms=5, pad_ms=0)
    mid = int(1.0 * sr)
    assert abs(float(masked[mid])) < 1e-6
    assert float(masked[0]) == pytest.approx(0.8)


def test_silence_bed_under_voice_clears_echo():
    from app.audio.separation import silence_bed_under_voice

    sr = 16000
    bed = np.ones(sr * 2, dtype=np.float32) * 0.5
    voice = np.zeros(sr * 2, dtype=np.float32)
    voice[sr // 2 : sr // 2 + sr // 4] = 0.4
    cleaned = silence_bed_under_voice(
        bed, voice, sr, voice_thresh=0.05, pad_ms=10.0, fade_ms=5.0
    )
    mid = sr // 2 + sr // 8
    assert abs(float(cleaned[mid])) < 0.05
    assert float(cleaned[10]) == pytest.approx(0.5)


def test_language_replace_mix_keeps_bed_outside_speech():
    """Outside speech windows the accompaniment stays at full amplitude."""
    from app.audio.separation import silence_bed_under_voice

    sr = 16000
    bed = np.ones(sr * 3, dtype=np.float32) * 0.6
    voice = np.zeros(sr * 3, dtype=np.float32)
    voice[sr : 2 * sr] = 0.35
    cleared = silence_bed_under_voice(
        bed, voice, sr, voice_thresh=0.05, pad_ms=5.0, fade_ms=3.0
    )
    mixed = mix_dub_tracks(
        cleared, voice, bg_volume=1.0, voice_volume=1.0, duck_floor=0.0, sample_rate=sr
    )
    # Between phrases / before speech: bed unchanged at 0.6
    assert float(mixed[int(0.2 * sr)]) == pytest.approx(0.6, abs=0.02)
    # Under voice: bed muted, mostly voice
    assert float(mixed[int(1.5 * sr)]) == pytest.approx(0.35, abs=0.05)


def test_demucs_language_replace_keeps_bed_under_speech():
    """Full original + TTS: bed stays at 100% under speech (duck_floor=1.0)."""
    sr = 16000
    bed = np.ones(sr * 2, dtype=np.float32) * 0.55
    voice = np.zeros(sr * 2, dtype=np.float32)
    voice[sr // 4 : 3 * sr // 4] = 0.4
    mixed = mix_dub_tracks(
        bed, voice, bg_volume=1.0, voice_volume=1.0, duck_floor=1.0, sample_rate=sr
    )
    mid = int(0.5 * sr)
    assert float(mixed[mid]) == pytest.approx(0.95, abs=0.05)


def test_language_swap_bed_preserves_sfx_outside_speech():
    sr = 16000
    orig = np.zeros(sr * 3, dtype=np.float32)
    orig[:sr] = 0.5
    orig[sr : 2 * sr] = 0.35
    orig[2 * sr :] = 0.6
    # Non-speech SFX outside dialogue (laugh / heel click after speech).
    orig[int(2.2 * sr) : int(2.25 * sr)] = 0.9
    vocals = np.zeros(sr * 3, dtype=np.float32)
    vocals[sr : 2 * sr] = 0.2
    windows = [TimedSegment(1.0, 2.0, "talk")]
    bed = build_language_swap_bed(orig, vocals, sr, windows, vocal_subtract=0.6)
    # Outside speech windows the bed is the untouched original.
    assert float(bed[int(0.5 * sr)]) == pytest.approx(0.5)
    assert float(bed[int(2.5 * sr)]) == pytest.approx(0.6)
    assert float(np.max(np.abs(bed[int(2.2 * sr) : int(2.25 * sr)]))) == pytest.approx(
        0.9, abs=1e-5
    )
    mid = int(1.5 * sr)
    # Soft vocal subtract only — SFX/music energy must remain under speech.
    assert float(bed[mid]) == pytest.approx(0.35 - 0.6 * 0.2, abs=0.02)


def test_language_swap_additive_mix_keeps_original_bed():
    sr = 16000
    bed = np.ones(sr, dtype=np.float32) * 0.4
    voice = np.zeros(sr, dtype=np.float32)
    voice[sr // 4 : 3 * sr // 4] = 0.3
    mixed = bed + voice
    assert float(mixed[10]) == pytest.approx(0.4)
    assert float(mixed[sr // 2]) == pytest.approx(0.7)


def test_match_voice_level_to_source_scales_toward_original():
    sr = 16000
    voice = np.ones(sr, dtype=np.float32) * 0.15
    source = np.ones(sr, dtype=np.float32) * 0.3
    out = match_voice_level_to_source(
        voice, source, sr, windows=[(0.0, 1.0)], min_gain=0.5, max_gain=2.5
    )
    out_rms = float(np.sqrt(np.mean(np.square(out))))
    src_rms = float(np.sqrt(np.mean(np.square(source))))
    assert abs(out_rms - src_rms) < 0.05
    assert out_rms > float(np.sqrt(np.mean(np.square(voice))))


def test_mix_dub_tracks_combines_background_and_voice():
    bg = np.ones(1000, dtype=np.float32) * 0.2
    vo = np.zeros(1000, dtype=np.float32)
    vo[100:200] = 0.5
    mixed = mix_dub_tracks(
        bg, vo, bg_volume=1.0, voice_volume=1.0, duck_floor=1.0
    )
    assert abs(mixed[50] - 0.2) < 1e-5
    assert abs(mixed[150] - 0.7) < 1e-5


def test_mix_ducks_background_under_voice():
    bg = np.ones(4000, dtype=np.float32) * 0.4
    vo = np.zeros(4000, dtype=np.float32)
    vo[1000:3000] = 0.6
    mixed = mix_dub_tracks(
        bg, vo, bg_volume=1.0, voice_volume=1.0, duck_floor=0.25, sample_rate=16000
    )
    quiet = float(np.mean(mixed[100:200]))
    speech = float(np.mean(mixed[1800:1900]))
    # под речью фон присел, но голос остался громче паузы
    assert speech > quiet
    assert mixed[1800] < 0.4 + 0.6  # не полная сумма 1.0


def test_subtract_vocal_leak_reduces_ghosts():
    bg = np.ones(100, dtype=np.float32) * 0.5
    vocals = np.ones(100, dtype=np.float32) * 0.4
    cleaned = subtract_vocal_leak(bg, vocals, leak=0.5)
    assert float(np.mean(cleaned)) < float(np.mean(bg))


def test_subtract_vocal_leak_is_conservative():
    bg = np.ones(200, dtype=np.float32) * 0.5
    vocals = np.ones(200, dtype=np.float32) * 0.4
    # Even a legacy caller asking for 1.5x subtraction is clamped, avoiding
    # phase inversion and metallic residue in the accompaniment.
    cleaned = subtract_vocal_leak(bg, vocals, leak=1.5)
    assert float(np.mean(cleaned)) == pytest.approx(0.36, abs=1e-5)


def test_preserve_background_events_keeps_transient_but_not_sustained_speech():
    sr = 1000
    n = 2000
    background = np.zeros(n, dtype=np.float32)
    original = np.zeros(n, dtype=np.float32)
    vocals = np.zeros(n, dtype=np.float32)
    # Sustained dialogue in 0.2–1.2 seconds.
    vocals[200:1200] = 0.04
    original[200:1200] = 0.04
    # A sharp click/impact inside the same dialogue window.
    vocals[650:660] = 0.8
    original[650:660] = 0.8
    segments = [TimedSegment(0.2, 1.2, "speech")]
    out = preserve_background_events(
        background, vocals, original, sr, segments, event_gain=0.8
    )
    assert float(np.mean(np.abs(out[300:500]))) < 0.01
    assert float(np.max(np.abs(out[640:680]))) > 0.2


def test_preserve_background_events_keeps_non_speech_outside_asr_windows():
    sr = 1000
    n = 2000
    background = np.zeros(n, dtype=np.float32)
    original = np.zeros(n, dtype=np.float32)
    vocals = np.zeros(n, dtype=np.float32)
    # Laughter/crying after recognized speech: all of it must survive.
    vocals[1400:1700] = 0.12
    original[1400:1700] = 0.12
    segments = [TimedSegment(0.2, 1.0, "speech")]
    out = preserve_background_events(
        background, vocals, original, sr, segments, event_gain=0.8
    )
    assert float(np.mean(np.abs(out[1450:1650]))) > 0.07


def test_assert_disk_space_rejects_impossible_budget(tmp_path):
    _assert_disk_space(tmp_path, 1, label="ok")
    with pytest.raises(OSError) as exc:
        _assert_disk_space(tmp_path, 10**18, label="fail")
    assert exc.value.errno == 28


def test_leftover_source_language_skips_untranslated_line():
    assert leftover_source_language(
        "The ingredients are really simple", source="en", target="ru"
    )
    assert not leftover_source_language("Ингредиенты простые", source="en", target="ru")
    assert not leftover_source_language("Walmart", source="en", target="ru")
    assert not leftover_source_language("OK", source="en", target="ru")
    assert leftover_source_language("Это совсем просто", source="ru", target="en")
    assert not leftover_source_language("Hello", source="en", target="en")


def test_voxcpm_wrap_locks_language_and_strips_style_parens():
    out = wrap_voxcpm_text("Hello (whisper) there", "ru")
    assert out.startswith("(speak Russian)")
    assert "whisper" not in out
    assert "Hello" in out and "there" in out
    assert wrap_voxcpm_text("Привет", "ru") == "(speak Russian)Привет"
    styled = wrap_voxcpm_text("Привет", "ru", extra="slowly, questioning")
    assert styled.startswith("(speak Russian, slowly, questioning)")
    assert "Привет" in styled
