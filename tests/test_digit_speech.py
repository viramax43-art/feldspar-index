"""Tests for countdown digits / ASMR speech helpers."""

from app.services.transcription import TimedSegment
from app.services.timeline_align import rebuild_video_dub_segments
from app.text.digit_speech import (
    build_asmr_digit_ssml,
    ensure_full_countdown,
    extract_digit_sequence,
    normalize_digit_token,
    split_countdown_cues,
    translate_digit_cue,
)
from app.text.preprocess import expand_numbers


def test_expand_numbers_does_not_make_decimals_from_list():
    assert "целая" not in expand_numbers("5, 4, 3, 2, 1")
    out = expand_numbers("5, 4, 3")
    assert "пять" in out and "четыре" in out and "три" in out


def test_expand_numbers_single_digit_dot_is_not_fraction_words():
    # countdown glue 5.4 → отдельные слова, не «пять целая четыре»
    out = expand_numbers("5.4")
    assert "целая" not in out
    assert "пять" in out and "четыре" in out


def test_real_decimal_still_ok_for_longer_fraction():
    out = expand_numbers("12.35")
    assert "целая" in out


def test_normalize_digit_tokens():
    assert normalize_digit_token("5") == "пять"
    assert normalize_digit_token("Five,") == "пять"
    assert normalize_digit_token("zero") == "ноль"
    assert translate_digit_cue("4") == "четыре"


def test_extract_digit_sequence():
    assert extract_digit_sequence("5, 4, 3, 2, 1, 0") == [
        "пять",
        "четыре",
        "три",
        "два",
        "один",
        "ноль",
    ]
    assert extract_digit_sequence("five four three two one zero") == [
        "пять",
        "четыре",
        "три",
        "два",
        "один",
        "ноль",
    ]


def test_split_countdown_cues_one_per_digit():
    segs = [
        TimedSegment(
            0.5,
            4.2,
            "5, 4, 3, 2, 1, 0",
            words=[
                ("5", 0.5, 0.8),
                ("4", 1.2, 1.5),
                ("3", 1.9, 2.2),
                ("2", 2.6, 2.9),
                ("1", 3.3, 3.55),
                ("0", 4.0, 4.2),
            ],
        )
    ]
    out = split_countdown_cues(segs)
    assert len(out) == 6
    assert [c.text for c in out] == ["пять", "четыре", "три", "два", "один", "ноль"]


def test_split_english_word_countdown():
    segs = [
        TimedSegment(
            0.5,
            5.0,
            "five four three two one zero",
            words=[
                ("five", 0.5, 0.85),
                ("four", 1.3, 1.55),
                ("three", 2.1, 2.4),
                ("two", 2.9, 3.15),
                ("one", 3.7, 3.95),
                ("zero", 4.5, 4.85),
            ],
        )
    ]
    out = split_countdown_cues(segs)
    assert len(out) == 6
    assert out[0].text == "пять"
    assert out[-1].text == "ноль"


def test_ensure_restores_missing_five_and_three():
    # ASR потерял 5 и 3 — как в багрепорте
    segs = [
        TimedSegment(1.2, 1.5, "четыре", words=[("four", 1.2, 1.5)], rms=0.03),
        TimedSegment(2.6, 2.9, "два", words=[("two", 2.6, 2.9)], rms=0.03),
        TimedSegment(3.3, 3.55, "один", words=[("one", 3.3, 3.55)], rms=0.03),
        TimedSegment(4.0, 4.2, "ноль", words=[("zero", 4.0, 4.2)], rms=0.03),
    ]
    out = ensure_full_countdown(segs, media_duration=15.0)
    assert [c.text for c in out] == ["пять", "четыре", "три", "два", "один", "ноль"]
    # монотонные таймкоды
    for a, b in zip(out, out[1:]):
        assert a.end <= b.start + 0.01 or a.start < b.start


def test_rebuild_splits_countdown():
    segs = [
        TimedSegment(
            0.5,
            4.2,
            "5 4 3 2 1 0",
            words=[
                ("5", 0.54, 0.85),
                ("4", 1.2, 1.45),
                ("3", 1.9, 2.15),
                ("2", 2.6, 2.85),
                ("1", 3.3, 3.5),
                ("0", 4.0, 4.15),
            ],
        )
    ]
    out = rebuild_video_dub_segments(segs, min_pause_sec=0.22, min_cue_sec=0.55)
    assert len(out) == 6
    assert [c.text for c in out] == ["пять", "четыре", "три", "два", "один", "ноль"]


def test_rebuild_english_words_countdown():
    segs = [
        TimedSegment(
            0.5,
            5.0,
            "five four three two one zero",
            words=[
                ("five", 0.5, 0.85),
                ("four", 1.3, 1.55),
                ("three", 2.1, 2.4),
                ("two", 2.9, 3.15),
                ("one", 3.7, 3.95),
                ("zero", 4.5, 4.85),
            ],
        )
    ]
    out = rebuild_video_dub_segments(segs, min_pause_sec=0.22, min_cue_sec=0.55)
    assert len(out) == 6
    assert out[2].text == "три"


def test_snap_countdown_to_energy_spreads_sparse_peaks():
    import numpy as np

    from app.text.digit_speech import snap_countdown_cues_to_energy

    sr = 16000
    dur = 12.0
    audio = np.zeros(int(dur * sr), dtype=np.float32)
    # Six whispered bursts matching a sparse countdown
    centers = [0.7, 2.4, 4.3, 6.2, 8.2, 11.0]
    for c in centers:
        a = int((c - 0.25) * sr)
        b = int((c + 0.30) * sr)
        n = b - a
        t = np.linspace(0, 1, n, endpoint=False)
        audio[a:b] = 0.05 * np.sin(2 * np.pi * 220 * t) * np.hanning(n)

    packed = [
        TimedSegment(0.4, 1.0, "пять", words=[("пять", 0.4, 1.0)], style="calm"),
        TimedSegment(1.2, 1.8, "четыре", words=[("четыре", 1.2, 1.8)], style="calm"),
        TimedSegment(2.0, 2.6, "три", words=[("три", 2.0, 2.6)], style="calm"),
        TimedSegment(2.8, 3.4, "два", words=[("два", 2.8, 3.4)], style="calm"),
        TimedSegment(3.6, 4.2, "один", words=[("один", 3.6, 4.2)], style="calm"),
        TimedSegment(4.4, 5.0, "ноль", words=[("ноль", 4.4, 5.0)], style="calm"),
    ]
    out = snap_countdown_cues_to_energy(packed, audio, sr)
    assert len(out) == 6
    assert out[-1].start > 9.5
    assert out[3].start > 5.0
    assert out[-1].end - out[0].start > 8.0


def test_expand_digit_windows_min_nol():
    from app.text.digit_speech import expand_digit_windows

    segs = [
        TimedSegment(4.0, 4.1, "ноль", words=[("zero", 4.0, 4.1)], rms=0.03),
    ]
    out = expand_digit_windows(segs, media_duration=15.0)
    assert out[0].end - out[0].start >= 0.40


def test_center_align_preroll_first():
    from app.services.video_dub import center_align_digit_placements

    segs = [
        TimedSegment(0.5, 0.95, "пять", words=[("five", 0.5, 0.95)]),
        TimedSegment(1.5, 2.0, "четыре", words=[("four", 1.5, 2.0)]),
    ]
    places = center_align_digit_placements(
        segs, [0.40, 0.45], media_duration=15.0, preroll_first_sec=0.08
    )
    # Stay near the lip window — do not jump 0.3s early.
    assert places[0][0] >= 0.40
    assert places[0][0] <= 0.55
    assert places[0][1] - places[0][0] >= 0.35


def test_asmr_digit_ssml_has_soft_prosody():
    ssml = build_asmr_digit_ssml("пять", pause_after_ms=0, rate=0.76, volume=0.64)
    assert "пять" in ssml
    assert "0.76" in ssml
    tail = build_asmr_digit_ssml("ноль", soft_tail=True, rate=0.76, volume=0.64)
    assert "break" in tail


def test_mixed_video_is_not_countdown():
    from app.text.digit_speech import looks_like_countdown

    segs = [
        TimedSegment(0.0, 1.2, "привет как дела"),
        TimedSegment(2.0, 2.4, "пять"),
        TimedSegment(3.0, 3.4, "четыре"),
        TimedSegment(4.0, 4.4, "три"),
        TimedSegment(5.0, 5.4, "два"),
        TimedSegment(6.0, 6.4, "один"),
        TimedSegment(7.0, 7.4, "ноль"),
        TimedSegment(8.0, 9.5, "давай поговорим о погоде"),
    ]
    assert looks_like_countdown(segs) is False
    out = rebuild_video_dub_segments(segs, media_duration=20.0)
    texts = " ".join(c.text for c in out)
    assert "привет" in texts
    assert "погод" in texts
    # Must NOT collapse the whole clip into 5…0 only
    assert out != ["пять", "четыре", "три", "два", "один", "ноль"]
    assert [c.text for c in out] != ["пять", "четыре", "три", "два", "один", "ноль"]


def test_pure_digit_countdown_still_detected():
    from app.text.digit_speech import looks_like_countdown

    segs = [
        TimedSegment(1.0, 1.4, "пять"),
        TimedSegment(2.0, 2.4, "четыре"),
        TimedSegment(3.0, 3.4, "три"),
        TimedSegment(4.0, 4.4, "два"),
        TimedSegment(5.0, 5.4, "один"),
        TimedSegment(6.0, 6.4, "ноль"),
    ]
    assert looks_like_countdown(segs) is True
