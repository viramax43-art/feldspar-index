from app.services.timeline_align import (
    WordToken,
    build_dub_cues_from_words,
    expand_cue_windows,
    merge_micro_cues,
    rebuild_video_dub_segments,
)
from app.services.transcription import TimedSegment


def test_build_dub_cues_splits_on_long_pause():
    words = [
        WordToken("hello", 0.0, 0.4),
        WordToken("there", 0.45, 0.9),
        WordToken("my", 1.5, 1.7),
        WordToken("friend", 1.75, 2.2),
    ]
    cues = build_dub_cues_from_words(words, min_pause_sec=0.35, min_cue_sec=0.5)
    assert len(cues) == 2
    assert cues[0].text == "hello there"
    assert cues[1].text == "my friend"


def test_build_dub_cues_keeps_hesitation_inside_phrase():
    """Mid-phrase 0.4s gaps without punctuation must not restart TTS."""
    words = [
        WordToken("this", 0.0, 0.3),
        WordToken("is", 0.35, 0.5),
        WordToken("a", 0.9, 1.0),
        WordToken("burger", 1.05, 1.5),
    ]
    cues = build_dub_cues_from_words(words, min_pause_sec=0.35, min_cue_sec=0.5)
    assert len(cues) == 1
    assert cues[0].text == "this is a burger"


def test_rebuild_merges_whisper_segments():
    segs = [
        TimedSegment(
            0.0,
            1.0,
            "hello there",
            words=[("hello", 0.0, 0.4), ("there", 0.45, 0.95)],
        ),
        TimedSegment(
            1.5,
            2.2,
            "friend",
            words=[("friend", 1.5, 2.1)],
        ),
    ]
    out = rebuild_video_dub_segments(segs, min_pause_sec=0.3, max_cue_sec=6.0, media_duration=5.0)
    assert len(out) >= 1
    assert out[0].duration > 0.8
    # speech windows stay on words (no expand-into-silence)
    assert out[0].end <= 2.3
    assert all("uh" not in (s.text or "").lower() for s in out)


def test_merge_micro_cues():
    segs = [
        TimedSegment(0.0, 2.0, "long phrase here"),
        TimedSegment(2.1, 2.18, "ok"),
    ]
    out = merge_micro_cues(segs, min_cue_sec=0.55)
    assert len(out) == 1
    assert "ok" in out[0].text


def test_expand_cue_windows_keeps_speech_end():
    segs = [
        TimedSegment(0.0, 1.0, "a"),
        TimedSegment(4.0, 4.2, "b"),
    ]
    out = expand_cue_windows(segs, 10.0, gap_sec=0.1)
    assert abs(out[0].end - 1.0) < 0.05
    assert abs(out[1].end - 4.2) < 0.05


def test_speech_window_from_words():
    from app.services.timeline_align import speech_window

    seg = TimedSegment(
        0.0,
        9.0,
        "hi there",
        words=[("hi", 1.0, 1.3), ("there", 1.4, 1.9)],
    )
    a, b = speech_window(seg)
    assert abs(a - 1.0) < 1e-6
    assert abs(b - 1.9) < 1e-6


def test_merge_sentence_fragments_joins_mid_sentence():
    from app.services.timeline_align import merge_sentence_fragments

    segs = [
        TimedSegment(
            0.0, 0.6, "So,", words=[("So,", 0.0, 0.5)]
        ),
        TimedSegment(
            0.8, 1.4, "I see you", words=[("I", 0.8, 0.9), ("see", 0.95, 1.1), ("you", 1.15, 1.4)]
        ),
        TimedSegment(
            1.6, 2.6, "are really nutty.", words=[("are", 1.6, 1.8), ("really", 1.85, 2.2), ("nutty.", 2.25, 2.6)]
        ),
        TimedSegment(
            3.2, 4.0, "Next sentence.", words=[("Next", 3.2, 3.5), ("sentence.", 3.6, 4.0)]
        ),
    ]
    out = merge_sentence_fragments(segs)
    assert len(out) == 2
    assert out[0].text == "So, I see you are really nutty."
    # words kept → inter-word pauses become interior break markers downstream
    assert len(out[0].words) == 7
    assert abs(out[0].end - 2.6) < 1e-6
    assert out[1].text == "Next sentence."


def test_merge_sentence_fragments_respects_gap_and_digits():
    from app.services.timeline_align import merge_sentence_fragments

    # gap too big → no merge even without sentence end
    segs = [
        TimedSegment(0.0, 0.5, "first part,"),
        TimedSegment(5.0, 5.8, "second part."),
    ]
    out = merge_sentence_fragments(segs, max_gap_sec=2.2)
    assert len(out) == 2
    # digit-like cues (countdown) are never merged
    segs = [
        TimedSegment(0.0, 0.4, "три"),
        TimedSegment(0.6, 1.0, "два"),
    ]
    out = merge_sentence_fragments(segs)
    assert len(out) == 2
    # merged duration cap
    segs = [
        TimedSegment(0.0, 10.0, "очень длинный кусок без точки,"),
        TimedSegment(10.2, 16.0, "ещё кусок."),
    ]
    out = merge_sentence_fragments(segs, max_merged_sec=14.0)
    assert len(out) == 2

