from app.services.transcription import TimedSegment
from app.services.timeline_align import rebuild_video_dub_segments
from app.text.vocalizations import (
    drop_background_vocalizations,
    is_background_vocalization,
)


def test_uh_huh_variants_are_background():
    assert is_background_vocalization("Uh-huh")
    assert is_background_vocalization("Uh -huh")
    assert is_background_vocalization("oh, oh.")
    assert is_background_vocalization("Uh -uh -uh, uh -uh, uh -uh")
    assert is_background_vocalization("mm-hmm")
    assert is_background_vocalization("угу")
    assert is_background_vocalization("ага…")


def test_remove_vocalization_tokens_mixed_cue():
    from app.text.vocalizations import remove_vocalization_tokens

    assert (
        remove_vocalization_tokens("Ах, ах, да, не останавливайся!")
        == "Да, не останавливайся!"
    )
    assert remove_vocalization_tokens("Oh! Oh! Yes, right there") == "Yes, right there"
    assert remove_vocalization_tokens("ммм… вкусно") == "вкусно"
    assert remove_vocalization_tokens("оаоаоа, боже, оаоа") == "боже"
    # digits survive
    assert remove_vocalization_tokens("пять, ах, четыре") == "пять, четыре"
    # pure moan → empty
    assert remove_vocalization_tokens("ах, ах, ох") == ""
    # real speech untouched
    assert remove_vocalization_tokens("Привет, как дела?") == "Привет, как дела?"


def test_shaking_babble_is_background():
    assert is_background_vocalization("оаоаооаоа")
    assert is_background_vocalization("оаоа оаоаоа")
    assert is_background_vocalization("ohohohoh")
    assert is_background_vocalization("а-а-а-а")
    assert is_background_vocalization("o-a-o-a-o")
    assert is_background_vocalization("ааоуаа")
    assert is_background_vocalization("aoaoaoa")
    assert is_background_vocalization("лалалала")


def test_short_vowel_words_survive():
    assert not is_background_vocalization("её голос")
    assert not is_background_vocalization("I am here")
    assert not is_background_vocalization("а он ушёл")


def test_moans_and_groans_are_background():
    assert is_background_vocalization("Ahh")
    assert is_background_vocalization("Ahhhh")
    assert is_background_vocalization("Ohhh ohhh")
    assert is_background_vocalization("Mmm mmm")
    assert is_background_vocalization("Nnngh")
    assert is_background_vocalization("Ungh ungh")
    assert is_background_vocalization("Ааа ааа")
    assert is_background_vocalization("Ммм…")
    assert is_background_vocalization("Ah ah ah oh oh")
    assert is_background_vocalization("moaning")
    assert is_background_vocalization("стон")


def test_real_speech_not_background():
    assert not is_background_vocalization("oh no")
    assert not is_background_vocalization("hello there")
    assert not is_background_vocalization("five")
    assert not is_background_vocalization("Oh my god")
    assert not is_background_vocalization("I love this")


def test_drop_keeps_real_cues():
    segs = [
        TimedSegment(0.0, 0.4, "Uh-huh"),
        TimedSegment(0.5, 1.5, "Look at this"),
        TimedSegment(1.6, 2.0, "oh, oh."),
        TimedSegment(2.1, 3.0, "Amazing"),
    ]
    out = drop_background_vocalizations(segs)
    assert [s.text for s in out] == ["Look at this", "Amazing"]


def test_rebuild_drops_fillers_before_merge():
    segs = [
        TimedSegment(0.0, 0.5, "Uh-huh"),
        TimedSegment(1.0, 2.5, "This is real speech here"),
        TimedSegment(3.0, 3.4, "oh oh"),
    ]
    out = rebuild_video_dub_segments(segs, min_cue_sec=0.4)
    texts = " ".join(s.text.lower() for s in out)
    assert "uh" not in texts or "speech" in texts
    assert all(not is_background_vocalization(s.text) for s in out)
    assert any("speech" in (s.text or "").lower() for s in out)
