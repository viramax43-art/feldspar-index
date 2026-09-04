"""SSML: снимаем просодию с оригинала и переносим на перевод."""

import numpy as np

from app.services.transcription import TimedSegment
from app.text.preprocess import prepare_text_for_tts
from app.text.ssml import (
    Prosody,
    apply_interior_breaks_plain,
    enrich_segments_ssml,
    intonation_from_prosody,
    pace_translation,
    parse_ssml,
    strip_ssml,
    to_ssml,
    transfer_ssml,
    transfer_ssml_for_slot,
    voxcpm_style_bits,
)


def test_to_ssml_parse_roundtrip():
    ssml = to_ssml("Привет, мир", Prosody(rate=1.15, volume=0.82, pause_after_ms=240))
    plain, prosody = parse_ssml(ssml)
    assert plain == "Привет, мир"
    assert abs(prosody.rate - 1.15) < 0.02
    assert abs(prosody.volume - 0.82) < 0.05
    assert prosody.pause_after_ms >= 200


def test_transfer_ssml_keeps_rate_on_translation():
    source = to_ssml("Hello there", Prosody(rate=0.88, volume=1.2, emphasis="moderate"))
    transferred = transfer_ssml(source, "Привет всем")
    plain, prosody = parse_ssml(transferred)
    assert "Привет всем" in plain
    assert "Hello" not in plain
    assert abs(prosody.rate - 0.88) < 0.02
    assert abs(prosody.volume - 1.2) < 0.08
    assert prosody.emphasis == "moderate"
    assert "<speak>" in transferred


def test_transfer_ssml_for_slot_does_not_slow_dense_text():
    source = to_ssml("easy ingredients", Prosody(rate=0.78, pause_after_ms=400))
    long_ru = (
        "Все что нужно – несколько простых ингредиентов. "
        "Мягкий черный перец, нарезанный лук, кетчуп, горчица, "
        "огурцы, говяжий фарш, американский сыр и кнопка бургера."
    )
    out = transfer_ssml_for_slot(source, long_ru, dense=True)
    plain, prosody = parse_ssml(out)
    assert "говяжий фарш" in plain
    assert abs(prosody.rate - 1.0) < 0.02
    assert not prosody.interior_breaks_ms
    assert prosody.pause_after_ms == 0


def test_strip_ssml_removes_tags():
    raw = '<speak><prosody rate="90%">стоп <break time="300ms"/> сейчас</prosody></speak>'
    assert strip_ssml(raw) == "стоп сейчас"
    chunks = prepare_text_for_tts(raw, max_chunk_chars=80)
    joined = " ".join(c.text for c in chunks)
    assert "<" not in joined
    assert "стоп" in joined.lower() or "сейчас" in joined.lower()


def test_plain_text_is_not_ssml():
    plain, prosody = parse_ssml("Просто фраза без тегов")
    assert plain == "Просто фраза без тегов"
    assert abs(prosody.rate - 1.0) < 1e-9


def test_interior_breaks_become_ellipsis():
    out = apply_interior_breaks_plain("один два, три четыре пять", [320])
    assert "…" in out


def test_multiple_interior_breaks_all_inserted():
    # Merged sentence: every real pause becomes a marker, not a new timeline.
    from app.text.ssml import Prosody, to_ssml

    text = "один два три четыре пять шесть семь восемь девять десять одиннадцать двенадцать"
    ssml = to_ssml(text, Prosody(interior_breaks_ms=[300, 450, 380]))
    assert ssml.count("<break") == 3
    plain = apply_interior_breaks_plain(text, [300, 450])
    assert plain.count("…") == 2
    # single pause still works
    one = apply_interior_breaks_plain("один два три четыре пять шесть", [300])
    assert one.count("…") == 1


def test_intonation_from_loud_ssml():
    assert intonation_from_prosody("neutral", Prosody(emphasis="strong")) == "expressive"
    assert intonation_from_prosody("calm", Prosody()) == "calm"
    assert intonation_from_prosody("question", Prosody()) == "expressive"


def test_enrich_segments_ssml_writes_markup():
    sr = 16000
    audio = np.zeros(sr * 6, dtype=np.float32)
    # тихая медленная фраза
    t0 = int(0.2 * sr)
    t1 = int(2.4 * sr)
    audio[t0:t1] = 0.04
    # громкая быстрая
    t2 = int(2.8 * sr)
    t3 = int(3.8 * sr)
    audio[t2:t3] = 0.22
    segs = [
        TimedSegment(0.2, 2.4, "Да тихо", style="calm", rms=0.04),
        TimedSegment(2.8, 3.8, "Говори скорее громче сейчас же", style="expressive", rms=0.18),
    ]
    enrich_segments_ssml(segs, audio, sr)
    assert segs[0].ssml.startswith("<speak>")
    assert "<prosody" in segs[0].ssml
    assert segs[1].rate > segs[0].rate
    assert segs[1].volume > segs[0].volume
    plain, _ = parse_ssml(segs[0].ssml)
    assert "тихо" in plain.lower() or "да" in plain.lower()


def test_pace_translation_keeps_original_word_pause():
    words = [
        ("easy", 0.0, 0.4),
        ("ingredients", 0.45, 1.1),
        ("here", 1.7, 2.1),
        ("now", 2.15, 2.5),
    ]
    runs = pace_translation(
        "простые ингредиенты вот сейчас",
        words,
        fallback_sec=2.5,
        pause_sec=0.18,
    )
    assert len(runs) == 2
    assert abs(runs[0].pause_after_sec - 0.6) < 0.05
    assert "простые" in runs[0].text
    assert "сейчас" in runs[1].text
    assert runs[-1].pause_after_sec == 0.0


def test_voxcpm_style_bits_from_original_prosody():
    slow = voxcpm_style_bits(Prosody(rate=0.86, pitch="medium"), "calm")
    assert "calm" in slow and "slowly" in slow
    q = voxcpm_style_bits(Prosody(rate=1.0, pitch="+8%"), "expressive", "Да?")
    assert "questioning" in q
    fast = voxcpm_style_bits(Prosody(rate=1.18), "expressive")
    assert "quickly" in fast


def test_voxcpm_style_bits_stable_for_dub():
    mixed = voxcpm_style_bits(
        Prosody(rate=1.18, pitch="medium"), "expressive", "Ну давай", stable=True
    )
    assert mixed == "natural"
    assert "quickly" not in mixed
    assert "expressive" not in mixed
    q = voxcpm_style_bits(
        Prosody(rate=0.86), "calm", "Что?", stable=True
    )
    assert "questioning" in q and "natural" in q
    assert "slowly" not in q
