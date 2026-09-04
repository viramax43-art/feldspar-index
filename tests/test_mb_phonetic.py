from app.text.language import (
    detect_script_language,
    detect_transcript_language,
    resolve_language,
)
from app.text.mb_phonetic import to_mockingbird_text


def test_detect_scripts():
    assert detect_script_language("Привет, как дела?") == "ru"
    assert detect_script_language("Guten Tag, schön") == "de"
    assert detect_script_language("Bonjour, ça va?") == "fr"
    assert detect_script_language("こんにちは") == "ja"
    assert detect_script_language("안녕하세요") == "ko"
    assert detect_script_language("Hello there") is None
    assert detect_transcript_language("Hey guys, if you're new to Spark") == "en"
    assert detect_transcript_language("Привет, это тест") == "ru"



def test_resolve_default_for_latin():
    assert resolve_language("Hello", default="de") == "de"
    assert resolve_language("Привет", default="de") == "ru"


def test_russian_to_ascii():
    out = to_mockingbird_text("Привет", language="ru")
    assert out.isascii()
    assert "preevet" in out.lower() or "privet" in out.lower()
    assert not any("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in out)


def test_german_umlauts():
    out = to_mockingbird_text("schön", language="de")
    assert out.isascii()
    assert "oe" in out.lower() or "er" in out.lower() or "sh" in out.lower()


def test_french_accents():
    out = to_mockingbird_text("café", language="fr")
    assert out.isascii()
    assert "caf" in out.lower()


def test_japanese_kana():
    out = to_mockingbird_text("こんにちは", language="ja")
    assert out.isascii()
    assert "konni" in out.lower()


def test_korean_hangul():
    out = to_mockingbird_text("안녕", language="ko")
    assert out.isascii()
    assert "an" in out.lower()


def test_digits_become_words():
    out = to_mockingbird_text("у меня 2 яблока", language="ru")
    assert not any(ch.isdigit() for ch in out)


def test_only_rtvc_symbols():
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!'\"(),-.:;? ")
    samples = [
        ("Это вкусно!", "ru"),
        ("Guten Tag!", "de"),
        ("Où est la gare?", "fr"),
        ("ありがとう", "ja"),
        ("감사합니다", "ko"),
    ]
    for text, lang in samples:
        out = to_mockingbird_text(text, language=lang)
        assert out, text
        assert set(out) <= allowed, (text, out)
