"""Curated HF scan: speech replacement / cross-lingual speech under ~4GB VRAM."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

# Explicit curated seeds that are known/relevant for speech replace / S2ST / cascade dubbing.
CURATED = [
    # End-to-end / unified S2ST
    ("facebook/seamless-m4t-unity-small", "S2ST on-device ~281M", "e2e"),
    ("facebook/seamless-m4t-unity-small-s2t", "ASR+S2TT on-device ~235M", "e2e"),
    ("facebook/seamless-m4t-medium", "S2ST/S2TT/T2ST ~1.2B (fp16 ~2.5–3.5GB)", "e2e"),
    ("facebook/hf-seamless-m4t-medium", "Transformers medium S2ST", "e2e"),
    ("facebook/seamless-m4t-v2-large", "S2ST large — use int8/4bit on 4GB", "e2e-quant"),
    ("facebook/hf-seamless-m4t-large", "Transformers large — quantize", "e2e-quant"),
    # Voice conversion / speech replace timbre
    ("microsoft/speecht5_vc", "SpeechT5 voice conversion", "vc"),
    ("microsoft/speecht5_hifigan", "SpeechT5 vocoder", "vc"),
    ("microsoft/speecht5_tts", "SpeechT5 TTS (cascade)", "tts"),
    ("microsoft/speecht5_asr", "SpeechT5 ASR (cascade)", "asr"),
    ("myshell-ai/OpenVoiceV2", "OpenVoice V2 tone color converter", "vc"),
    ("myshell-ai/OpenVoice", "OpenVoice V1", "vc"),
    ("lj1995/VoiceConversionWebUI", "RVC ecosystem weights hub", "vc"),
    ("lj1995/VoiceConversionWebUI", "RVC base hub", "vc"),
    # ASR for cascade dubbing (<=4GB)
    ("openai/whisper-tiny", "ASR tiny ~39M", "asr"),
    ("openai/whisper-tiny.en", "ASR tiny EN", "asr"),
    ("openai/whisper-base", "ASR base ~74M", "asr"),
    ("openai/whisper-base.en", "ASR base EN", "asr"),
    ("openai/whisper-small", "ASR small ~244M", "asr"),
    ("openai/whisper-small.en", "ASR small EN", "asr"),
    ("Systran/faster-whisper-tiny", "CTranslate2 whisper tiny", "asr"),
    ("Systran/faster-whisper-tiny.en", "CTranslate2 whisper tiny.en", "asr"),
    ("Systran/faster-whisper-base", "CTranslate2 whisper base", "asr"),
    ("Systran/faster-whisper-base.en", "CTranslate2 whisper base.en", "asr"),
    ("Systran/faster-whisper-small", "CTranslate2 whisper small", "asr"),
    ("Systran/faster-whisper-small.en", "CTranslate2 whisper small.en", "asr"),
    ("rhasspy/faster-whisper-tiny-int8", "Whisper tiny INT8", "asr"),
    ("rhasspy/faster-whisper-base-int8", "Whisper base INT8", "asr"),
    ("rhasspy/faster-whisper-small-int8", "Whisper small INT8", "asr"),
    ("Xenova/whisper-tiny", "ONNX whisper tiny", "asr"),
    ("Xenova/whisper-base", "ONNX whisper base", "asr"),
    ("Xenova/whisper-small", "ONNX whisper small", "asr"),
    ("onnx-community/whisper-tiny", "ONNX whisper tiny", "asr"),
    ("onnx-community/whisper-base", "ONNX whisper base", "asr"),
    ("onnx-community/whisper-small", "ONNX whisper small", "asr"),
    ("facebook/wav2vec2-base-960h", "ASR wav2vec2 base", "asr"),
    ("facebook/wav2vec2-large-xlsr-53", "ASR multilingual XLSR — borderline", "asr"),
    ("facebook/mms-1b-all", "MMS ASR 1B — borderline/fp16", "asr"),
    # MT for cascade
    ("facebook/nllb-200-distilled-600M", "MT 200 langs distilled", "mt"),
    ("facebook/nllb-200-distilled-1.3B", "MT 200 langs 1.3B", "mt"),
    ("Helsinki-NLP/opus-mt-en-ru", "Marian EN→RU", "mt"),
    ("Helsinki-NLP/opus-mt-ru-en", "Marian RU→EN", "mt"),
    ("Helsinki-NLP/opus-mt-en-de", "Marian EN→DE", "mt"),
    ("Helsinki-NLP/opus-mt-en-fr", "Marian EN→FR", "mt"),
    ("Helsinki-NLP/opus-mt-en-es", "Marian EN→ES", "mt"),
    ("Helsinki-NLP/opus-mt-en-zh", "Marian EN→ZH", "mt"),
    ("Helsinki-NLP/opus-mt-mul-en", "Marian multilingual→EN", "mt"),
    ("facebook/m2m100_418M", "M2M100 418M", "mt"),
    # TTS / clone for cascade dubbing
    ("facebook/mms-tts-eng", "MMS TTS English", "tts"),
    ("facebook/mms-tts-rus", "MMS TTS Russian", "tts"),
    ("facebook/mms-tts-deu", "MMS TTS German", "tts"),
    ("facebook/mms-tts-fra", "MMS TTS French", "tts"),
    ("facebook/mms-tts-spa", "MMS TTS Spanish", "tts"),
    ("facebook/mms-tts-hin", "MMS TTS Hindi", "tts"),
    ("facebook/mms-tts-ara", "MMS TTS Arabic", "tts"),
    ("facebook/mms-tts-jpn", "MMS TTS Japanese", "tts"),
    ("facebook/mms-tts-kor", "MMS TTS Korean", "tts"),
    ("facebook/mms-tts-por", "MMS TTS Portuguese", "tts"),
    ("facebook/mms-tts-ita", "MMS TTS Italian", "tts"),
    ("facebook/mms-tts-tur", "MMS TTS Turkish", "tts"),
    ("facebook/mms-tts-pol", "MMS TTS Polish", "tts"),
    ("facebook/mms-tts-ukr", "MMS TTS Ukrainian", "tts"),
    ("suno/bark-small", "Bark small TTS", "tts"),
    ("coqui/XTTS-v2", "XTTS-v2 clone (fits GTX 1650 4GB)", "tts"),
    ("espnet/kan-bayashi_ljspeech_vits", "VITS LJSpeech", "tts"),
    ("facebook/fastspeech2-en-ljspeech", "FastSpeech2", "tts"),
    ("microsoft/speecht5_tts", "SpeechT5 TTS", "tts"),
    ("ylacombe/mms-tts-rus", "MMS TTS RU community", "tts"),
    # Fairseq S2T / older speech translation
    ("facebook/s2t-small-librispeech-asr", "Fairseq S2T small ASR", "asr"),
    ("facebook/s2t-medium-librispeech-asr", "Fairseq S2T medium ASR", "asr"),
    ("facebook/s2t-small-mustc-en-de-st", "Speech translation EN→DE", "s2t"),
    ("facebook/s2t-medium-mustc-en-fr-st", "Speech translation EN→FR", "s2t"),
    ("facebook/s2t-medium-mustc-en-es-st", "Speech translation EN→ES", "s2t"),
    ("facebook/s2t-small-covost2-en-fa-st", "Speech translation EN→FA", "s2t"),
    ("facebook/s2t-wav2vec2-large-en-de", "S2T wav2vec EN→DE", "s2t"),
    # Extra VC / clone helpers
    ("speechbrain/spkrec-ecapa-voxceleb", "Speaker embedding for VC", "vc"),
    ("facebook/hubert-base-ls960", "HuBERT for RVC/content encoder", "vc"),
    ("lengyue233/content-vec-best", "ContentVec for RVC", "vc"),
]


def exists(model_id: str) -> tuple[bool, int, str]:
    url = f"https://huggingface.co/api/models/{urllib.parse.quote(model_id, safe='/')}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode())
            return True, int(data.get("downloads") or 0), data.get("pipeline_tag") or "-"
    except Exception:
        return False, 0, "-"


def main() -> int:
    out = []
    seen = set()
    for mid, note, kind in CURATED:
        if mid in seen:
            continue
        seen.add(mid)
        ok, downloads, pipe = exists(mid)
        if not ok:
            print("MISSING", mid)
            continue
        out.append((mid, note, kind, downloads, pipe))

    # Prefer e2e/vc/s2t first, then cascade pieces
    order = {"e2e": 0, "e2e-quant": 1, "s2t": 2, "vc": 3, "asr": 4, "mt": 5, "tts": 6}
    out.sort(key=lambda x: (order.get(x[2], 9), -x[3]))
    picks = out[:50]
    print(f"verified={len(out)} picked={len(picks)}")
    lines = []
    for i, (mid, note, kind, downloads, pipe) in enumerate(picks, 1):
        line = f"{i:02d}\t{kind:<10}\t{downloads:>10}\t{pipe:<28}\t{mid}\t{note}"
        lines.append(line)
    text = "\n".join(lines)
    Path("scripts/_hf_s2st_50.txt").write_text(text, encoding="utf-8")
    print(text.encode("ascii", "replace").decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
