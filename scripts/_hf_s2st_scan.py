"""Search Hugging Face for speech replacement / translation models likely fit for <=4GB VRAM."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

QUERIES = [
    "seamless-m4t",
    "speech-to-speech",
    "s2st",
    "voice conversion",
    "speech translation",
    "openvoice",
    "rvc",
    "whisper tiny",
    "whisper base",
    "whisper small",
    "mms-tts",
    "speecht5",
    "unitY",
    "nllb-200-distilled",
    "marianmt",
    "vits",
    "coqui xtts",
    "bark small",
    "yourtts",
    "freevc",
    "so-vits-svc",
    "fairseq s2t",
    "whisper-tiny.en",
]


def hf_search(q: str, limit: int = 80) -> list[dict]:
    url = "https://huggingface.co/api/models?" + urllib.parse.urlencode(
        {"search": q, "limit": str(limit), "sort": "downloads", "direction": "-1"}
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=45) as resp:
        return json.loads(resp.read().decode())


def keep(mid: str, tags: list[str], pipe: str) -> bool:
    blob = f"{mid} {' '.join(tags)} {pipe}".lower()
    keys = (
        "speech",
        "audio",
        "voice",
        "tts",
        "asr",
        "seamless",
        "whisper",
        "mms",
        "rvc",
        "openvoice",
        "s2st",
        "translation",
        "dub",
        "vocoder",
        "xtts",
        "coqui",
        "bark",
        "vits",
        "speecht5",
        "fairseq",
        "nllb",
        "marian",
        "svc",
        "freevc",
        "yourtts",
        "styletts",
        "parler",
        "tortoise",
        "so-vits",
        "hubert",
        "wav2vec",
        "s2t",
        "stt",
        "t5",
    )
    return any(k in blob for k in keys)


def estimate_fit_4gb(mid: str, tags: list[str]) -> str:
    low = mid.lower()
    tag_blob = " ".join(tags).lower()
    # Hard no for huge models without quantization note
    huge = (
        "large-v3",
        "large-v2",
        "whisper-large",
        "seamless-m4t-v2-large",
        "seamless-m4t-large",
        "nllb-200-3.3b",
        "3.3b",
        "7b",
        "13b",
        "70b",
        "xtts_v2",  # often ok at ~2-4GB but borderline with clone refs
    )
    small = (
        "tiny",
        "base",
        "small",
        "mini",
        "distil",
        "unity-small",
        "medium",
        "600m",
        "1.3b",
        "418m",
        "300m",
        "125m",
        "80m",
        "vits",
        "speecht5",
        "mms-tts",
        "marian",
        "opus-mt",
        "openvoice",
        "rvc",
        "freevc",
        "bark-small",
        "whisper-small",
        "whisper-base",
        "whisper-tiny",
        "nllb-200-distilled-600m",
        "nllb-200-1.3b",
        "seamless-m4t-medium",
        "hf-seamless-m4t-medium",
    )
    if any(h in low for h in huge) and "int8" not in low and "4bit" not in low and "gguf" not in low:
        if "medium" in low or "distil" in low or "tiny" in low or "base" in low or "small" in low:
            return "likely"
        return "borderline/quantize"
    if any(s in low for s in small) or "gguf" in low or "int8" in low or "4bit" in low or "onnx" in tag_blob:
        return "likely"
    return "possible"


def main() -> int:
    seen: dict[str, dict] = {}
    for q in QUERIES:
        try:
            rows = hf_search(q)
        except Exception as exc:  # noqa: BLE001
            print("FAIL", q, type(exc).__name__, exc)
            continue
        for m in rows:
            mid = m.get("id") or m.get("modelId")
            if not mid or mid in seen:
                continue
            tags = list(m.get("tags") or [])
            pipe = m.get("pipeline_tag") or ""
            if not keep(mid, tags, pipe):
                continue
            seen[mid] = {
                "id": mid,
                "downloads": int(m.get("downloads") or 0),
                "likes": int(m.get("likes") or 0),
                "pipeline": pipe or "-",
                "fit": estimate_fit_4gb(mid, tags),
            }

    preferred = [x for x in seen.values() if x["fit"] in {"likely", "possible"}]
    borderline = [x for x in seen.values() if x["fit"] == "borderline/quantize"]
    preferred.sort(key=lambda x: (-x["downloads"], -x["likes"]))
    borderline.sort(key=lambda x: (-x["downloads"], -x["likes"]))
    picks = preferred[:50]
    if len(picks) < 50:
        picks.extend(borderline[: 50 - len(picks)])

    print(f"candidates={len(seen)} picked={len(picks)}")
    for i, m in enumerate(picks, 1):
        print(
            f"{i:02d}\t{m['fit']}\t{m['downloads']:>10}\t{m['pipeline']:<28}\t{m['id']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
