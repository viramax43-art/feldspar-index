#!/usr/bin/env python3
"""Долгоживущий worker MockingBird (отдельный venv). JSON-lines stdin/stdout."""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

# Запускать из корня клона MockingBird
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# NumPy 2 убрал np.cumproduct — подменяем до импорта моделей
import numpy as np

if not hasattr(np, "cumproduct"):
    np.cumproduct = np.cumprod  # type: ignore[attr-defined]


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# Библиотеки MockingBird пишут в stdout — оставляем stdout только для JSON
_REAL_STDOUT = sys.stdout
sys.stdout = sys.stderr


def _reply(payload: dict) -> None:
    _REAL_STDOUT.write(json.dumps(payload, ensure_ascii=False) + "\n")
    _REAL_STDOUT.flush()


def load_stack(enc: Path, syn: Path, voc: Path):
    import numpy as np
    import torch
    from models.encoder import inference as encoder
    from models.synthesizer.inference import Synthesizer

    vocoder_kind = "hifigan"
    if voc.name.lower().startswith("g_hifi") or "hifi" in voc.name.lower():
        from models.vocoder.hifigan import inference as vocoder
    else:
        from models.vocoder.wavernn import inference as vocoder
        vocoder_kind = "wavernn"

    _log(f"cuda={torch.cuda.is_available()} encoder={enc} synth={syn} voc={voc} ({vocoder_kind})")
    encoder.load_model(enc)
    synthesizer = Synthesizer(syn)
    vocoder.load_model(voc)
    return encoder, synthesizer, vocoder, vocoder_kind


def _speaker_embed(encoder, synthesizer, speaker_paths: list[Path]):
    import numpy as np

    embeds = []
    for path in speaker_paths[:8]:
        if not path.exists():
            continue
        wav_in = synthesizer.load_preprocess_wav(path)
        embed, _, _ = encoder.embed_utterance(wav_in, return_partials=True)
        embeds.append(embed)
    if not embeds:
        raise ValueError("нет speaker wav для клонирования")
    return np.mean(np.stack(embeds, axis=0), axis=0)


def synth_one(encoder, synthesizer, vocoder, speaker_paths: list[Path], text: str, out_path: Path) -> int:
    import numpy as np
    import soundfile as sf

    embed = _speaker_embed(encoder, synthesizer, speaker_paths)
    specs = synthesizer.synthesize_spectrograms(
        [text],
        [embed],
        style_idx=-1,
        min_stop_token=4,
        steps=400,
    )
    spec = specs[0]
    generated, sr = vocoder.infer_waveform(
        spec,
        batched=True,
        target=8000,
        overlap=400,
        progress_callback=lambda *_args: None,
    )
    generated = encoder.preprocess_wav(generated)
    peak = float(np.abs(generated).max() or 1.0)
    generated = generated / peak * 0.97
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), generated, int(sr or synthesizer.sample_rate))
    return int(sr or synthesizer.sample_rate)


def main() -> None:
    enc = Path(os.environ["MB_ENCODER"])
    syn = Path(os.environ["MB_SYNTHESIZER"])
    voc = Path(os.environ["MB_VOCODER"])
    for p in (enc, syn, voc):
        if not p.exists():
            _reply({"ok": False, "error": f"нет файла модели: {p}"})
            return
    try:
        encoder, synthesizer, vocoder, _kind = load_stack(enc, syn, voc)
    except Exception as exc:
        _reply({"ok": False, "error": f"load failed: {exc}"})
        traceback.print_exc()
        return
    _reply({"ok": True, "ready": True})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            _reply({"ok": False, "error": f"bad json: {exc}"})
            continue
        cmd = req.get("cmd")
        if cmd == "quit":
            _reply({"ok": True, "bye": True})
            return
        if cmd == "ping":
            _reply({"ok": True, "pong": True})
            continue
        if cmd != "synth":
            _reply({"ok": False, "error": f"unknown cmd {cmd}"})
            continue
        try:
            speakers = req.get("speakers") or [req.get("speaker")]
            sr = synth_one(
                encoder,
                synthesizer,
                vocoder,
                [Path(p) for p in speakers if p],
                str(req["text"]),
                Path(req["out"]),
            )
            _reply({"ok": True, "path": req["out"], "sr": sr})
        except Exception as exc:
            traceback.print_exc()
            _reply({"ok": False, "error": str(exc)})


if __name__ == "__main__":
    main()
