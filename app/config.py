"""Конфигурация приложения."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        # utf-8-sig: игнорирует BOM от Windows/PowerShell Set-Content
        env_file_encoding="utf-8-sig",
        extra="ignore",
    )

    telegram_bot_token: str = Field("", alias="TELEGRAM_BOT_TOKEN")
    # User API (Telethon) — сбор своих голосовых из всех чатов аккаунта
    telegram_api_id: int | None = Field(None, alias="TELEGRAM_API_ID")
    telegram_api_hash: str = Field("", alias="TELEGRAM_API_HASH")
    telegram_phone: str = Field("", alias="TELEGRAM_PHONE")
    telegram_session_name: str = Field(
        "session_user",
        alias="TELEGRAM_SESSION_NAME",
    )
    # SOCKS5/HTTP proxy for Telethon MTProto (нужен, если DC Telegram заблокированы)
    # Примеры: socks5://127.0.0.1:1080  или  socks5://user:pass@host:1080
    telegram_proxy: str = Field("", alias="TELEGRAM_PROXY")
    telegram_bot_proxy: str = Field("", alias="TELEGRAM_BOT_PROXY")
    # Bot API file download (aiogram default = 30с — мало для MP4)
    telegram_download_timeout_sec: float = Field(
        300.0, alias="TELEGRAM_DOWNLOAD_TIMEOUT_SEC"
    )
    # GigaChat API: Authorization key из GigaChat Studio (base64 client_id:secret)
    gigachat_credentials: str = Field("", alias="GIGACHAT_CREDENTIALS")
    gigachat_scope: Literal[
        "GIGACHAT_API_PERS",
        "GIGACHAT_API_B2B",
        "GIGACHAT_API_CORP",
    ] = Field("GIGACHAT_API_PERS", alias="GIGACHAT_SCOPE")
    gigachat_model: str = Field("GigaChat", alias="GIGACHAT_MODEL")
    gigachat_system_prompt: str = Field(
        (
            "Ты голосовой ассистент для телефонного разговора. Отвечай на русском "
            "кратко, естественно и разговорно. Не используй Markdown, таблицы, "
            "эмодзи, списки и латиницу без необходимости. Пиши только то, что "
            "будет озвучено синтезатором речи.\n\n"
            "ИНТОНАЦИЯ И ВЫРАЗИТЕЛЬНОСТЬ (ОЧЕНЬ ВАЖНО):\n"
            "Твой текст будет озвучен нейросетью TTS, поэтому:\n"
            "— Используй разнообразные знаки препинания для управления интонацией: "
            "вопросительные знаки (?) для вопросов, восклицательные (!) для эмоций, "
            "многоточия (...) для пауз и задумчивости, тире (—) для смысловых пауз.\n"
            "— Пиши живо и выразительно, как настоящий человек по телефону: "
            "«Ну конечно!», «Да, слушаю вас...», «О, это интересно!», "
            "«Хм, давайте подумаем...»\n"
            "— Чередуй длинные и короткие предложения для естественного ритма.\n"
            "— Используй вводные слова и междометия: ну, вот, конечно, разумеется, "
            "кстати, знаете.\n"
            "— Избегай сухих однообразных конструкций. Каждый ответ должен звучать "
            "живо и эмоционально.\n\n"
            "КРИТИЧЕСКИ ВАЖНО — ударения для TTS:\n"
            "1) В КАЖДОМ русском слове с гласной поставь ровно один знак + "
            "непосредственно ПЕРЕД ударной гласной. Формат: зам+ок, звон+ит, "
            "догов+ор, н+аш, к+аждую, нед+елю.\n"
            "2) Букву «ё» пиши ТОЛЬКО где она нужна по орфографии "
            "(всё, ещё, её, чёрный, берёза, зелёный). "
            "Ударная «е» в словах вроде неделя, время, телефон, конечно — "
            "это «е», НЕ «ё»: нед+еля, вр+емя, телеф+он, кон+ечно.\n"
            "3) Не пропускай служебные и короткие слова: н+а, п+о, +и, н+е, "
            "м+еня, теб+я, сег+одня, сейч+ас.\n"
            "4) В каждом слове только один '+'. Не ставь '+' перед согласной.\n"
            "5) Учитывай омографы: з+амок / зам+ок, п+исьма / письм+а.\n"
            "Пример: "
            "«Н+у кон+ечно! Н+аш катал+ог обновл+яется к+аждую нед+елю. "
            "Звон+ите п+о догов+ору — б+удем р+ады пом+очь!»"
        ),
        alias="GIGACHAT_SYSTEM_PROMPT",
    )
    gigachat_temperature: float = Field(0.7, alias="GIGACHAT_TEMPERATURE")
    gigachat_max_tokens: int = Field(500, alias="GIGACHAT_MAX_TOKENS")
    gigachat_history_turns: int = Field(6, alias="GIGACHAT_HISTORY_TURNS")
    gigachat_timeout_sec: float = Field(60.0, alias="GIGACHAT_TIMEOUT_SEC")
    gigachat_verify_ssl: bool = Field(True, alias="GIGACHAT_VERIFY_SSL")
    gigachat_ca_bundle_file: Path | None = Field(
        None,
        alias="GIGACHAT_CA_BUNDLE_FILE",
    )
    gigachat_base_url: str = Field(
        "https://gigachat.devices.sberbank.ru/api/v1",
        alias="GIGACHAT_BASE_URL",
    )
    gigachat_auth_url: str = Field(
        "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
        alias="GIGACHAT_AUTH_URL",
    )
    enable_ruaccent: bool = Field(True, alias="ENABLE_RUACCENT")
    ruaccent_model_size: str = Field("turbo3.1", alias="RUACCENT_MODEL_SIZE")
    ruaccent_use_dictionary: bool = Field(True, alias="RUACCENT_USE_DICTIONARY")
    ruaccent_workdir: Path = Field(
        Path("./data/models/ruaccent"),
        alias="RUACCENT_WORKDIR",
    )
    custom_accents_path: Path = Field(
        Path("./data/custom_accents_ru.json"),
        alias="CUSTOM_ACCENTS_PATH",
    )

    # Telegram voice → faster-whisper (CPU по умолчанию, CUDA оставляем XTTS)
    stt_model_size: str = Field("small", alias="STT_MODEL_SIZE")
    stt_device: Literal["cpu", "cuda"] = Field("cpu", alias="STT_DEVICE")
    stt_compute_type: str = Field("int8", alias="STT_COMPUTE_TYPE")
    stt_language: str = Field("ru", alias="STT_LANGUAGE")
    stt_beam_size: int = Field(3, alias="STT_BEAM_SIZE")
    stt_chunk_seconds: float = Field(4.0, alias="STT_CHUNK_SECONDS")
    stt_max_voice_seconds: float = Field(180.0, alias="STT_MAX_VOICE_SECONDS")
    stt_max_analytics_chunks: int = Field(
        8,
        alias="STT_MAX_ANALYTICS_CHUNKS",
    )
    # Видео-дубляж: без жёсткого VAD (песни/музыка), чанки по длинному ролику
    video_dub_stt_language: str = Field("auto", alias="VIDEO_DUB_STT_LANGUAGE")
    # Video uses its own size: `small` misses words on noisy/conversational clips.
    video_dub_stt_model_size: str = Field("medium", alias="VIDEO_DUB_STT_MODEL_SIZE")
    video_dub_stt_beam_size: int = Field(8, alias="VIDEO_DUB_STT_BEAM_SIZE")
    video_dub_stt_vad: bool = Field(False, alias="VIDEO_DUB_STT_VAD")
    video_dub_stt_chunk_sec: float = Field(40.0, alias="VIDEO_DUB_STT_CHUNK_SEC")
    video_dub_stt_overlap_sec: float = Field(2.5, alias="VIDEO_DUB_STT_OVERLAP_SEC")
    video_dub_stt_no_speech_threshold: float = Field(
        0.35, alias="VIDEO_DUB_STT_NO_SPEECH_THRESHOLD"
    )
    # Quiet/ASMR STT: hard loudness boost + softer no-speech gate
    video_dub_quiet_stt_target_db: float = Field(
        -12.0, alias="VIDEO_DUB_QUIET_STT_TARGET_DB"
    )
    video_dub_quiet_stt_no_speech_threshold: float = Field(
        0.12, alias="VIDEO_DUB_QUIET_STT_NO_SPEECH_THRESHOLD"
    )
    video_dub_quiet_stt_min_coverage: float = Field(
        0.08, alias="VIDEO_DUB_QUIET_STT_MIN_COVERAGE"
    )
    # whisper = faster-whisper; whisperx = forced-align (wav2vec2) для точных слотов
    video_dub_stt_aligner: Literal["whisper", "whisperx"] = Field(
        "whisperx",
        alias="VIDEO_DUB_STT_ALIGNER",
    )
    video_dub_cue_max_sec: float = Field(6.5, alias="VIDEO_DUB_CUE_MAX_SEC")
    video_dub_cue_min_pause_sec: float = Field(0.22, alias="VIDEO_DUB_CUE_MIN_PAUSE_SEC")
    video_dub_cue_min_sec: float = Field(0.55, alias="VIDEO_DUB_CUE_MIN_SEC")
    video_dub_whisperx_chunk_sec: float = Field(90.0, alias="VIDEO_DUB_WHISPERX_CHUNK_SEC")
    telegram_edit_interval_sec: float = Field(
        1.1,
        alias="TELEGRAM_EDIT_INTERVAL_SEC",
    )
    account_collect_limit: int = Field(1000, alias="ACCOUNT_COLLECT_LIMIT")
    account_collect_per_dialog: int = Field(300, alias="ACCOUNT_COLLECT_PER_DIALOG")
    account_collect_max_seconds: float = Field(
        20000.0,
        alias="ACCOUNT_COLLECT_MAX_SECONDS",
    )
    account_collect_min_duration: float = Field(
        2.0,
        alias="ACCOUNT_COLLECT_MIN_DURATION",
    )
    account_collect_max_duration: float = Field(
        90.0,
        alias="ACCOUNT_COLLECT_MAX_DURATION",
    )
    # 0 = сканировать всю историю диалога
    account_collect_messages_per_dialog: int = Field(
        0,
        alias="ACCOUNT_COLLECT_MESSAGES_PER_DIALOG",
    )
    # Параллельные воркеры: загрузка+обработка / сканирование чатов
    account_collect_workers: int = Field(6, alias="ACCOUNT_COLLECT_WORKERS")
    # 1 = без flood wait от параллельного GetHistory
    account_collect_scan_workers: int = Field(1, alias="ACCOUNT_COLLECT_SCAN_WORKERS")
    # Сколько секунд лучших референсов брать в итоговый XTTS-профиль
    profile_max_seconds: float = Field(3600.0, alias="PROFILE_MAX_SECONDS")
    profile_max_files: int = Field(500, alias="PROFILE_MAX_FILES")
    profile_min_files: int = Field(500, alias="PROFILE_MIN_FILES")
    # Сколько лучших клипов реально кормить в XTTS conditioning (остальное — для fine-tune)
    xtts_conditioning_max_files: int = Field(80, alias="XTTS_CONDITIONING_MAX_FILES")
    xtts_conditioning_max_seconds: float = Field(
        600.0,
        alias="XTTS_CONDITIONING_MAX_SECONDS",
    )
    device: Literal["cuda", "cpu"] = Field("cuda", alias="DEVICE")
    tts_model_name: str = Field(
        "tts_models/multilingual/multi-dataset/xtts_v2",
        alias="TTS_MODEL_NAME",
    )
    default_language: str = Field("ru", alias="DEFAULT_LANGUAGE")
    max_text_length: int = Field(2000, alias="MAX_TEXT_LENGTH")
    max_reference_seconds: float = Field(180.0, alias="MAX_REFERENCE_SECONDS")
    min_reference_seconds: float = Field(30.0, alias="MIN_REFERENCE_SECONDS")
    min_voice_messages: int = Field(500, alias="MIN_VOICE_MESSAGES")
    max_voice_messages: int = Field(2000, alias="MAX_VOICE_MESSAGES")
    enable_denoise: bool = Field(True, alias="ENABLE_DENOISE")
    enable_ai_audio_marker: bool = Field(False, alias="ENABLE_AI_AUDIO_MARKER")
    ai_marker_text: str = Field(
        "Синтезировано искусственным интеллектом.",
        alias="AI_MARKER_TEXT",
    )
    data_dir: Path = Field(Path("./data"), alias="DATA_DIR")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    tts_queue_size: int = Field(1, alias="TTS_QUEUE_SIZE")
    default_speed: float = Field(1.0, alias="DEFAULT_SPEED")
    default_temperature: float = Field(0.75, alias="DEFAULT_TEMPERATURE")
    default_intonation: Literal["neutral", "calm", "expressive"] = Field(
        "expressive",
        alias="DEFAULT_INTONATION",
    )
    model_cache_dir: Path | None = Field(None, alias="MODEL_CACHE_DIR")
    use_fp16: bool = Field(False, alias="USE_FP16")

    # Hybrid TTS: auto = XTTS при наличии профиля, иначе Silero
    tts_engine: Literal[
        "auto",
        "xtts",
        "silero",
        "mockingbird",
        "voxcpm2",
        "seamless_m4t",
        "openrouter_fish",
    ] = Field(
        "auto",
        alias="TTS_ENGINE",
    )
    openrouter_api_key: str = Field("", alias="OPENROUTER_API_KEY")
    openrouter_tts_model: str = Field(
        "fish-audio/s2.1-pro-free:free",
        alias="OPENROUTER_TTS_MODEL",
    )
    openrouter_tts_timeout_sec: float = Field(
        120.0, alias="OPENROUTER_TTS_TIMEOUT_SEC"
    )
    # Free tier ≈ 20 req/min: minimal interval between speech requests.
    openrouter_tts_min_interval_sec: float = Field(
        3.1, alias="OPENROUTER_TTS_MIN_INTERVAL_SEC"
    )
    openrouter_tts_format: str = Field("mp3", alias="OPENROUTER_TTS_FORMAT")
    seamless_model_id: str = Field(
        "facebook/hf-seamless-m4t-medium",
        alias="SEAMLESS_MODEL_ID",
    )
    seamless_speaker_id: int = Field(0, alias="SEAMLESS_SPEAKER_ID")
    voxcpm_model_id: str = Field("openbmb/VoxCPM2", alias="VOXCPM_MODEL_ID")
    voxcpm_cfg_value: float = Field(1.6, alias="VOXCPM_CFG_VALUE")
    voxcpm_inference_timesteps: int = Field(20, alias="VOXCPM_INFERENCE_TIMESTEPS")
    voxcpm_timeout_sec: float = Field(180.0, alias="VOXCPM_TIMEOUT_SEC")
    mockingbird_root: Path = Field(
        Path("./third_party/MockingBird"),
        alias="MOCKINGBIRD_ROOT",
    )
    mockingbird_python: Path = Field(
        Path("./third_party/mockingbird-venv/Scripts/python.exe"),
        alias="MOCKINGBIRD_PYTHON",
    )
    mockingbird_encoder: Path = Field(
        Path("./data/mockingbird/encoder.pt"),
        alias="MOCKINGBIRD_ENCODER",
    )
    mockingbird_synthesizer: Path = Field(
        Path("./data/mockingbird/synthesizer.pt"),
        alias="MOCKINGBIRD_SYNTHESIZER",
    )
    mockingbird_vocoder: Path = Field(
        Path("./data/mockingbird/vocoder.pt"),
        alias="MOCKINGBIRD_VOCODER",
    )
    mockingbird_timeout_sec: float = Field(180.0, alias="MOCKINGBIRD_TIMEOUT_SEC")
    silero_model_path: Path = Field(
        Path("./assets/tts/silero/v5_5_ru.pt"),
        alias="SILERO_MODEL_PATH",
    )
    silero_speaker: str = Field("xenia", alias="SILERO_SPEAKER")
    silero_device: Literal["cpu", "cuda"] = Field("cpu", alias="SILERO_DEVICE")
    silero_cpu_threads: int = Field(4, alias="SILERO_CPU_THREADS")
    xtts_timeout_sec: float = Field(120.0, alias="XTTS_TIMEOUT_SEC")
    # Опциональный fine-tuned GPT checkpoint (после scripts/finetune_xtts.py)
    xtts_finetune_checkpoint: Path | None = Field(
        None,
        alias="XTTS_FINETUNE_CHECKPOINT",
    )
    xtts_finetune_config: Path | None = Field(
        None,
        alias="XTTS_FINETUNE_CONFIG",
    )
    # XTTS conditioning quality: больше = лучше клон, но медленнее computation
    xtts_gpt_cond_len: int = Field(30, alias="XTTS_GPT_COND_LEN")
    xtts_gpt_cond_chunk_len: int = Field(6, alias="XTTS_GPT_COND_CHUNK_LEN")
    xtts_max_ref_len: int = Field(30, alias="XTTS_MAX_REF_LEN")
    xtts_sound_norm_refs: bool = Field(True, alias="XTTS_SOUND_NORM_REFS")
    enable_audio_cache: bool = Field(True, alias="ENABLE_AUDIO_CACHE")
    audio_cache_dir: Path = Field(Path("./data/cache/tts"), alias="AUDIO_CACHE_DIR")
    # Более длинные фразы = связная речь, без «дикторских» обрывов
    phrase_min_chars: int = Field(60, alias="PHRASE_MIN_CHARS")
    phrase_soft_max_chars: int = Field(160, alias="PHRASE_SOFT_MAX_CHARS")
    phrase_max_chars: int = Field(250, alias="PHRASE_MAX_CHARS")

    # Call-feel: гудок → «алло» → ответ (иллюзия обзвона без Telegram VoIP)
    call_feel_enabled: bool = Field(True, alias="CALL_FEEL_ENABLED")
    call_feel_ringback_path: Path = Field(
        Path("./assets/call/ringback.ogg"),
        alias="CALL_FEEL_RINGBACK_PATH",
    )
    call_feel_ring_delay_sec: float = Field(
        1.1,
        alias="CALL_FEEL_RING_DELAY_SEC",
    )
    call_feel_pickup_delay_sec: float = Field(
        0.55,
        alias="CALL_FEEL_PICKUP_DELAY_SEC",
    )
    call_feel_alo_text: str = Field(
        "Алло? Да, слушаю.",
        alias="CALL_FEEL_ALO_TEXT",
    )
    # 0 = не греть на старте; иначе user_id с готовым XTTS-профилем
    call_feel_warmup_user_id: int = Field(
        0,
        alias="CALL_FEEL_WARMUP_USER_ID",
    )
    call_feel_early_first_phrase: bool = Field(
        True,
        alias="CALL_FEEL_EARLY_FIRST_PHRASE",
    )
    call_feel_min_first_chars: int = Field(
        12,
        alias="CALL_FEEL_MIN_FIRST_CHARS",
    )
    call_feel_min_rest_chars: int = Field(
        24,
        alias="CALL_FEEL_MIN_REST_CHARS",
    )

    # Живой Telegram-звонок (Telethon userbot) + barge-in
    call_enabled: bool = Field(True, alias="CALL_ENABLED")
    call_barge_in_enabled: bool = Field(True, alias="CALL_BARGE_IN_ENABLED")
    call_vad_silence_ms: int = Field(750, alias="CALL_VAD_SILENCE_MS")
    call_vad_speech_ms: int = Field(250, alias="CALL_VAD_SPEECH_MS")
    call_pcm_sample_rate: int = Field(48000, alias="CALL_PCM_SAMPLE_RATE")
    call_stop_phrases: str = Field(
        "стоп,подожди,подождите,молча,не говори,заткнись,хватит",
        alias="CALL_STOP_PHRASES",
    )
    call_topic_shift_phrases: str = Field(
        "другой вопрос,сменим тему,не об этом,погодите про другое,слушай другое",
        alias="CALL_TOPIC_SHIFT_PHRASES",
    )
    call_ring_timeout_sec: float = Field(45.0, alias="CALL_RING_TIMEOUT_SEC")
    call_interrupt_system_hint: str = Field(
        (
            "Пользователь перебил предыдущий ответ или сменил тему. "
            "Ответь только на новую реплику, коротко, без продолжения старого монолога."
        ),
        alias="CALL_INTERRUPT_SYSTEM_HINT",
    )

    # 0 = без ограничения длительности (лимит Telegram — размер файла)
    video_dub_max_seconds: float = Field(0.0, alias="VIDEO_DUB_MAX_SECONDS")
    video_dub_max_mb: float = Field(19.0, alias="VIDEO_DUB_MAX_MB")
    # Крупные файлы: Telethon / URL / папка inbox (без веб-студии)
    video_dub_use_telethon: bool = Field(True, alias="VIDEO_DUB_USE_TELETHON")
    video_dub_max_download_mb: float = Field(512.0, alias="VIDEO_DUB_MAX_DOWNLOAD_MB")
    video_dub_inbox_dir: str = Field("./data/inbox/dub", alias="VIDEO_DUB_INBOX_DIR")
    video_dub_output_dir: str = Field("./output", alias="VIDEO_DUB_OUTPUT_DIR")
    video_dub_bot_upload_mb: float = Field(49.0, alias="VIDEO_DUB_BOT_UPLOAD_MB")
    # replace = language replacement / vocal swap (фон 100%); duck = mix поверх исходника
    video_dub_mix_mode: Literal["replace", "duck"] = Field(
        "replace",
        alias="VIDEO_DUB_MIX_MODE",
    )
    video_dub_separation: Literal["auto", "demucs", "mask"] = Field(
        "auto",
        alias="VIDEO_DUB_SEPARATION",
    )
    video_dub_separation_device: Literal["cpu", "cuda"] = Field(
        "cpu",
        alias="VIDEO_DUB_SEPARATION_DEVICE",
    )
    video_dub_speech_mask_gain: float = Field(
        0.02,
        alias="VIDEO_DUB_SPEECH_MASK_GAIN",
    )
    video_dub_vocal_leak: float = Field(0.92, alias="VIDEO_DUB_VOCAL_LEAK")
    video_dub_duck_floor: float = Field(0.06, alias="VIDEO_DUB_DUCK_FLOOR")
    # replace/Demucs: bed at 1.0; duck: multiplier for original under new speech
    video_dub_bg_volume: float = Field(1.0, alias="VIDEO_DUB_BG_VOLUME")
    video_dub_voice_volume: float = Field(1.0, alias="VIDEO_DUB_VOICE_VOLUME")
    # Extra bed attenuation under placed TTS (1.0 = none).
    # 0.49 = default 0.70 × ещё −30% (мультипликативно, не аддитивно).
    video_dub_speech_duck: float = Field(0.49, alias="VIDEO_DUB_SPEECH_DUCK")
    # How far a cue may start BEFORE its original speech onset when the dub
    # chain overflows (pull-back into preceding silence/pause).
    video_dub_layout_max_early_sec: float = Field(
        1.5, alias="VIDEO_DUB_LAYOUT_MAX_EARLY_SEC"
    )
    # Drop Whisper hallucinations on moans/noise: high no_speech + low logprob.
    video_dub_drop_no_speech_prob: float = Field(
        0.6, alias="VIDEO_DUB_DROP_NO_SPEECH_PROB"
    )
    video_dub_drop_min_logprob: float = Field(
        -1.0, alias="VIDEO_DUB_DROP_MIN_LOGPROB"
    )
    # Max cues per video re-translated tighter when TTS overflows the slot.
    video_dub_overflow_retranslate: int = Field(
        8, alias="VIDEO_DUB_OVERFLOW_RETRANSLATE"
    )
    # Tempo fit disabled in render; kept for legacy callers / .env compat.
    video_dub_min_speed: float = Field(1.0, alias="VIDEO_DUB_MIN_SPEED")
    video_dub_max_speed: float = Field(1.0, alias="VIDEO_DUB_MAX_SPEED")
    # Min pause between placed TTS cues (silence-borrow layout).
    video_dub_phrase_gap_sec: float = Field(0.12, alias="VIDEO_DUB_PHRASE_GAP_SEC")
    video_dub_min_phrase_gap_sec: float = Field(
        0.12, alias="VIDEO_DUB_MIN_PHRASE_GAP_SEC"
    )
    # малый запас: иначе озвучка уезжает от губ оригинала
    video_dub_slot_slack_sec: float = Field(0.15, alias="VIDEO_DUB_SLOT_SLACK_SEC")
    # Референсы для клона — из речи в самом видео (не профиль пользователя)
    video_dub_clone_max_sec: float = Field(24.0, alias="VIDEO_DUB_CLONE_MAX_SEC")
    video_dub_clone_max_clips: int = Field(4, alias="VIDEO_DUB_CLONE_MAX_CLIPS")
    video_dub_clone_min_clip_sec: float = Field(2.4, alias="VIDEO_DUB_CLONE_MIN_CLIP_SEC")
    video_dub_clone_fallback_sec: float = Field(12.0, alias="VIDEO_DUB_CLONE_FALLBACK_SEC")
    # Шумодав клон-референсов (только при низком SNR — иначе рябь в TTS)
    video_dub_clone_denoise: bool = Field(True, alias="VIDEO_DUB_CLONE_DENOISE")
    video_dub_clone_denoise_snr_db: float = Field(
        14.0, alias="VIDEO_DUB_CLONE_DENOISE_SNR_DB"
    )
    video_dub_clone_denoise_prop: float = Field(
        0.72, alias="VIDEO_DUB_CLONE_DENOISE_PROP"
    )

    # Веб-студия дубляжа (без лимита Telegram Bot API ~20 МБ)
    web_enabled: bool = Field(True, alias="WEB_ENABLED")
    web_host: str = Field("127.0.0.1", alias="WEB_HOST")
    web_port: int = Field(8765, alias="WEB_PORT")
    web_max_upload_mb: float = Field(512.0, alias="WEB_MAX_UPLOAD_MB")
    web_user_id: int = Field(1327953308, alias="WEB_USER_ID")
    web_public_url: str = Field("http://127.0.0.1:8765", alias="WEB_PUBLIC_URL")

    # XTTS ожидает 22050 Hz для референсов при get_conditioning_latents
    reference_sample_rate: int = 22050
    output_sample_rate: int = 24000
    max_chunk_chars: int = 170

    @property
    def call_stop_phrase_list(self) -> list[str]:
        return [p.strip().lower() for p in self.call_stop_phrases.split(",") if p.strip()]

    @property
    def call_topic_shift_phrase_list(self) -> list[str]:
        return [
            p.strip().lower()
            for p in self.call_topic_shift_phrases.split(",")
            if p.strip()
        ]

    @property
    def mtproto_proxy_url(self) -> str:
        """SOCKS для Telethon: локальный hop (BOT_PROXY), иначе TELEGRAM_PROXY."""
        return (self.telegram_bot_proxy or self.telegram_proxy or "").strip()

    @property
    def users_dir(self) -> Path:
        return self.data_dir / "users"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def pronunciation_dict_path(self) -> Path:
        return self.data_dir / "pronunciation_ru.json"


def get_settings() -> Settings:
    return Settings()
