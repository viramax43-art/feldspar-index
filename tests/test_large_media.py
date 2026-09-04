from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from app.services.large_media import LargeMediaError, LargeMediaService


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "TELEGRAM_BOT_TOKEN": "t",
        "DATA_DIR": tmp_path / "data",
        "TELEGRAM_SESSION_NAME": "session_user",
        "TELEGRAM_PROXY": "",
        "TELEGRAM_BOT_PROXY": "",
        "_env_file": None,
    }
    values.update(overrides)
    return Settings(**values)


def test_session_path_creates_parent(tmp_path: Path) -> None:
    service = LargeMediaService(_settings(tmp_path))
    path = service.session_path()
    assert path.parent.is_dir()
    assert path.parent.name == "sessions"
    with pytest.raises(LargeMediaError, match="Нет файла сессии"):
        service._telegram_client()


def test_telethon_prefers_bot_proxy_hop(tmp_path: Path) -> None:
    service = LargeMediaService(
        _settings(
            tmp_path,
            TELEGRAM_PROXY="socks5://user:pass@10.0.0.1:1080",
            TELEGRAM_BOT_PROXY="socks5://127.0.0.1:11080",
        )
    )
    kwargs = service._client_kwargs()
    assert kwargs["proxy"][1] == "127.0.0.1"
    assert kwargs["proxy"][2] == 11080


def test_telethon_falls_back_to_telegram_proxy(tmp_path: Path) -> None:
    service = LargeMediaService(
        _settings(tmp_path, TELEGRAM_PROXY="socks5://127.0.0.1:1080")
    )
    kwargs = service._client_kwargs()
    assert kwargs["proxy"][1] == "127.0.0.1"
    assert kwargs["proxy"][2] == 1080


def test_partial_cache_path(tmp_path: Path) -> None:
    service = LargeMediaService(_settings(tmp_path))
    path = service._partial_path("abc/DEF 12")
    assert path is not None
    assert path.parent.name == "tg_cache"
    assert path.name.endswith(".part")
    assert "/" not in path.name
    assert service._partial_path(None) is None
