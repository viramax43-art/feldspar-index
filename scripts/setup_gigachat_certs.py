#!/usr/bin/env python3
"""
Собрать CA-bundle с российскими корневыми сертификатами Минцифры для GigaChat.

GigaChat (ngw.devices.sberbank.ru, gigachat.devices.sberbank.ru) использует
цепочку, подписанную «Russian Trusted Root CA». Его нет в стандартном хранилище,
поэтому Python выдаёт SSL: CERTIFICATE_VERIFY_FAILED.

Скрипт скачивает Root + Sub CA Минцифры и объединяет их со стандартным
bundle certifi в один файл. Затем укажите его в .env:

    GIGACHAT_CA_BUNDLE_FILE=./assets/certs/russian_trusted_ca.pem
"""

from __future__ import annotations

import argparse
import ssl
import sys
import urllib.request
from pathlib import Path

# Официальные адреса сертификатов Минцифры
ROOT_CA_URL = "https://gu-st.ru/content/lending/russian_trusted_root_ca_pem.crt"
SUB_CA_URL = "https://gu-st.ru/content/lending/russian_trusted_sub_ca_pem.crt"
DEFAULT_OUT = Path("assets/certs/russian_trusted_ca.pem")


def _download(url: str) -> bytes:
    # Bootstrap: сам gu-st.ru тоже подписан российским CA, которого пока нет
    # в хранилище, поэтому одноразовая загрузка идёт без верификации.
    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(url, headers={"User-Agent": "setup"})
    with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
        data = resp.read()
    if b"BEGIN CERTIFICATE" not in data:
        raise RuntimeError(f"Ответ {url} не похож на PEM-сертификат")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Собрать CA-bundle для GigaChat")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--no-certifi",
        action="store_true",
        help="Не добавлять стандартный bundle certifi (только российские CA)",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    parts: list[bytes] = []
    if not args.no_certifi:
        try:
            import certifi

            parts.append(Path(certifi.where()).read_bytes())
            print(f"Добавлен certifi: {certifi.where()}")
        except Exception as exc:
            print(f"certifi недоступен ({exc}), продолжаю только с российскими CA")

    try:
        print(f"Скачиваю Root CA: {ROOT_CA_URL}")
        parts.append(b"\n" + _download(ROOT_CA_URL) + b"\n")
        print(f"Скачиваю Sub CA:  {SUB_CA_URL}")
        parts.append(b"\n" + _download(SUB_CA_URL) + b"\n")
    except Exception as exc:
        print(f"Ошибка загрузки сертификатов: {exc}", file=sys.stderr)
        print(
            "Если gu-st.ru недоступен, установите сертификаты вручную с "
            "https://www.gosuslugi.ru/crt",
            file=sys.stderr,
        )
        return 1

    args.output.write_bytes(b"".join(parts))
    print(f"\nГотово: {args.output}")
    print("Добавьте в .env:")
    print(f"  GIGACHAT_CA_BUNDLE_FILE=./{args.output.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
