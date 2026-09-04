"""Локальная веб-студия дубляжа (без лимита Telegram ~20 МБ)."""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

from aiohttp import web

from app.config import Settings
from app.database import Database
from app.services.large_media import LargeMediaService
from app.services.video_dub import VideoDubService
from app.text.reply_lang import REPLY_LANGUAGES, normalize_reply_lang
from app.web.jobs import JobStore

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _client_error(message: str, status: int = 400) -> web.Response:
    return web.json_response({"ok": False, "error": message}, status=status)


async def create_web_app(
    *,
    settings: Settings,
    db: Database,
    video_dub: VideoDubService,
) -> web.Application:
    store = JobStore(settings.data_dir / "tmp" / "web_dub")
    user_id = int(settings.web_user_id)
    if user_id > 0:
        await db.set_consent(user_id, True)

    app = web.Application(client_max_size=int(settings.web_max_upload_mb * 1024 * 1024))
    app["settings"] = settings
    app["db"] = db
    app["video_dub"] = video_dub
    app["jobs"] = store
    app["user_id"] = user_id

    app.router.add_get("/", handle_index)
    app.router.add_get("/api/health", handle_health)
    app.router.add_get("/api/languages", handle_languages)
    app.router.add_post("/api/jobs", handle_upload)
    app.router.add_get("/api/jobs/{job_id}", handle_job)
    app.router.add_post("/api/jobs/{job_id}/dub", handle_start_dub)
    app.router.add_get("/api/jobs/{job_id}/video", handle_download_video)
    app.router.add_get("/api/jobs/{job_id}/srt", handle_download_srt)
    app.router.add_static("/static/", path=str(STATIC_DIR), name="static")
    return app


async def start_web_server(
    *,
    settings: Settings,
    db: Database,
    video_dub: VideoDubService,
) -> web.AppRunner:
    app = await create_web_app(settings=settings, db=db, video_dub=video_dub)
    runner = web.AppRunner(app)
    await runner.setup()
    host = settings.web_host or "0.0.0.0"
    ports = [int(settings.web_port)]
    # Публичный IP без :порта — слушаем и 80, если основной порт другой
    if host not in {"127.0.0.1", "localhost"} and 80 not in ports:
        ports.append(80)
    bound: list[int] = []
    for port in ports:
        try:
            site = web.TCPSite(runner, host, port)
            await site.start()
            bound.append(port)
        except OSError as exc:
            logger.warning("Не удалось слушать %s:%s (%s)", host, port, exc)
    if not bound:
        raise RuntimeError("Веб-сервер не смог открыть ни один порт")
    public = (settings.web_public_url or "").rstrip("/")
    logger.info(
        "Web dub studio bind %s ports=%s public=%s (max upload %.0f МБ)",
        host,
        bound,
        public or f"http://{host}:{bound[0]}",
        settings.web_max_upload_mb,
    )
    return runner


async def handle_index(_: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC_DIR / "index.html")


async def handle_health(request: web.Request) -> web.Response:
    settings: Settings = request.app["settings"]
    return web.json_response(
        {
            "ok": True,
            "max_upload_mb": settings.web_max_upload_mb,
            "telegram_limit_mb": settings.video_dub_max_mb,
        }
    )


async def handle_languages(_: web.Request) -> web.Response:
    return web.json_response(
        {
            "ok": True,
            "languages": [
                {"code": code, "label": label} for code, label in REPLY_LANGUAGES.items()
            ],
        }
    )


async def handle_upload(request: web.Request) -> web.Response:
    settings: Settings = request.app["settings"]
    store: JobStore = request.app["jobs"]
    video_dub: VideoDubService = request.app["video_dub"]

    reader = await request.multipart()
    field = await reader.next()
    if field is None or field.name != "video":
        return _client_error("Ожидалось поле video")

    filename = Path(field.filename or "upload.mp4").name
    suffix = Path(filename).suffix.lower() or ".mp4"
    if suffix not in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}:
        return _client_error(f"Неподдерживаемый формат: {suffix}")

    job = await store.create()
    dest = job.workdir / f"source{suffix}"
    size = 0
    max_bytes = int(settings.web_max_upload_mb * 1024 * 1024)
    with dest.open("wb") as out:
        while True:
            chunk = await field.read_chunk(size=1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                out.close()
                shutil.rmtree(job.workdir, ignore_errors=True)
                return _client_error(
                    f"Файл больше {settings.web_max_upload_mb:.0f} МБ",
                    status=413,
                )
            out.write(chunk)

    job.video_path = dest
    job.touch("analyzing", "Разбираю речь и тайминги…")

    async def _analyze() -> None:
        try:
            assert job.video_path is not None
            segments, duration = await video_dub.analyze(job.video_path)
            job.segments = segments
            job.duration_sec = duration
            job.touch(
                "ready",
                f"Готово: {len(segments)} фраз · {sum(s.duration for s in segments):.1f}с речи",
            )
        except Exception as exc:
            logger.exception("Web analyze failed %s", job.id)
            job.error = str(exc)
            job.touch("error", "Ошибка разбора")

    job.task = asyncio.create_task(_analyze())
    return web.json_response({"ok": True, "job": job.to_public()})


async def handle_job(request: web.Request) -> web.Response:
    store: JobStore = request.app["jobs"]
    job = store.get(request.match_info["job_id"])
    if job is None:
        return _client_error("Задача не найдена", status=404)
    lite = request.rel_url.query.get("lite")
    if lite is None:
        lite_flag = job.status in {"translating", "rendering", "queued"}
    else:
        lite_flag = lite.lower() in {"1", "true", "yes"}
    return web.json_response({"ok": True, "job": job.to_public(lite=lite_flag)})


async def handle_start_dub(request: web.Request) -> web.Response:
    store: JobStore = request.app["jobs"]
    video_dub: VideoDubService = request.app["video_dub"]
    user_id: int = request.app["user_id"]
    job = store.get(request.match_info["job_id"])
    if job is None:
        return _client_error("Задача не найдена", status=404)
    if job.status not in {"ready", "done", "error"}:
        return _client_error(f"Сейчас статус: {job.status}")
    if not job.segments or job.video_path is None:
        return _client_error("Нет сегментов — сначала дождитесь разбора")

    body = await request.json()
    lang = normalize_reply_lang(str(body.get("language") or "en"))
    job.language = lang
    job.translated = []
    job.result_video = None
    job.result_srt = None
    job.error = ""
    job.progress_done = 0
    job.progress_total = len(job.segments)
    job.touch("translating", f"Перевод → {REPLY_LANGUAGES.get(lang, lang)}")

    async def _run() -> None:
        try:
            translated = await video_dub.translate_segments(
                job.segments,
                lang,
                media_duration=getattr(job, "duration_sec", None),
                user_id=user_id,
            )
            job.translated = translated
            job.touch("rendering", "Клон голоса из видео и озвучка…")

            async def on_progress(done: int, total: int, preview: str) -> None:
                job.progress_done = done
                job.progress_total = total
                job.progress_preview = preview
                if done <= 0:
                    job.message = preview or "Клонирую голос…"
                else:
                    job.message = f"Озвучка {done}/{total}"

            result = await video_dub.render(
                user_id,
                job.video_path,
                job.segments,
                translated,
                lang,
                job.duration_sec,
                on_progress=on_progress,
            )
            # копируем в workdir задачи для скачивания
            out_video = job.workdir / "dubbed.mp4"
            out_srt = job.workdir / "dubbed.srt"
            shutil.copy2(result.video_path, out_video)
            if result.srt_path and result.srt_path.exists():
                shutil.copy2(result.srt_path, out_srt)
                job.result_srt = out_srt
            LargeMediaService(video_dub.settings).save_to_output(
                result.video_path,
                result.srt_path,
                label=lang,
            )
            job.result_video = out_video
            job.clone_sec = result.clone_sec
            job.clone_clips = len(result.clone_refs)
            job.touch("done", "Замена языка готова")
        except Exception as exc:
            logger.exception("Web dub failed %s", job.id)
            job.error = str(exc)
            job.touch("error", "Ошибка дубляжа")

    if job.task and not job.task.done():
        job.task.cancel()
    job.task = asyncio.create_task(_run())
    return web.json_response({"ok": True, "job": job.to_public()})


async def handle_download_video(request: web.Request) -> web.StreamResponse:
    store: JobStore = request.app["jobs"]
    job = store.get(request.match_info["job_id"])
    if job is None:
        return _client_error("Задача не найдена", status=404)
    if job.status != "done" or not job.result_video or not job.result_video.exists():
        msg = "Видео ещё не готово"
        if job.status in {"translating", "rendering", "analyzing", "queued"}:
            msg = (
                f"Ещё идёт {job.status}: {job.progress_done}/{job.progress_total}. "
                "Дождитесь статуса «Замена языка готова»."
            )
        elif job.status == "error":
            msg = f"Замена языка упала: {job.error or job.message}"
        return _client_error(msg, status=409)
    return web.FileResponse(
        job.result_video,
        headers={"Content-Disposition": 'attachment; filename="dubbed.mp4"'},
    )


async def handle_download_srt(request: web.Request) -> web.StreamResponse:
    store: JobStore = request.app["jobs"]
    job = store.get(request.match_info["job_id"])
    if job is None or not job.result_srt or not job.result_srt.exists():
        return _client_error("SRT ещё не готов", status=404)
    return web.FileResponse(
        job.result_srt,
        headers={"Content-Disposition": 'attachment; filename="dubbed.srt"'},
    )
