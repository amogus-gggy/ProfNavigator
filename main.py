from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import ujson
import random
import uuid
import time
from datetime import datetime
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor
import asyncio
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache
from contextlib import asynccontextmanager


# ── ProcessPoolExecutor worker (module-level для pickle) ──────────────────────

_worker_model: Optional["SurveyModel"] = None


def _init_worker():
    """Инициализатор воркера: загружает модель один раз при старте процесса."""
    global _worker_model
    from model import SurveyModel
    _worker_model = SurveyModel()


def _predict_in_worker(category_counts: dict) -> dict:
    """CPU-bound предсказание — выполняется в отдельном процессе."""
    return _worker_model.predict(category_counts)


# ── Job queue ─────────────────────────────────────────────────────────────────

JOB_TTL = 300  # секунд до удаления завершённых задач


@dataclass
class Job:
    id: str
    status: str          # pending | processing | done | error
    category_counts: dict
    result: Optional[dict] = None
    error: Optional[str] = None
    finished_at: Optional[float] = None


_jobs: Dict[str, Job] = {}
_pending_order: List[str] = []   # упорядоченный список ожидающих job_id
_job_queue: asyncio.Queue = None
_process_pool: ProcessPoolExecutor = None
_job_worker_task: asyncio.Task = None


def get_cloudflare_ip(request: Request) -> str:
    return request.headers.get("cf-connecting-ip") or (
        request.client.host if request.client else "127.0.0.1"
    )


limiter = Limiter(key_func=get_cloudflare_ip)

# Глобальный кеш вопросов и options_map (инициализируется при старте)
_questions_data: dict = {}
_options_map: dict = {}

# Очередь и лок для батч-записи ответов
_responses_queue: asyncio.Queue = None
_responses_lock: asyncio.Lock = None
_flush_task: asyncio.Task = None

FLUSH_INTERVAL = 5  # секунд между записями на диск
FLUSH_BATCH_SIZE = 50  # максимум записей за раз


async def _process_jobs():
    """Фоновый воркер: обрабатывает джобы из очереди по одному."""
    while True:
        job_id = await _job_queue.get()
        job = _jobs.get(job_id)

        if job is None:
            _job_queue.task_done()
            continue

        try:
            _pending_order.remove(job_id)
        except ValueError:
            pass

        job.status = "processing"

        try:
            loop = asyncio.get_event_loop()
            prediction = await loop.run_in_executor(
                _process_pool, _predict_in_worker, job.category_counts
            )
            sphere_info = _questions_data["spheres"][prediction["primary"]]
            job.result = {
                "primary": prediction["primary"],
                "sphere": sphere_info["name"],
                "description": sphere_info["description"],
                "confidence": prediction["confidence"],
                "probabilities": prediction["probabilities"],
                "ranking": prediction["ranking"],
                "answer_distribution": prediction["answer_distribution"],
                "reasoning": prediction["reasoning"],
            }
            job.status = "done"
            _enqueue_response(job.category_counts, prediction["primary"])
        except Exception as e:
            job.status = "error"
            job.error = str(e)
        finally:
            job.finished_at = time.time()
            _job_queue.task_done()
            _cleanup_expired_jobs()


def _cleanup_expired_jobs():
    """Удаляет завершённые джобы старше JOB_TTL секунд."""
    now = time.time()
    expired = [
        jid for jid, j in _jobs.items()
        if j.finished_at and (now - j.finished_at) > JOB_TTL
    ]
    for jid in expired:
        _jobs.pop(jid, None)


async def _flush_responses_worker():
    """Фоновая задача: периодически сбрасывает очередь ответов на диск."""
    while True:
        await asyncio.sleep(FLUSH_INTERVAL)
        await _flush_to_disk()


async def _flush_to_disk():
    """Записывает все накопленные ответы из очереди в responses.json."""
    if _responses_queue.empty():
        return

    batch = []
    while not _responses_queue.empty() and len(batch) < FLUSH_BATCH_SIZE:
        try:
            batch.append(_responses_queue.get_nowait())
        except asyncio.QueueEmpty:
            break

    if not batch:
        return

    async with _responses_lock:
        def _write():
            data = {"samples": []}
            try:
                with open(RESPONSES_FILE, "r", encoding="utf-8") as f:
                    data = ujson.load(f)
            except FileNotFoundError:
                pass
            data["samples"].extend(batch)
            with open(RESPONSES_FILE, "w", encoding="utf-8") as f:
                ujson.dump(data, f, ensure_ascii=False, indent=2)

        await asyncio.get_event_loop().run_in_executor(None, _write)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _questions_data, _options_map, _responses_queue, _responses_lock, _flush_task
    global _job_queue, _process_pool, _job_worker_task

    FastAPICache.init(InMemoryBackend())

    # Загружаем вопросы один раз при старте
    with open("questions.json", "r", encoding="utf-8") as f:
        _questions_data = ujson.load(f)

    # Строим options_map один раз
    _options_map = {}
    for q in _questions_data["questions"]:
        for opt in q["options"]:
            _options_map[(q["id"], opt["id"])] = opt["category"]

    # Инициализируем очередь и лок для записи ответов
    _responses_queue = asyncio.Queue()
    _responses_lock = asyncio.Lock()

    # Запускаем фоновый воркер записи
    _flush_task = asyncio.create_task(_flush_responses_worker())

    # Инициализируем очередь джобов и ProcessPoolExecutor
    _job_queue = asyncio.Queue()
    _process_pool = ProcessPoolExecutor(max_workers=1, initializer=_init_worker)
    _job_worker_task = asyncio.create_task(_process_jobs())

    yield

    # При завершении — останавливаем воркеры
    _job_worker_task.cancel()
    _process_pool.shutdown(wait=False)
    _flush_task.cancel()
    await _flush_to_disk()


app = FastAPI(title="ProfNavigator", lifespan=lifespan)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda req, exc: JSONResponse(
    {"detail": "Слишком много запросов. Попробуйте через 3 минуты."}, status_code=429
))
app.add_middleware(SlowAPIMiddleware)


class NoTransformMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-transform"
        return response

app.add_middleware(NoTransformMiddleware)

app.mount("/static", StaticFiles(directory="static"), name="static")


def load_questions() -> dict:
    """Возвращает кешированные вопросы (загружены при старте)."""
    return _questions_data


def _enqueue_response(category_counts: Dict, primary_label: str):
    """Добавляет семпл в очередь записи (не блокирует)."""
    sample = {
        "features": category_counts,
        "label": primary_label,
        "timestamp": datetime.now().isoformat()
    }
    try:
        _responses_queue.put_nowait(sample)
    except asyncio.QueueFull:
        pass  # При переполнении пропускаем запись


class Answer(BaseModel):
    question_id: int
    option_id: str


class SurveyRequest(BaseModel):
    answers: List[Answer]



@app.get("/questions")
async def get_questions(n: int = 15) -> dict:
    """
    Получить N случайных вопросов в случайном порядке.
    
    Args:
        n: количество вопросов для выбора (по умолчанию 15)
    """
    data = load_questions()
    questions = data["questions"].copy()

    # Перемешиваем вопросы
    random.shuffle(questions)

    # Выбираем N случайных вопросов
    num_questions = min(n, len(questions))
    selected_questions = questions[:num_questions]

    # Перемешиваем варианты ответов в каждом вопросе
    for q in selected_questions:
        options = q["options"].copy()
        random.shuffle(options)
        q["options"] = options

    return {
        "questions": selected_questions,
        "spheres": data["spheres"],
        "total_available": len(data["questions"]),
        "selected_count": len(selected_questions)
    }


@app.post("/submit")
@limiter.limit("1000/1minutes")
async def submit_survey(request: Request, request_body: SurveyRequest = Body(...)):
    """Отправить ответы — возвращает job_id для отслеживания через GET /job/{job_id}"""
    category_counts = {
        "analytical": 0, "social": 0, "creative": 0,
        "managerial": 0, "practical": 0, "research": 0,
        "technical": 0, "artistic": 0, "entrepreneurial": 0, "scientific": 0
    }

    for answer in request_body.answers:
        category = _options_map.get((answer.question_id, answer.option_id))
        if category:
            category_counts[category] += 1

    job_id = str(uuid.uuid4())
    job = Job(id=job_id, status="pending", category_counts=category_counts)
    _jobs[job_id] = job
    _pending_order.append(job_id)
    await _job_queue.put(job_id)

    return {"job_id": job_id, "position": len(_pending_order)}


@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Получить статус джоба: позицию в очереди или результат"""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Джоб не найден или уже удалён")

    if job.status == "pending":
        try:
            position = _pending_order.index(job_id) + 1
        except ValueError:
            position = 1
        return {"status": "pending", "position": position}

    if job.status == "processing":
        return {"status": "processing", "position": 0}

    if job.status == "done":
        return {"status": "done", "result": job.result}

    # error
    raise HTTPException(status_code=500, detail=job.error)


@app.get("/", response_class=HTMLResponse)
@limiter.limit("1000/second")
@cache(expire=3600)
async def root(request: Request):
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "queue_size": _job_queue.qsize() if _job_queue else 0}


# Админский эндпоинт для переобучения модели на основе накопленных данных(закоментирован, потому что в падлу авторизацию писать)
"""@app.post("/api/retrain")
async def retrain_model():

    import subprocess
    import sys

    if not Path(RESPONSES_FILE).exists():
        raise HTTPException(status_code=400, detail="Нет накопленных данных для обучения")

    # Копируем responses.json в dataset.json для trainer
    with open(RESPONSES_FILE, "r", encoding="utf-8") as f:
        responses_data = ujson.load(f)

    # Оставляем только samples
    dataset = {"samples": responses_data.get("samples", [])}

    with open("dataset.json", "w", encoding="utf-8") as f:
        ujson.dump(dataset, f, ensure_ascii=False, indent=2)
    
    # Запускаем trainer
    try:
        result = subprocess.run(
            [sys.executable, "trainer.py", "--model-type", "random_forest"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            # Перезагружаем модель
            global survey_model
            survey_model = SurveyModel()
            
            return {
                "status": "success",
                "message": "Модель переобучена",
                "samples_count": len(dataset["samples"]),
                "output": result.stdout[-500:]  # Последние 500 символов лога
            }
        else:
            raise HTTPException(status_code=500, detail=f"Ошибка обучения: {result.stderr}")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Таймаут обучения")


@app.post("/api/reload-model")
async def reload_model():
    
    global survey_model
    survey_model = SurveyModel()
    return {"status": "success", "model_fitted": survey_model.is_fitted}"""
