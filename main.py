from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import ujson
import random
from datetime import datetime
import asyncio
from model import SurveyModel
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache
from contextlib import asynccontextmanager


def get_cloudflare_ip(request: Request) -> str:
    return request.headers.get("cf-connecting-ip") or (
        request.client.host if request.client else "127.0.0.1"
    )


limiter = Limiter(key_func=get_cloudflare_ip)

semaphore = asyncio.Semaphore(150)

# Глобальный кеш вопросов и options_map (инициализируется при старте)
_questions_data: dict = {}
_options_map: dict = {}

# Очередь и лок для батч-записи ответов
_responses_queue: asyncio.Queue = None
_responses_lock: asyncio.Lock = None
_flush_task: asyncio.Task = None

FLUSH_INTERVAL = 5  # секунд между записями на диск
FLUSH_BATCH_SIZE = 50  # максимум записей за раз


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
        data = {"samples": []}
        try:
            with open(RESPONSES_FILE, "r", encoding="utf-8") as f:
                data = ujson.load(f)
        except FileNotFoundError:
            pass

        data["samples"].extend(batch)

        with open(RESPONSES_FILE, "w", encoding="utf-8") as f:
            ujson.dump(data, f, ensure_ascii=False, indent=2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _questions_data, _options_map, _responses_queue, _responses_lock, _flush_task

    FastAPICache.init(InMemoryBackend())

    # Загружаем вопросы один раз при старте
    with open("questions.json", "r", encoding="utf-8") as f:
        _questions_data = ujson.load(f)

    # Строим options_map один раз
    _options_map = {}
    for q in _questions_data["questions"]:
        for opt in q["options"]:
            _options_map[(q["id"], opt["id"])] = opt["category"]

    # Инициализируем очередь и лок
    _responses_queue = asyncio.Queue()
    _responses_lock = asyncio.Lock()

    # Запускаем фоновый воркер записи
    _flush_task = asyncio.create_task(_flush_responses_worker())

    yield

    # При завершении — сбрасываем остаток на диск
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

#ML init
survey_model = SurveyModel()


RESPONSES_FILE = "responses.json"


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


class SurveyResponse(BaseModel):
    primary: str  # category key (analytical, social, etc.)
    sphere: str  # название сферы (IT и аналитика)
    description: str
    confidence: str  # high / medium / low
    probabilities: Dict[str, float]
    ranking: List[tuple]
    answer_distribution: Dict[str, float]  # нормализованное распределение ответов
    reasoning: List[str]  # текстовое объяснение результата


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


@app.post("/submit", response_model=SurveyResponse)
@limiter.limit("1000/1minutes")
async def submit_survey(request: Request, request_body: SurveyRequest = Body(...)) -> SurveyResponse:
    """Отправить ответы и получить результат на основе ML модели"""
    async with semaphore:
        # Подсчёт ответов по категориям через кешированный options_map
        category_counts = {
            "analytical": 0, "social": 0, "creative": 0,
            "managerial": 0, "practical": 0, "research": 0,
            "technical": 0, "artistic": 0, "entrepreneurial": 0, "scientific": 0
        }

        for answer in request_body.answers:
            category = _options_map.get((answer.question_id, answer.option_id))
            if category:
                category_counts[category] += 1

        prediction = survey_model.predict(category_counts)

        sphere_info = _questions_data["spheres"][prediction["primary"]]

        result = SurveyResponse(
            primary=prediction["primary"],
            sphere=sphere_info["name"],
            description=sphere_info["description"],
            confidence=prediction["confidence"],
            probabilities=prediction["probabilities"],
            ranking=prediction["ranking"],
            answer_distribution=prediction["answer_distribution"],
            reasoning=prediction["reasoning"],
        )

        # Неблокирующая постановка в очередь записи
        _enqueue_response(category_counts, prediction["primary"])

        return result


@app.get("/", response_class=HTMLResponse)
@limiter.limit("1000/second")
@cache(expire=3600)
async def root(request: Request):
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "model_fitted": survey_model.is_fitted}


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
