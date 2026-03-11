from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
import ujson
import random
import uuid
import time
from dataclasses import dataclass
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


# ── ProcessPoolExecutor worker ────────────────────────────────────────────────

_worker_model: Optional["SurveyModel"] = None


def _init_worker():
    """Initialize worker process by loading the model once at startup."""
    global _worker_model
    from model import SurveyModel
    _worker_model = SurveyModel()


def _predict_in_worker(category_counts: dict) -> dict:
    """Run CPU-bound prediction in a separate process."""
    return _worker_model.predict(category_counts)


# ── Job queue ─────────────────────────────────────────────────────────────────

JOB_TTL = 300  # seconds before completed jobs are removed


@dataclass
class Job:
    id: str
    status: str  # pending | processing | done | error
    category_counts: dict
    result: Optional[dict] = None
    error: Optional[str] = None
    finished_at: Optional[float] = None


_jobs: Dict[str, Job] = {}
_pending_order: List[str] = []
_job_queue: asyncio.Queue = None
_process_pool: ProcessPoolExecutor = None
_job_worker_task: asyncio.Task = None


def get_cloudflare_ip(request: Request) -> str:
    """Extract client IP from Cloudflare header or fallback to direct connection."""
    return request.headers.get("cf-connecting-ip") or (
        request.client.host if request.client else "127.0.0.1"
    )


limiter = Limiter(key_func=get_cloudflare_ip)

_questions_data: dict = {}
_options_map: dict = {}


async def _process_jobs():
    """Background worker that processes jobs from the queue one at a time."""
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
        except Exception as e:
            job.status = "error"
            job.error = str(e)
        finally:
            job.finished_at = time.time()
            _job_queue.task_done()
            _cleanup_expired_jobs()


def _cleanup_expired_jobs():
    """Remove completed jobs older than JOB_TTL seconds."""
    now = time.time()
    expired = [
        jid for jid, j in _jobs.items()
        if j.finished_at and (now - j.finished_at) > JOB_TTL
    ]
    for jid in expired:
        _jobs.pop(jid, None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager: initialize caches, pools, and workers."""
    global _questions_data, _options_map
    global _job_queue, _process_pool, _job_worker_task

    FastAPICache.init(InMemoryBackend())

    with open("questions.json", "r", encoding="utf-8") as f:
        _questions_data = ujson.load(f)

    _options_map = {}
    for q in _questions_data["questions"]:
        for opt in q["options"]:
            if "categories" in opt:
                _options_map[(q["id"], opt["id"])] = opt["categories"]
            elif "category" in opt:
                _options_map[(q["id"], opt["id"])] = {opt["category"]: 1.0}
            else:
                _options_map[(q["id"], opt["id"])] = {}

    _job_queue = asyncio.Queue()
    _process_pool = ProcessPoolExecutor(max_workers=1, initializer=_init_worker)
    _job_worker_task = asyncio.create_task(_process_jobs())

    yield

    _job_worker_task.cancel()
    _process_pool.shutdown(wait=False)


app = FastAPI(title="ProfNavigator", lifespan=lifespan)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda req, exc: JSONResponse(
    {"detail": "Too many requests. Please try again later."}, status_code=429
))
app.add_middleware(SlowAPIMiddleware)


class NoTransformMiddleware(BaseHTTPMiddleware):
    """Middleware to prevent automatic content transformation by proxies."""
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-transform"
        return response


app.add_middleware(NoTransformMiddleware)

app.mount("/static", StaticFiles(directory="static"), name="static")


def load_questions() -> dict:
    """Return cached questions loaded at startup."""
    return _questions_data


def _calculate_category_scores(questions: list) -> Dict[str, float]:
    """
    Calculate total category scores across all options in the given questions.

    Sums up weights for each category from all options of all questions.
    """
    category_totals = {
        "analytical": 0.0, "social": 0.0, "creative": 0.0,
        "managerial": 0.0, "practical": 0.0, "research": 0.0,
        "technical": 0.0, "artistic": 0.0, "entrepreneurial": 0.0, "scientific": 0.0
    }

    for q in questions:
        for opt in q["options"]:
            cats = opt.get("categories", {})
            for cat, weight in cats.items():
                if cat in category_totals:
                    category_totals[cat] += weight

    return category_totals


def _check_imbalance(
    category_scores: Dict[str, float],
    threshold: float = 1.5
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Check for category imbalance based on score ratio.

    Returns:
        Tuple of (has_imbalance, weak_category, strong_category).
        Imbalance exists when max_score / min_score > threshold.
    """
    active_scores = {cat: score for cat, score in category_scores.items() if score > 0}

    if len(active_scores) < 2:
        return False, None, None

    min_cat = min(active_scores, key=active_scores.get)
    max_cat = max(active_scores, key=active_scores.get)

    min_score = active_scores[min_cat]
    max_score = active_scores[max_cat]

    if min_score == 0:
        return True, min_cat, max_cat

    ratio = max_score / min_score
    if ratio > threshold:
        return True, min_cat, max_cat

    return False, None, None


def _find_replacement_question(
    selected_questions: list,
    all_questions: list,
    weak_category: str,
    strong_category: str
) -> Optional[dict]:
    """
    Find a question to replace that will strengthen weak category and reduce strong one.

    Returns:
        A question from all_questions (not in selected_questions) that has
        higher contribution to weak_category than to strong_category.
    """
    selected_ids = {q["id"] for q in selected_questions}
    candidates = []

    for q in all_questions:
        if q["id"] in selected_ids:
            continue

        cat_scores = {
            "analytical": 0.0, "social": 0.0, "creative": 0.0,
            "managerial": 0.0, "practical": 0.0, "research": 0.0,
            "technical": 0.0, "artistic": 0.0, "entrepreneurial": 0.0, "scientific": 0.0
        }

        for opt in q["options"]:
            cats = opt.get("categories", {})
            for cat, weight in cats.items():
                if cat in cat_scores:
                    cat_scores[cat] += weight

        weak_score = cat_scores.get(weak_category, 0)
        strong_score = cat_scores.get(strong_category, 0)
        usefulness = weak_score - strong_score * 0.5

        if usefulness > 0:
            candidates.append((q, usefulness, cat_scores))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def _balance_questions(
    selected_questions: list,
    all_questions: list,
    threshold: float = 1.5,
    max_iterations: int = 10
) -> list:
    """
    Balance question selection by iteratively replacing questions.

    Replaces questions that contribute heavily to dominant categories
    with questions that strengthen weaker categories until the ratio
    between max and min category scores is within threshold.
    """
    questions = selected_questions.copy()

    for _ in range(max_iterations):
        category_scores = _calculate_category_scores(questions)
        has_imbalance, weak_cat, strong_cat = _check_imbalance(category_scores, threshold)

        if not has_imbalance:
            break

        replacement = _find_replacement_question(questions, all_questions, weak_cat, strong_cat)

        if replacement is None:
            break

        question_to_remove = None
        max_contribution = 0

        for q in questions:
            strong_contrib = 0
            weak_contrib = 0
            for opt in q["options"]:
                cats = opt.get("categories", {})
                strong_contrib += cats.get(strong_cat, 0)
                weak_contrib += cats.get(weak_cat, 0)

            contribution = strong_contrib - weak_contrib
            if contribution > max_contribution:
                max_contribution = contribution
                question_to_remove = q

        if question_to_remove is None:
            question_to_remove = random.choice(questions)

        questions.remove(question_to_remove)
        questions.append(replacement)

    return questions


class Answer(BaseModel):
    question_id: int
    option_id: str


class SurveyRequest(BaseModel):
    answers: List[Answer]


@app.get("/questions")
async def get_questions(n: int = 30) -> dict:
    """
    Get N random questions with balanced category distribution.

    Args:
        n: Number of questions to return (default: 30)

    Returns:
        Dictionary with shuffled questions, shuffled options, and sphere metadata.
        Questions are balanced so no category has >1.5x more options than another.
    """
    data = load_questions()
    questions = data["questions"].copy()

    random.shuffle(questions)

    num_questions = min(n, len(questions))
    selected_questions = questions[:num_questions]

    selected_questions = _balance_questions(
        selected_questions,
        questions,
        threshold=1.2,
        max_iterations=10
    )

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
    """
    Submit survey answers and queue for processing.

    Returns:
        job_id for tracking progress via GET /job/{job_id}
    """
    category_counts = {
        "analytical": 0, "social": 0, "creative": 0,
        "managerial": 0, "practical": 0, "research": 0,
        "technical": 0, "artistic": 0, "entrepreneurial": 0, "scientific": 0
    }

    for answer in request_body.answers:
        cat_weights = _options_map.get((answer.question_id, answer.option_id), {})
        for cat, weight in cat_weights.items():
            if cat in category_counts:
                category_counts[cat] += weight

    job_id = str(uuid.uuid4())
    job = Job(id=job_id, status="pending", category_counts=category_counts)
    _jobs[job_id] = job
    _pending_order.append(job_id)
    await _job_queue.put(job_id)

    return {"job_id": job_id, "position": len(_pending_order)}


@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Get job status: position in queue or prediction result."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found or already removed")

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

    raise HTTPException(status_code=500, detail=job.error)


@app.get("/", response_class=HTMLResponse)
@limiter.limit("1000/second")
@cache(expire=3600)
async def root(request: Request):
    """Serve the main HTML page."""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/health")
async def health_check():
    """Health check endpoint with queue status."""
    return {"status": "ok", "queue_size": _job_queue.qsize() if _job_queue else 0}
