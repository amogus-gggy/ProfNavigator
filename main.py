from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, UJSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import ujson
import random
from datetime import datetime
from model import SurveyModel

app = FastAPI(title="ProfNavigator", default_response_class=UJSONResponse)


class NoTransformMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-transform"
        return response

app.add_middleware(NoTransformMiddleware)

app.mount("/static", StaticFiles(directory="static"), name="static")


def load_questions() -> dict:
    with open("questions.json", "r", encoding="utf-8") as f:
        return ujson.load(f)

#ML init
survey_model = SurveyModel()


RESPONSES_FILE = "responses.json"


def save_response(answers: List[Dict], result: Dict):
    """Сохранение ответа пользователя для будущего переобучения"""
    data = {"samples": []}

    try:
        with open(RESPONSES_FILE, "r", encoding="utf-8") as f:
            data = ujson.load(f)
    except FileNotFoundError:
        pass

    # Подсчёт категорий
    category_counts = {
        "analytical": 0, "social": 0, "creative": 0,
        "managerial": 0, "practical": 0, "research": 0,
        "technical": 0, "artistic": 0, "entrepreneurial": 0, "scientific": 0
    }

    data["questions"] = load_questions()["questions"]
    options_map = {}
    for q in data["questions"]:
        for opt in q["options"]:
            options_map[(q["id"], opt["id"])] = opt["category"]

    for answer in answers:
        category = options_map.get((answer["question_id"], answer["option_id"]))
        if category and category in category_counts:
            category_counts[category] += 1

    # Добавление нового семпла
    data["samples"].append({
        "features": category_counts,
        "label": result["primary"],
        "timestamp": datetime.now().isoformat()
    })

    with open(RESPONSES_FILE, "w", encoding="utf-8") as f:
        ujson.dump(data, f, ensure_ascii=False, indent=2)


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
async def submit_survey(request: SurveyRequest) -> SurveyResponse:
    """Отправить ответы и получить результат на основе ML модели"""
    data = load_questions()

    # Подсчёт ответов по категориям
    category_counts = {
        "analytical": 0,
        "social": 0,
        "creative": 0,
        "managerial": 0,
        "practical": 0,
        "research": 0,
        "technical": 0,
        "artistic": 0,
        "entrepreneurial": 0,
        "scientific": 0
    }

    # Построение карты question_id -> options
    options_map = {}
    for q in data["questions"]:
        for opt in q["options"]:
            options_map[(q["id"], opt["id"])] = opt["category"]

    # Подсчёт категорий
    for answer in request.answers:
        category = options_map.get((answer.question_id, answer.option_id))
        if category:
            category_counts[category] += 1

    
    prediction = survey_model.predict(category_counts)

    
    sphere_info = data["spheres"][prediction["primary"]]

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

    # Сохранение ответа для будущего переобучения
    save_response(
        [a.dict() for a in request.answers],
        {"primary": prediction["primary"]}
    )

    return result


@app.get("/")
async def root():
    return FileResponse("static/index.html")


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
