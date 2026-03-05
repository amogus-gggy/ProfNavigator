# Career Survey API

Приложение для профориентации с ML-классификацией результатов.

## Структура проекта

```
project/
├── main.py              # FastAPI сервер
├── model.py             # ML модель (загрузка и предсказание)
├── trainer.py           # Скрипт для обучения модели
├── questions.json       # Вопросы опроса (6 категорий)
├── dataset.json         # Датасет для обучения
├── responses.json       # Накопленные ответы пользователей
├── model_artifact.pkl   # Обученная модель
├── requirements.txt     # Зависимости
└── static/
    └── index.html       # Веб-интерфейс
```

## Категории

1. **analytical** — IT и аналитика 💻
2. **social** — Коммуникации и сервис 🤝
3. **creative** — Творчество и дизайн 🎨
4. **managerial** — Менеджмент и управление 📊
5. **practical** — Производство и технологии 🔧
6. **research** — Наука и исследования 🔬

## Запуск сервера

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Откройте http://127.0.0.1:8000

## Обучение модели

### 1. Быстрое обучение (на готовом датасете)

```bash
python trainer.py --model-type random_forest
```

### 2. Обучение на накопленных данных

После того как пользователи прошли опрос:

```bash
# POST запрос к API для переобучения
curl -X POST http://127.0.0.1:8000/api/retrain
```

Или вручную:
```bash
# responses.json копируется в dataset.json
python trainer.py --model-type random_forest
```

### 3. Доступные типы моделей

- `decision_tree` — дерево решений
- `random_forest` — случайный лес (по умолчанию)
- `neural_network` — нейронная сеть (MLP)

```bash
python trainer.py --model-type neural_network
```

## Формат датасета

`dataset.json`:

```json
{
  "samples": [
    {
      "features": {
        "analytical": 5,
        "social": 1,
        "creative": 1,
        "managerial": 1,
        "practical": 1,
        "research": 1
      },
      "label": "analytical"
    }
  ]
}
```

Где:
- `features` — количество баллов по каждой категории (сумма = количеству вопросов)
- `label` — целевая категория для обучения

## API Endpoints

| Метод | Endpoint | Описание |
|-------|----------|----------|
| GET | `/` | Главная страница |
| GET | `/questions` | Получить случайные вопросы |
| POST | `/submit` | Отправить ответы, получить результат |
| GET | `/api/health` | Проверка статуса модели |
| POST | `/api/retrain` | Переобучить модель на накопленных данных |

## Как это работает

1. Пользователь проходит опрос (5 случайных вопросов из 12)
2. Ответы подсчитываются по 6 категориям
3. ML модель предсказывает наиболее подходящую сферу
4. Результат с вероятностями показывается пользователю
5. Ответы сохраняются в `responses.json` для будущего переобучения
