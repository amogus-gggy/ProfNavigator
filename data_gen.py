import json
import random
import numpy as np
from collections import defaultdict

# Категории (10 штук)
CATEGORIES = [
    "analytical", "social", "creative",
    "managerial", "practical", "research",
    "technical", "artistic", "entrepreneurial", "scientific"
]

# Признаки (совпадают с категориями для упрощения)
FEATURES = CATEGORIES.copy()

# Матрица весов: для каждой категории определяем важность каждого признака (0-5)
# Более чёткие профили для лучшей разделимости
CATEGORY_PROFILES = {
    "analytical":    {"analytical": 5, "social": 0, "creative": 1, "managerial": 1, "practical": 1, "research": 3, "technical": 2, "artistic": 0, "entrepreneurial": 1, "scientific": 2},
    "social":        {"analytical": 1, "social": 5, "creative": 2, "managerial": 2, "practical": 1, "research": 1, "technical": 0, "artistic": 1, "entrepreneurial": 2, "scientific": 1},
    "creative":      {"analytical": 1, "social": 1, "creative": 5, "managerial": 1, "practical": 2, "research": 1, "technical": 1, "artistic": 4, "entrepreneurial": 2, "scientific": 1},
    "managerial":    {"analytical": 2, "social": 3, "creative": 1, "managerial": 5, "practical": 2, "research": 1, "technical": 1, "artistic": 0, "entrepreneurial": 3, "scientific": 1},
    "practical":     {"analytical": 1, "social": 1, "creative": 1, "managerial": 2, "practical": 5, "research": 1, "technical": 4, "artistic": 1, "entrepreneurial": 2, "scientific": 2},
    "research":      {"analytical": 3, "social": 0, "creative": 2, "managerial": 1, "practical": 1, "research": 5, "technical": 2, "artistic": 1, "entrepreneurial": 1, "scientific": 4},
    "technical":     {"analytical": 2, "social": 0, "creative": 1, "managerial": 1, "practical": 4, "research": 2, "technical": 5, "artistic": 0, "entrepreneurial": 2, "scientific": 3},
    "artistic":      {"analytical": 0, "social": 2, "creative": 4, "managerial": 1, "practical": 2, "research": 1, "technical": 0, "artistic": 5, "entrepreneurial": 1, "scientific": 1},
    "entrepreneurial": {"analytical": 2, "social": 3, "creative": 2, "managerial": 4, "practical": 2, "research": 1, "technical": 2, "artistic": 1, "entrepreneurial": 5, "scientific": 1},
    "scientific":    {"analytical": 3, "social": 0, "creative": 1, "managerial": 1, "practical": 2, "research": 4, "technical": 3, "artistic": 0, "entrepreneurial": 1, "scientific": 5}
}

def generate_sample():
    """Генерирует один семпл с перекрывающимися категориями"""

    # Сначала выбираем основную категорию (с небольшим перекосом для баланса)
    primary_category = random.choices(
        CATEGORIES,
        weights=[1.0, 0.9, 1.0, 0.8, 1.0, 0.9, 0.8, 0.8, 0.7, 0.9],  # небольшая регулировка баланса
        k=1
    )[0]

    # Определяем, сколько дополнительных категорий будет влиять на семпл (0-2)
    num_secondary = random.choices([0, 1, 2], weights=[0.4, 0.4, 0.2])[0]

    # Выбираем вторичные категории (исключая первичную)
    other_categories = [c for c in CATEGORIES if c != primary_category]
    secondary_categories = random.sample(other_categories, min(num_secondary, len(other_categories)))

    # Все категории, влияющие на семпл
    influencing_categories = [primary_category] + secondary_categories

    # Веса влияния каждой категории
    primary_weight = random.uniform(0.6, 1.0)  # основная категория важнее
    secondary_weights = [random.uniform(0.2, 0.5) for _ in secondary_categories]

    all_weights = [primary_weight] + secondary_weights

    # Генерируем признаки на основе взвешенной суммы профилей
    features = defaultdict(float)

    for cat, weight in zip(influencing_categories, all_weights):
        profile = CATEGORY_PROFILES[cat]
        for feature in FEATURES:
            # Добавляем шум для реалистичности (уменьшенный)
            noise = random.gauss(0, 0.15)
            value = profile.get(feature, 0) * weight + noise
            features[feature] += max(0, value)  # неотрицательные значения

    # Нормализуем до целых чисел 0-5 (с сохранением распределения)
    max_possible = sum(weight * 5 for weight in all_weights)  # максимальная сумма
    if max_possible > 0:
        scale = 5.0 / max_possible
        for feature in FEATURES:
            features[feature] = min(5, max(0, round(features[feature] * scale)))
    else:
        # fallback - равномерное распределение
        for feature in FEATURES:
            features[feature] = random.randint(0, 5)

    # Преобразуем в целые числа
    features = {k: int(v) for k, v in features.items()}

    # Определяем лейбл (основная категория, но с 3% шансом может быть другой)
    if random.random() < 0.03:  # 3% шум
        return {
            "features": features,
            "label": random.choice(CATEGORIES)
        }
    else:
        return {
            "features": features,
            "label": primary_category
        }

# Генерируем датасет
dataset = {"samples": []}
label_counts = defaultdict(int)

for i in range(3000):
    sample = generate_sample()
    dataset["samples"].append(sample)
    label_counts[sample["label"]] += 1

    # Прогресс
    if (i + 1) % 500 == 0:
        print(f"Generated {i + 1} samples")

# Сохраняем в файл
with open("dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

# Статистика
print("\n" + "="*50)
print("DATASET STATISTICS")
print("="*50)
print(f"Total samples: {len(dataset['samples'])}")
print("\nLabel distribution:")
for label in CATEGORIES:
    count = label_counts[label]
    percentage = (count / 3000) * 100
    print(f"  {label:15}: {count:4} ({percentage:5.1f}%)")

# Проверка корреляции признаков
print("\nFeature correlation with labels (sample):")
feature_importance = defaultdict(lambda: defaultdict(float))
for sample in dataset["samples"][:100]:  # первые 100 для примера
    label = sample["label"]
    for feature, value in sample["features"].items():
        feature_importance[feature][label] += value

print("\nAverage feature values by label (first 3 features):")
for feature in list(FEATURES)[:3]:
    print(f"\n{feature}:")
    for label in CATEGORIES[:5]:  # первые 5 категорий
        avg = feature_importance[feature][label] / 100
        print(f"  {label:12}: {avg:.2f}")
