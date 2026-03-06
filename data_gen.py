

import json
import random
import numpy as np
from collections import defaultdict


CATEGORIES = [
    "analytical", "social", "creative",
    "managerial", "practical", "research",
    "technical", "artistic", "entrepreneurial", "scientific"
]

FEATURES = CATEGORIES.copy()


CATEGORY_PROFILES = {
    "analytical":      {"analytical": 14, "social": 0, "creative": 0, "managerial": 0, "practical": 0, "research": 1, "technical": 0, "artistic": 0, "entrepreneurial": 0, "scientific": 0},
    "social":          {"analytical": 0, "social": 14, "creative": 0, "managerial": 0, "practical": 0, "research": 0, "technical": 0, "artistic": 0, "entrepreneurial": 0, "scientific": 1},
    "creative":        {"analytical": 0, "social": 0, "creative": 14, "managerial": 0, "practical": 0, "research": 0, "technical": 0, "artistic": 1, "entrepreneurial": 0, "scientific": 0},
    "managerial":      {"analytical": 0, "social": 0, "creative": 0, "managerial": 14, "practical": 0, "research": 0, "technical": 0, "artistic": 0, "entrepreneurial": 1, "scientific": 0},
    "practical":       {"analytical": 0, "social": 0, "creative": 0, "managerial": 0, "practical": 14, "research": 0, "technical": 1, "artistic": 0, "entrepreneurial": 0, "scientific": 0},
    "research":        {"analytical": 1, "social": 0, "creative": 0, "managerial": 0, "practical": 0, "research": 14, "technical": 0, "artistic": 0, "entrepreneurial": 0, "scientific": 0},
    "technical":       {"analytical": 0, "social": 0, "creative": 0, "managerial": 0, "practical": 1, "research": 0, "technical": 14, "artistic": 0, "entrepreneurial": 0, "scientific": 0},
    "artistic":        {"analytical": 0, "social": 0, "creative": 1, "managerial": 0, "practical": 0, "research": 0, "technical": 0, "artistic": 14, "entrepreneurial": 0, "scientific": 0},
    "entrepreneurial": {"analytical": 0, "social": 0, "creative": 0, "managerial": 1, "practical": 0, "research": 0, "technical": 0, "artistic": 0, "entrepreneurial": 14, "scientific": 0},
    "scientific":      {"analytical": 0, "social": 0, "creative": 0, "managerial": 0, "practical": 0, "research": 1, "technical": 0, "artistic": 0, "entrepreneurial": 0, "scientific": 14}
}


def generate_sample(noise_level: float = 0.05):
    
    # Основная категория (равномерное распределение)
    primary_category = random.choice(CATEGORIES)

    # 0-1 дополнительных категории (редко) для смешанных профилей
    num_secondary = random.choices([0, 1], weights=[0.85, 0.15])[0]

    other_categories = [c for c in CATEGORIES if c != primary_category]
    secondary_categories = random.sample(other_categories, min(num_secondary, len(other_categories)))

    influencing_categories = [primary_category] + secondary_categories

    # Веса: основная категория доминирует сильнее
    primary_weight = random.uniform(0.85, 1.0)
    secondary_weights = [random.uniform(0.1, 0.25) for _ in secondary_categories]
    all_weights = [primary_weight] + secondary_weights

    # Генерируем признаки на основе взвешенной суммы профилей
    features = defaultdict(float)

    for cat, weight in zip(influencing_categories, all_weights):
        profile = CATEGORY_PROFILES[cat]
        for feature in FEATURES:
            # УМЕНЬШЕННЫЙ шум
            noise = random.gauss(0, noise_level * 2)
            value = profile.get(feature, 0) * weight + noise
            features[feature] += max(0, value)

    # Нормализуем до 0-15 (общая сумма ~15, как количество вопросов)
    total = sum(features.values())
    if total > 0:
        scale = 15.0 / total
        for feature in FEATURES:
            features[feature] = min(15, max(0, round(features[feature] * scale)))
    else:
        for feature in FEATURES:
            features[feature] = random.randint(0, 2)

    # Гарантируем, что сумма = 15 (корректировка)
    current_sum = sum(features.values())
    if current_sum != 15:
        diff = 15 - current_sum
        # Распределяем разницу по признакам основной категории
        primary_feat = primary_category
        features[primary_feat] = max(0, min(15, features[primary_feat] + diff))

    features = {k: int(v) for k, v in features.items()}

    # УМЕНЬШЕННЫЙ шанс неправильного лейбла (1% вместо 3%)
    if random.random() < 0.01:
        return {
            "features": features,
            "label": random.choice(CATEGORIES)
        }
    else:
        return {
            "features": features,
            "label": primary_category
        }


def generate_balanced_dataset(n_samples: int = 10000, noise_level: float = 0.05):
    """
    Генерирует сбалансированный датасет.
    УВЕЛИЧЕН размер датасета для лучшего обучения.
    
    Args:
        n_samples: общее количество семплов
        noise_level: уровень шума (0.0-1.0) - УМЕНЬШЕН
    """
    dataset = {"samples": []}
    samples_per_class = n_samples // len(CATEGORIES)
    
    for category in CATEGORIES:
        for _ in range(samples_per_class):
            sample = generate_sample_for_category(category, noise_level)
            dataset["samples"].append(sample)
    
    # Добавляем смешанные профили
    n_mixed = n_samples - len(dataset["samples"])
    for _ in range(n_mixed):
        dataset["samples"].append(generate_sample(noise_level))
    
    # Перемешиваем
    random.shuffle(dataset["samples"])
    
    return dataset


def generate_sample_for_category(category: str, noise_level: float = 0.05):
    """Генерирует семпл для конкретной основной категории с минимальным шумом"""
    # Редко добавляем вторичную категорию
    num_secondary = random.choices([0, 1], weights=[0.85, 0.15])[0]
    
    other_categories = [c for c in CATEGORIES if c != category]
    secondary_categories = random.sample(other_categories, min(num_secondary, len(other_categories)))
    
    influencing_categories = [category] + secondary_categories
    primary_weight = random.uniform(0.85, 1.0)
    secondary_weights = [random.uniform(0.1, 0.25) for _ in secondary_categories]
    all_weights = [primary_weight] + secondary_weights
    
    features = defaultdict(float)
    
    for cat, weight in zip(influencing_categories, all_weights):
        profile = CATEGORY_PROFILES[cat]
        for feature in FEATURES:
            noise = random.gauss(0, noise_level * 2)
            value = profile.get(feature, 0) * weight + noise
            features[feature] += max(0, value)
    
    # Нормализуем до 0-15
    total = sum(features.values())
    if total > 0:
        scale = 15.0 / total
        for feature in FEATURES:
            features[feature] = min(15, max(0, round(features[feature] * scale)))
    
    features = {k: int(v) for k, v in features.items()}
    
    # Корректировка суммы до 15
    current_sum = sum(features.values())
    if current_sum != 15:
        diff = 15 - current_sum
        features[category] = max(0, min(15, features[category] + diff))
    
    # 1% шанс неправильного лейбла
    if random.random() < 0.01:
        label = random.choice([c for c in CATEGORIES if c != category])
    else:
        label = category
    
    return {
        "features": features,
        "label": label
    }


# === Основной запуск ===
if __name__ == "__main__":
    print("🔧 Генерация датасета для 15 вопросов (улучшенная версия)...")
    print("=" * 50)
    
    # Генерируем УВЕЛИЧЕННЫЙ сбалансированный датасет с МЕНЬШИМ шумом
    dataset = generate_balanced_dataset(n_samples=10000, noise_level=0.05)
    
    # Статистика
    label_counts = defaultdict(int)
    for sample in dataset["samples"]:
        label_counts[sample["label"]] += 1
    
    # Сохранение
    with open("dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Датасет сохранён в dataset.json")
    print(f"   Всего семплов: {len(dataset['samples'])}")
    
    print("\n📊 Распределение классов:")
    for label in CATEGORIES:
        count = label_counts[label]
        percentage = (count / len(dataset["samples"])) * 100
        bar = "█" * int(percentage / 2)
        print(f"   {label:15}: {count:4} ({percentage:5.1f}%) {bar}")
    
    # Примеры семплов
    print("\n📋 Примеры семплов (первые 5):")
    for i, sample in enumerate(dataset["samples"][:5]):
        features = sample["features"]
        top3 = sorted(features.items(), key=lambda x: x[1], reverse=True)[:3]
        top3_str = ", ".join([f"{k}:{v}" for k, v in top3])
        print(f"   {i+1}. label={sample['label']:12} | {top3_str}")