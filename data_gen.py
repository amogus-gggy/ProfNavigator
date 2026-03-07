import json
import random
import numpy as np
from collections import defaultdict

CATEGORIES = [
    "analytical", "social", "creative",
    "managerial", "practical", "research",
    "technical", "artistic", "entrepreneurial", "scientific"
]

N_QUESTIONS = 15

# Параметры Дирихле для каждого профиля.
# Высокий alpha у основной категории -> она доминирует, но не абсолютно.
# Смежные категории получают реалистичный "фон".
# Ожидаемая доля = alpha_i / sum(alphas), т.е. основная ~40%, остальные распределены.
CATEGORY_ALPHAS = {
    "analytical": {
        "analytical": 4.0, "research": 1.5, "technical": 1.2, "scientific": 1.2,
        "managerial": 0.5, "practical": 0.5, "creative": 0.4,
        "social": 0.3, "entrepreneurial": 0.2, "artistic": 0.2
    },
    "social": {
        "social": 4.0, "managerial": 1.5, "entrepreneurial": 1.2, "creative": 1.0,
        "artistic": 0.5, "practical": 0.5, "analytical": 0.4,
        "research": 0.3, "technical": 0.3, "scientific": 0.3
    },
    "creative": {
        "creative": 4.0, "artistic": 2.0, "social": 1.0, "entrepreneurial": 0.8,
        "research": 0.5, "managerial": 0.5, "analytical": 0.4,
        "practical": 0.3, "technical": 0.3, "scientific": 0.2
    },
    "managerial": {
        "managerial": 5.0, "social": 1.8, "analytical": 1.0, "practical": 0.7,
        "entrepreneurial": 0.5, "research": 0.3, "creative": 0.3,
        "technical": 0.2, "artistic": 0.1, "scientific": 0.1
    },
    "practical": {
        "practical": 4.0, "technical": 2.0, "managerial": 0.8, "analytical": 0.7,
        "scientific": 0.5, "research": 0.5, "entrepreneurial": 0.5,
        "creative": 0.4, "social": 0.3, "artistic": 0.3
    },
    "research": {
        "research": 4.0, "analytical": 1.5, "scientific": 1.5, "technical": 0.8,
        "creative": 0.5, "practical": 0.4, "managerial": 0.3,
        "social": 0.3, "artistic": 0.2, "entrepreneurial": 0.2
    },
    "technical": {
        "technical": 4.0, "practical": 1.8, "analytical": 1.2, "scientific": 0.8,
        "research": 0.5, "managerial": 0.5, "entrepreneurial": 0.4,
        "creative": 0.3, "social": 0.3, "artistic": 0.2
    },
    "artistic": {
        "artistic": 4.0, "creative": 2.5, "social": 0.8, "entrepreneurial": 0.6,
        "managerial": 0.4, "practical": 0.4, "analytical": 0.3,
        "research": 0.3, "technical": 0.2, "scientific": 0.1
    },
    "entrepreneurial": {
        "entrepreneurial": 5.0, "creative": 1.5, "analytical": 1.0, "practical": 0.8,
        "social": 0.6, "managerial": 0.4, "technical": 0.3,
        "research": 0.2, "artistic": 0.1, "scientific": 0.1
    },
    "scientific": {
        "scientific": 4.0, "research": 2.0, "analytical": 1.5, "technical": 0.8,
        "practical": 0.4, "creative": 0.3, "managerial": 0.2,
        "social": 0.2, "artistic": 0.1, "entrepreneurial": 0.1
    },
}


def generate_sample_for_category(category: str) -> dict:
    """
    Генерирует один семпл для заданной категории.
    Использует распределение Дирихле для реалистичного смешивания признаков.
    Сохраняет soft_labels (топ-3 категории с пропорциями) для weighted обучения.
    """
    alpha_dict = CATEGORY_ALPHAS[category]
    alpha_vec = np.array([alpha_dict[cat] for cat in CATEGORIES])

    # Сэмплируем пропорции из Дирихле
    proportions = np.random.dirichlet(alpha_vec)

    # Переводим в целые счётчики (сумма = N_QUESTIONS)
    raw_counts = proportions * N_QUESTIONS
    counts = np.floor(raw_counts).astype(int)
    remainder = N_QUESTIONS - counts.sum()
    if remainder > 0:
        fracs = raw_counts - counts
        top_indices = np.argsort(fracs)[::-1][:remainder]
        counts[top_indices] += 1

    features = {cat: int(counts[i]) for i, cat in enumerate(CATEGORIES)}

    # Топ-3 soft labels по пропорциям Дирихле (порог > 8%)
    props = {cat: float(proportions[i]) for i, cat in enumerate(CATEGORIES)}
    top3 = sorted(props.items(), key=lambda x: x[1], reverse=True)[:3]
    soft_labels = {cat: w for cat, w in top3 if w > 0.08}
    total_w = sum(soft_labels.values())
    soft_labels = {cat: round(w / total_w, 4) for cat, w in soft_labels.items()}

    # 1% шанс мислейбла
    if random.random() < 0.01:
        label = random.choice([c for c in CATEGORIES if c != category])
    else:
        label = category

    return {"features": features, "label": label, "soft_labels": soft_labels}


def generate_balanced_dataset(n_samples: int = 10000) -> dict:
    """
    Генерирует сбалансированный датасет: равное число семплов на каждый класс.
    """
    dataset = {"samples": []}
    samples_per_class = n_samples // len(CATEGORIES)

    for category in CATEGORIES:
        for _ in range(samples_per_class):
            dataset["samples"].append(generate_sample_for_category(category))

    # Добираем остаток случайными классами
    remaining = n_samples - len(dataset["samples"])
    for _ in range(remaining):
        cat = random.choice(CATEGORIES)
        dataset["samples"].append(generate_sample_for_category(cat))

    random.shuffle(dataset["samples"])
    return dataset


if __name__ == "__main__":
    print("Генерация датасета...")

    dataset = generate_balanced_dataset(n_samples=10000)

    label_counts = defaultdict(int)
    for sample in dataset["samples"]:
        label_counts[sample["label"]] += 1

    with open("dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Датасет сохранён в dataset.json")
    print(f"Всего семплов: {len(dataset['samples'])}")

    print("\nРаспределение классов:")
    for label in CATEGORIES:
        count = label_counts[label]
        pct = count / len(dataset["samples"]) * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:15}: {count:4} ({pct:4.1f}%) {bar}")

    print("\nПримеры семплов (первые 5):")
    for i, sample in enumerate(dataset["samples"][:5]):
        feats = sample["features"]
        top3 = sorted(feats.items(), key=lambda x: x[1], reverse=True)[:3]
        top3_str = ", ".join(f"{k}:{v}" for k, v in top3)
        print(f"  {i+1}. label={sample['label']:15} | top3: {top3_str}")

    # Показываем среднее распределение для одного профиля (sanity check)
    print("\nSanity check — среднее распределение для 'analytical' (из 100 семплов):")
    samples_check = [generate_sample_for_category("analytical") for _ in range(100)]
    avg = defaultdict(float)
    for s in samples_check:
        for cat, val in s["features"].items():
            avg[cat] += val / 100
    for cat in CATEGORIES:
        bar = "█" * int(avg[cat])
        print(f"  {cat:15}: {avg[cat]:4.1f} {bar}")
