"""
Trainer для ML модели профориентации.

Использование:
    python trainer.py [--dataset dataset.json] [--output model.pkl] [--epochs 100]

Формат dataset.json:
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
    },
    ...
  ]
}
"""

import json
import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score


CATEGORIES = [
    "analytical", "social", "creative",
    "managerial", "practical", "research",
    "technical", "artistic", "entrepreneurial", "scientific"
]


def load_dataset(path: str) -> tuple:
    """Загрузка датасета и преобразование в numpy массивы"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    X = []
    y = []
    
    for sample in data["samples"]:
        features = [
            sample["features"].get(cat, 0)
            for cat in CATEGORIES
        ]
        X.append(features)
        y.append(sample["label"])
    
    return np.array(X), np.array(y)


def train_model(X: np.ndarray, y: np.ndarray, model_type: str = "random_forest"):
    """Обучение модели"""
    # Разделение на train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Выбор модели
    if model_type == "decision_tree":
        model = DecisionTreeClassifier(
            max_depth=8,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42
        )
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            class_weight='balanced'
        )
    elif model_type == "neural_network":
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            max_iter=2000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            learning_rate='adaptive',
            learning_rate_init=0.001
        )
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")
    
    # Обучение
    print(f"Обучение модели: {model_type}")
    print(f"Размер train: {len(X_train)}, validation: {len(X_val)}")
    
    model.fit(X_train, y_train)
    
    # Валидация
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    print(f"\nТочность на validation: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_val, y_pred, zero_division=0))
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"\nCross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return model


def save_model(model, label_encoder: LabelEncoder, output_path: str):
    """Сохранение модели"""
    artifact = {
        "model": model,
        "label_encoder": label_encoder,
        "categories": CATEGORIES
    }
    
    with open(output_path, "wb") as f:
        pickle.dump(artifact, f)
    
    print(f"\nМодель сохранена в {output_path}")
    print(f"Классы в label_encoder: {list(label_encoder.classes_)}")


def main():
    parser = argparse.ArgumentParser(description="Trainer для модели профориентации")
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="dataset.json",
        help="Путь к файлу с датасетом"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="model_artifact.pkl",
        help="Путь для сохранения модели"
    )
    parser.add_argument(
        "--model-type", 
        type=str, 
        default="random_forest",
        choices=["decision_tree", "random_forest", "neural_network"],
        help="Тип модели для обучения"
    )
    
    args = parser.parse_args()
    
    # Проверка существования датасета
    if not Path(args.dataset).exists():
        print(f"Ошибка: файл {args.dataset} не найден")
        return 1
    
    # Загрузка данных
    print(f"Загрузка датасета из {args.dataset}")
    X, y = load_dataset(args.dataset)
    print(f"Загружено {len(X)} примеров")
    print(f"Классы в датасете: {np.unique(y)}")

    # Кодирование лейблов - фитим на ВСЕХ возможных категориях, а не только на тех, что в датасете
    label_encoder = LabelEncoder()
    label_encoder.fit(CATEGORIES)  # Фитим на всех категориях, чтобы порядок был фиксированным

    # Преобразуем y с помощью label_encoder
    y_encoded = label_encoder.transform(y)

    # Обучение
    model = train_model(X, y_encoded, args.model_type)
    
    # Сохранение
    save_model(model, label_encoder, args.output)
    
    print("\n✅ Обучение завершено!")
    return 0


if __name__ == "__main__":
    exit(main())