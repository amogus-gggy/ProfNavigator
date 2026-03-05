"""
ML модель для классификации результатов опроса.
Загружает обученную модель из model_artifact.pkl
"""

import pickle
from pathlib import Path
import numpy as np


class SurveyModel:
    """ML модель для классификации результатов опроса"""
    
    DEFAULT_CATEGORIES = [
        "analytical", "social", "creative",
        "managerial", "practical", "research",
        "technical", "artistic", "entrepreneurial", "scientific"
    ]
    
    def __init__(self, model_path: str = "model_artifact.pkl"):
        self.model_path = model_path
        self.model = None
        self.label_encoder = None
        self.categories = self.DEFAULT_CATEGORIES.copy()
        self.is_fitted = False
        
        self._load_model()
    
    def _load_model(self):
        """Загрузка обученной модели из файла"""
        path = Path(self.model_path)
        
        if path.exists():
            try:
                with open(path, "rb") as f:
                    artifact = pickle.load(f)
                
                self.model = artifact["model"]
                self.label_encoder = artifact["label_encoder"]
                self.categories = artifact.get("categories", self.DEFAULT_CATEGORIES)
                self.is_fitted = True
                print(f"✅ Модель загружена из {self.model_path}")
                print(f"   Классы модели: {list(self.label_encoder.classes_)}")
            except Exception as e:
                print(f"⚠️ Ошибка загрузки модели: {e}")
                self._init_default_model()
        else:
            print(f"⚠️ Модель не найдена в {self.model_path}, использую default модель")
            self._init_default_model()
    
    def _init_default_model(self):
        """Инициализация простой модели для fallback"""
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.preprocessing import LabelEncoder
        
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.categories)
        
        # Простая синтетическая тренировка
        X_train = []
        y_train = []
        
        # Чистые типы
        for i, cat in enumerate(self.categories):
            for _ in range(5):
                features = [1] * len(self.categories)
                features[i] = 5
                X_train.append(features)
                y_train.append(cat)
        
        # Смешанные типы
        for i, cat1 in enumerate(self.categories):
            for j, cat2 in enumerate(self.categories):
                if i < j:
                    features = [1] * len(self.categories)
                    features[i] = 4
                    features[j] = 3
                    X_train.append(features)
                    y_train.append(cat1)
        
        self.model = DecisionTreeClassifier(random_state=42)
        self.model.fit(X_train, self.label_encoder.transform(y_train))
        self.is_fitted = True
    
    def predict(self, answers: dict) -> dict:
        """
        Предсказание результата на основе ответов

        Args:
            answers: dict {category: count} - количество ответов по категориям

        Returns:
            dict с результатом предсказания
        """
        # Вектор признаков в порядке классов label_encoder (а не DEFAULT_CATEGORIES!)
        X = np.array([[
            answers.get(cat, 0) for cat in self.label_encoder.classes_
        ]])

        # Предсказание
        prediction_idx = self.model.predict(X)[0]
        prediction = self.label_encoder.inverse_transform([prediction_idx])[0]
        probabilities = self.model.predict_proba(X)[0]

        # Получение вероятностей для каждой категории
        category_probs = {
            cat: float(prob)
            for cat, prob in zip(self.label_encoder.classes_, probabilities)
        }

        # Сортировка по убыванию вероятности
        sorted_categories = sorted(
            category_probs.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return {
            "primary": prediction,
            "probabilities": category_probs,
            "ranking": sorted_categories
        }