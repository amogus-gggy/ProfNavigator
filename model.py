import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class SurveyModel:
    """ML модель для классификации результатов опроса"""

    
    DEFAULT_CATEGORIES = [
        "analytical", "social", "creative",
        "managerial", "practical", "research",
        "technical", "artistic", "entrepreneurial", "scientific"
    ]

    N_QUESTIONS = 15  # количество вопросов в опросе

    def __init__(self, model_path: str = "model_artifact.pkl"):
        self.model_path = model_path
        self.model: Optional[Any] = None
        self.label_encoder: Optional[Any] = None
        self.categories: List[str] = self.DEFAULT_CATEGORIES.copy()
        self.is_fitted = False
        self.metrics: Dict[str, Any] = {}

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
                self.metrics = artifact.get("metrics", {})
                self.is_fitted = True
                
                # Информация о модели
                model_info = self._get_model_info()
                print(f"✅ Модель загружена из {self.model_path}")
                print(f"   Тип: {model_info['type']}")
                print(f"   Классы: {list(self.label_encoder.classes_)}")
                if self.metrics.get('validation_accuracy'):
                    print(f"   Validation accuracy: {self.metrics['validation_accuracy']:.4f}")
                if self.metrics.get('cv_mean'):
                    print(f"   CV stability: {self.metrics['cv_mean']:.4f} (+/- {self.metrics['cv_std'] * 2:.4f})")
                if self.metrics.get('calibrated'):
                    print(f"   🔹 Калибрована")
                    
            except Exception as e:
                print(f"⚠️ Ошибка загрузки модели: {e}")
                self._init_default_model()
        else:
            print(f"⚠️ Модель не найдена в {self.model_path}, использую default модель")
            self._init_default_model()

    def _get_model_info(self) -> Dict[str, str]:
        """Получение информации о типе модели"""
        if self.model is None:
            return {"type": "None"}
        
        model_class = self.model.__class__.__name__
        
        # Проверка на обёртки
        if hasattr(self.model, 'estimator'):
            # CalibratedClassifierCV
            base_model = self.model.estimator
            return {
                "type": f"Calibrated({base_model.__class__.__name__})",
                "base": base_model.__class__.__name__
            }
        elif hasattr(self.model, 'estimators_'):
            # VotingClassifier
            return {
                "type": "VotingClassifier",
                "models": ", ".join([name for name, _ in self.model.estimators_])
            }
        elif hasattr(self.model, 'best_estimator_'):
            # GridSearchCV / RandomizedSearchCV
            return {
                "type": f"Best({self.model.best_estimator_.__class__.__name__})"
            }
        else:
            return {"type": model_class}

    def _init_default_model(self):
        """Инициализация простой модели для fallback"""
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.preprocessing import LabelEncoder

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.categories)

        # Простая синтетическая тренировка для 15 вопросов
        # Сумма признаков = 15
        X_train = []
        y_train = []

        # Чистые типы: один признак доминирует (10-15 баллов)
        for i, cat in enumerate(self.categories):
            for _ in range(5):
                features = [1] * len(self.categories)
                features[i] = random.randint(10, 15)
                # Нормализуем до 15
                total = sum(features)
                if total > 15:
                    scale = 15 / total
                    features = [max(0, int(f * scale)) for f in features]
                    features[i] = 15 - sum(features[:i]) - sum(features[i+1:])
                X_train.append(features)
                y_train.append(cat)

        # Смешанные типы
        for i, cat1 in enumerate(self.categories):
            for j, cat2 in enumerate(self.categories):
                if i < j:
                    features = [1] * len(self.categories)
                    features[i] = 8
                    features[j] = 5
                    # Нормализуем до 15
                    total = sum(features)
                    if total > 15:
                        scale = 15 / total
                        features = [max(0, int(f * scale)) for f in features]
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
        if not self.is_fitted or self.label_encoder is None:
            raise RuntimeError("Модель не обучена")

        # Вектор признаков в порядке классов label_encoder
        X = np.array([[
            answers.get(cat, 0) for cat in self.label_encoder.classes_
        ]])

        # Предсказание
        prediction_idx = self.model.predict(X)[0]
        prediction = self.label_encoder.inverse_transform([prediction_idx])[0]
        
        # Вероятности
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

        # Определение уверенности модели
        top_prob = sorted_categories[0][1]
        confidence = "high" if top_prob >= 0.7 else "medium" if top_prob >= 0.4 else "low"

        return {
            "primary": prediction,
            "confidence": confidence,
            "probabilities": category_probs,
            "ranking": sorted_categories
        }

    def predict_batch(self, answers_list: List[dict]) -> List[dict]:
        """
        Пакетное предсказание

        Args:
            answers_list: список dict с ответами

        Returns:
            список предсказаний
        """
        return [self.predict(answers) for answers in answers_list]

    def get_feature_importances(self) -> Optional[Dict[str, float]]:
        """
        Получение важности признаков
        
        Returns:
            dict {category: importance} или None если модель не поддерживает
        """
        if self.model is None:
            return None

        # Распаковка обёрток
        base_model = self.model
        if hasattr(self.model, 'estimator'):
            base_model = self.model.estimator
        elif hasattr(self.model, 'best_estimator_'):
            base_model = self.model.best_estimator_

        if hasattr(base_model, 'feature_importances_'):
            importances = base_model.feature_importances_
            return {
                cat: float(imp)
                for cat, imp in zip(self.label_encoder.classes_, importances)
            }
        
        return None

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Получение сводной информации о модели
        
        Returns:
            dict с информацией о модели
        """
        summary = {
            "is_fitted": self.is_fitted,
            "model_type": self._get_model_info()["type"],
            "categories": self.categories,
            "n_classes": len(self.categories),
            "metrics": self.metrics
        }

        
        importances = self.get_feature_importances()
        if importances:
            sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            summary["feature_importances"] = sorted_imp
            summary["top_features"] = [cat for cat, _ in sorted_imp[:3]]
            summary["low_importance_features"] = [
                cat for cat, imp in sorted_imp if imp < 0.1
            ]

        return summary

    def get_recommendations(self, answers: dict) -> List[str]:
        """
        Получение рекомендаций на основе предсказания
        
        Args:
            answers: dict с ответами пользователя
        
        Returns:
            список рекомендаций
        """
        result = self.predict(answers)
        recommendations = []

        primary = result["primary"]
        ranking = result["ranking"]

        # Рекомендация основной категории
        recommendations.append(
            f"Ваш основной профиль: **{primary}** "
            f"(вероятность: {result['probabilities'][primary]:.1%})"
        )

        # Рекомендация по развитию
        if len(ranking) >= 2:
            second = ranking[1][0]
            recommendations.append(
                f"Дополнительный профиль: **{second}** "
                f"(вероятность: {ranking[1][1]:.1%})"
            )

        # Анализ уверенности
        if result["confidence"] == "low":
            recommendations.append(
                "Результат неоднозначный. Рекомендуется пройти расширенное тестирование."
            )

        return recommendations
