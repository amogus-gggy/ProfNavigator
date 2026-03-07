import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


CATEGORY_NAMES_RU = {
    "analytical": "Аналитика",
    "social": "Социальная сфера",
    "creative": "Творчество",
    "managerial": "Менеджмент",
    "practical": "Практика",
    "research": "Исследования",
    "technical": "Техника",
    "artistic": "Искусство",
    "entrepreneurial": "Предпринимательство",
    "scientific": "Наука"
}


class SurveyModel:
    """ML модель для классификации результатов опроса"""

    DEFAULT_CATEGORIES = [
        "analytical", "social", "creative",
        "managerial", "practical", "research",
        "technical", "artistic", "entrepreneurial", "scientific"
    ]

    N_QUESTIONS = 15

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

                model_info = self._get_model_info()
                print(f"Модель загружена из {self.model_path}")
                print(f"   Тип: {model_info['type']}")
                print(f"   Классы: {list(self.label_encoder.classes_)}")
                if self.metrics.get('validation_accuracy'):
                    print(f"   Validation accuracy: {self.metrics['validation_accuracy']:.4f}")
                if self.metrics.get('cv_mean'):
                    print(f"   CV stability: {self.metrics['cv_mean']:.4f} (+/- {self.metrics['cv_std'] * 2:.4f})")
                if self.metrics.get('calibrated'):
                    print(f"   Калибрована")

            except Exception as e:
                print(f"Ошибка загрузки модели: {e}")
                self._init_default_model()
        else:
            print(f"Модель не найдена в {self.model_path}, использую default модель")
            self._init_default_model()

    def _get_model_info(self) -> Dict[str, str]:
        """Получение информации о типе модели"""
        if self.model is None:
            return {"type": "None"}

        model_class = self.model.__class__.__name__

        if hasattr(self.model, 'estimator'):
            base_model = self.model.estimator
            return {
                "type": f"Calibrated({base_model.__class__.__name__})",
                "base": base_model.__class__.__name__
            }
        elif hasattr(self.model, 'estimators_'):
            return {
                "type": "VotingClassifier",
                "models": ", ".join([name for name, _ in self.model.estimators_])
            }
        elif hasattr(self.model, 'best_estimator_'):
            return {
                "type": f"Best({self.model.best_estimator_.__class__.__name__})"
            }
        else:
            return {"type": model_class}

    def _init_default_model(self):
        """Инициализация простой модели для fallback (обучается на нормализованных признаках)"""
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.preprocessing import LabelEncoder

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.categories)

        # Алфавитный порядок совпадает с label_encoder.classes_
        sorted_cats = sorted(self.categories)
        n_cats = len(sorted_cats)
        X_train = []
        y_train = []

        # Чистые типы: один признак доминирует
        for cat in sorted_cats:
            i = sorted_cats.index(cat)
            for _ in range(5):
                raw = [1.0] * n_cats
                raw[i] = random.uniform(8, 12)
                total = sum(raw)
                features = [v / total for v in raw]
                X_train.append(features)
                y_train.append(cat)

        # Смешанные типы
        for i, cat1 in enumerate(sorted_cats):
            for j, cat2 in enumerate(sorted_cats):
                if i < j:
                    raw = [1.0] * n_cats
                    raw[i] = 8.0
                    raw[j] = 5.0
                    total = sum(raw)
                    features = [v / total for v in raw]
                    X_train.append(features)
                    y_train.append(cat1)

        self.model = DecisionTreeClassifier(random_state=42)
        self.model.fit(X_train, self.label_encoder.transform(y_train))
        self.is_fitted = True

    def _generate_reasoning(
        self,
        answer_distribution: Dict[str, float],
        prediction: str,
        confidence: str,
        sorted_categories: List[Tuple[str, float]]
    ) -> List[str]:
        """Генерация текстового объяснения результата"""
        reasoning = []

        # Топ категорий в ответах пользователя
        sorted_answers = sorted(answer_distribution.items(), key=lambda x: x[1], reverse=True)
        top_answers = [(cat, pct) for cat, pct in sorted_answers if pct > 0.05][:3]

        if top_answers:
            top_cat, top_pct = top_answers[0]
            name = CATEGORY_NAMES_RU.get(top_cat, top_cat)
            reasoning.append(
                f"В ваших ответах доминирует сфера «{name}» — {top_pct:.0%} от всех вариантов"
            )

        if len(top_answers) >= 2:
            second_cat, second_pct = top_answers[1]
            name = CATEGORY_NAMES_RU.get(second_cat, second_cat)
            reasoning.append(
                f"Заметен также интерес к «{name}» — {second_pct:.0%}"
            )

        # Объяснение уверенности модели
        top_prob = sorted_categories[0][1]
        if confidence == "high":
            reasoning.append(
                f"Высокая уверенность модели ({top_prob:.0%}): ваши ответы однозначно указывают на эту сферу"
            )
        elif confidence == "medium":
            if len(sorted_categories) > 1:
                second_name = CATEGORY_NAMES_RU.get(sorted_categories[1][0], sorted_categories[1][0])
                second_prob = sorted_categories[1][1]
                reasoning.append(
                    f"Умеренная уверенность ({top_prob:.0%}): близкая альтернатива — «{second_name}» ({second_prob:.0%})"
                )
            else:
                reasoning.append(f"Умеренная уверенность модели ({top_prob:.0%})")
        else:
            reasoning.append(
                f"Низкая уверенность ({top_prob:.0%}): интересы распределены равномерно — "
                "рассмотрите смежные профессии из нескольких сфер"
            )

        # Разнообразие ответов
        active_categories = sum(1 for pct in answer_distribution.values() if pct > 0.07)
        if active_categories >= 5:
            reasoning.append(
                "Ваши интересы охватывают много областей — это признак широкого кругозора и гибкости"
            )
        elif active_categories <= 2:
            pred_name = CATEGORY_NAMES_RU.get(prediction, prediction)
            reasoning.append(
                f"Ваши ответы сфокусированы: вы целенаправленно тяготеете к «{pred_name}»"
            )

        return reasoning

    def predict(self, answers: dict) -> dict:
        """
        Предсказание результата на основе ответов.

        Args:
            answers: dict {category: count} — количество ответов по категориям

        Returns:
            dict с результатом предсказания, распределением ответов и объяснением
        """
        if not self.is_fitted or self.label_encoder is None:
            raise RuntimeError("Модель не обучена")

        # Строим вектор признаков в порядке классов label_encoder
        raw = np.array([[
            answers.get(cat, 0) for cat in self.label_encoder.classes_
        ]], dtype=float)

        # Нормализуем до пропорций (сумма = 1) для стабильных предсказаний
        total = raw.sum()
        X = raw / total if total > 0 else raw

        # Распределение ответов пользователя (для объяснения)
        answer_distribution = {
            cat: float(raw[0, i] / total) if total > 0 else 0.0
            for i, cat in enumerate(self.label_encoder.classes_)
        }

        # Предсказание
        prediction_idx = self.model.predict(X)[0]
        prediction = self.label_encoder.inverse_transform([prediction_idx])[0]

        # Вероятности от модели
        raw_probs = self.model.predict_proba(X)[0]
        ml_probs = {
            cat: float(prob)
            for cat, prob in zip(self.label_encoder.classes_, raw_probs)
        }

        # Блендинг: смешиваем ML-вероятности с реальным распределением ответов.
        # Это позволяет вторичным категориям (напр. 20% managerial) получить
        # ненулевую вероятность, не ломая основное предсказание.
        BLEND_ML = 0.65      # вес ML
        BLEND_ANSWER = 0.35  # вес ответов пользователя
        category_probs = {
            cat: BLEND_ML * ml_probs[cat] + BLEND_ANSWER * answer_distribution.get(cat, 0.0)
            for cat in self.label_encoder.classes_
        }
        # Нормализуем до суммы 1
        total_p = sum(category_probs.values())
        if total_p > 0:
            category_probs = {cat: p / total_p for cat, p in category_probs.items()}

        # Пересчитываем предсказание по блендированным вероятностям
        prediction = max(category_probs, key=category_probs.get)

        # Сортировка по убыванию вероятности
        sorted_categories = sorted(
            category_probs.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Уверенность по блендированной топ-вероятности
        top_prob = sorted_categories[0][1]
        confidence = "high" if top_prob >= 0.6 else "medium" if top_prob >= 0.35 else "low"

        # Объяснение результата
        reasoning = self._generate_reasoning(
            answer_distribution, prediction, confidence, sorted_categories
        )

        return {
            "primary": prediction,
            "confidence": confidence,
            "probabilities": category_probs,
            "ranking": sorted_categories,
            "answer_distribution": answer_distribution,
            "reasoning": reasoning,
        }

    def predict_batch(self, answers_list: List[dict]) -> List[dict]:
        return [self.predict(answers) for answers in answers_list]

    def get_feature_importances(self) -> Optional[Dict[str, float]]:
        if self.model is None:
            return None

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
        result = self.predict(answers)
        recommendations = []

        primary = result["primary"]
        ranking = result["ranking"]

        recommendations.append(
            f"Ваш основной профиль: **{primary}** "
            f"(вероятность: {result['probabilities'][primary]:.1%})"
        )

        if len(ranking) >= 2:
            second = ranking[1][0]
            recommendations.append(
                f"Дополнительный профиль: **{second}** "
                f"(вероятность: {ranking[1][1]:.1%})"
            )

        if result["confidence"] == "low":
            recommendations.append(
                "Результат неоднозначный. Рекомендуется пройти расширенное тестирование."
            )

        return recommendations
