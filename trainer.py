import json
import argparse
import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
    ExtraTreesClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RandomizedSearchCV,
    StratifiedKFold,
    GridSearchCV
)
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV

# Попытка импорта продвинутых моделей
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


#категории
CATEGORIES = [
    "analytical", "social", "creative",
    "managerial", "practical", "research",
    "technical", "artistic", "entrepreneurial", "scientific"
]

N_QUESTIONS = 15


#параметры RandomizedSearchCV
RF_PARAM_DIST = {
    'n_estimators': [200, 300, 500, 700, 1000, 1500, 2000],
    'max_depth': [None, 10, 15, 20, 25, 30, 35, 40],
    'min_samples_split': [2, 3, 5, 7, 10],
    'min_samples_leaf': [1, 2, 3, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy']
}

ET_PARAM_DIST = {
    'n_estimators': [200, 300, 500, 700, 1000, 1500],
    'max_depth': [None, 10, 15, 20, 25, 30, 35],
    'min_samples_split': [2, 3, 5, 7],
    'min_samples_leaf': [1, 2, 3, 4],
    'max_features': ['sqrt', 'log2', None],
    'criterion': ['gini', 'entropy']
}

GB_PARAM_DIST = {
    'n_estimators': [200, 300, 500, 700, 1000, 1500],
    'max_depth': [3, 4, 5, 6, 7, 8, 9],
    'learning_rate': [0.01, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25],
    'min_samples_split': [2, 3, 5, 7],
    'min_samples_leaf': [1, 2, 3, 4],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'max_features': ['sqrt', 'log2', None]
}

#CatBoost
CATBOOST_PARAM_DIST = {
    'iterations': [500, 800, 1000, 1500, 2000, 2500],
    'depth': [4, 6, 8, 10, 12],
    'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2],
    'l2_leaf_reg': [1, 3, 5, 7, 9, 15, 20],
    'border_count': [32, 64, 128, 254],
    'bagging_temperature': [0, 0.5, 1, 2],
    'random_strength': [1, 2, 5, 10]
}

#LightGBM
LIGHTGBM_PARAM_DIST = {
    'n_estimators': [500, 1000, 1500, 2000],
    'max_depth': [-1, 10, 15, 20, 25, 30],
    'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1, 0.15],
    'num_leaves': [20, 31, 50, 70, 100, 150],
    'min_child_samples': [5, 10, 20, 30],
    'feature_fraction': [0.7, 0.8, 0.9, 1.0],
    'bagging_fraction': [0.7, 0.8, 0.9, 1.0],
    'bagging_freq': [3, 5, 7]
}

#XGBoost
XGBOOST_PARAM_DIST = {
    'n_estimators': [500, 1000, 1500, 2000],
    'max_depth': [4, 6, 8, 10, 12],
    'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1, 0.15],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3],
    'reg_alpha': [0, 0.1, 1, 10],
    'reg_lambda': [1, 5, 10]
}


def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray]:
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


def create_model(model_type: str, **kwargs) -> Any:
    """Создание модели по типу"""
    if model_type == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=kwargs.get('max_depth', 12),
            min_samples_split=kwargs.get('min_samples_split', 2),
            min_samples_leaf=kwargs.get('min_samples_leaf', 1),
            criterion='entropy',
            random_state=42
        )
    elif model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 1000),
            max_depth=kwargs.get('max_depth', None),
            min_samples_split=kwargs.get('min_samples_split', 2),
            min_samples_leaf=kwargs.get('min_samples_leaf', 1),
            max_features='sqrt',
            bootstrap=True,
            criterion='entropy',
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
    elif model_type == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=kwargs.get('n_estimators', 1000),
            max_depth=kwargs.get('max_depth', None),
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            criterion='entropy',
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
    elif model_type == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=kwargs.get('n_estimators', 500),
            max_depth=kwargs.get('max_depth', 5),
            learning_rate=kwargs.get('learning_rate', 0.1),
            min_samples_split=2,
            min_samples_leaf=1,
            subsample=0.9,
            random_state=42
        )
    elif model_type == "lightgbm":
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM не установлен")
        return lgb.LGBMClassifier(
            n_estimators=kwargs.get('n_estimators', 1000),
            max_depth=kwargs.get('max_depth', -1),
            learning_rate=kwargs.get('learning_rate', 0.1),
            num_leaves=kwargs.get('num_leaves', 31),
            min_child_samples=kwargs.get('min_child_samples', 10),
            feature_fraction=0.9,
            bagging_fraction=0.9,
            bagging_freq=5,
            random_state=42,
            class_weight='balanced',
            verbose=-1
        )
    elif model_type == "catboost":
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost не установлен")
        return cb.CatBoostClassifier(
            iterations=kwargs.get('iterations', 1000),
            depth=kwargs.get('depth', 8),
            learning_rate=kwargs.get('learning_rate', 0.1),
            l2_leaf_reg=kwargs.get('l2_leaf_reg', 3),
            border_count=kwargs.get('border_count', 254),
            bagging_temperature=kwargs.get('bagging_temperature', 0),
            random_strength=kwargs.get('random_strength', 1),
            random_seed=42,
            verbose=0,
            loss_function='MultiClass',
            eval_metric='Accuracy'
        )
    elif model_type == "xgboost":
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost не установлен")
        return xgb.XGBClassifier(
            n_estimators=kwargs.get('n_estimators', 1000),
            max_depth=kwargs.get('max_depth', 6),
            learning_rate=kwargs.get('learning_rate', 0.1),
            subsample=0.9,
            colsample_bytree=0.9,
            gamma=0,
            reg_alpha=0,
            reg_lambda=1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
    elif model_type == "neural_network":
        return MLPClassifier(
            hidden_layer_sizes=kwargs.get('hidden_layer_sizes', (256, 128, 64)),
            activation='relu',
            solver='adam',
            alpha=kwargs.get('alpha', 0.0001),
            max_iter=3000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            batch_size='auto'
        )
    elif model_type == "voting":
        models = kwargs.get('models', [])
        if not models:
            #voting
            models = [
                ("rf", RandomForestClassifier(
                    n_estimators=1000, max_depth=None, min_samples_leaf=1,
                    criterion='entropy', random_state=42, n_jobs=-1
                )),
                ("et", ExtraTreesClassifier(
                    n_estimators=1000, max_depth=None, min_samples_leaf=1,
                    criterion='entropy', random_state=42, n_jobs=-1
                )),
                ("gb", GradientBoostingClassifier(
                    n_estimators=500, max_depth=5, learning_rate=0.1, random_state=42
                ))
            ]
        return VotingClassifier(models, voting='soft')
    elif model_type == "stacking":
        
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)),
            ('et', ExtraTreesClassifier(n_estimators=500, random_state=42, n_jobs=-1)),
            ('gb', GradientBoostingClassifier(n_estimators=300, random_state=42))
        ]
        final_estimator = RandomForestClassifier(n_estimators=200, random_state=42)
        return StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")


def optimize_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "random_forest",
    n_iter: int = 100,
    cv: int = 10
) -> Tuple[Any, Dict[str, Any]]:
    """
    Оптимизация гиперпараметров с помощью RandomizedSearchCV
    
    Returns:
        best_model, best_params
    очень долгое.
    """
    print(f"\n🔍 Оптимизация гиперпараметров для {model_type}...")
    print(f"   Итераций: {n_iter}, CV folds: {cv}")
    
    base_model = create_model(model_type)
    
    #Выбор параметров в зависимости от модели
    if model_type == "random_forest":
        param_dist = RF_PARAM_DIST
    elif model_type == "extra_trees":
        param_dist = ET_PARAM_DIST
    elif model_type == "gradient_boosting":
        param_dist = GB_PARAM_DIST
    elif model_type == "catboost":
        param_dist = CATBOOST_PARAM_DIST
        print(f"   Размер пространства параметров CatBoost: ~{np.prod([len(v) for v in CATBOOST_PARAM_DIST.values()])} комбинаций")
    elif model_type == "lightgbm":
        param_dist = LIGHTGBM_PARAM_DIST
    elif model_type == "xgboost":
        param_dist = XGBOOST_PARAM_DIST
    elif model_type == "neural_network":
        param_dist = {
            'hidden_layer_sizes': [(128, 64), (256, 128), (256, 128, 64), (512, 256, 128), (512, 256, 128, 64)],
            'alpha': [0.00001, 0.0001, 0.001, 0.01],
            'learning_rate_init': [0.0001, 0.001, 0.01]
        }
    else:
        param_dist = {}
    
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1,
        refit=True
    )
    
    search.fit(X, y)
    
    print(f"\n✅ Лучшие параметры:")
    for param, value in search.best_params_.items():
        print(f"   {param}: {value}")
    print(f"\nЛучшая CV accuracy: {search.best_score_:.4f}")
    
    return search.best_estimator_, search.best_params_


def evaluate_model_stability(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 10
) -> Dict[str, float]:
    """
    Оценка стабильности модели через кросс-валидацию
    
    Returns:
        dict со статистикой стабильности
    """
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    
    return {
        'mean': float(cv_scores.mean()),
        'std': float(cv_scores.std()),
        'min': float(cv_scores.min()),
        'max': float(cv_scores.max()),
        'scores': cv_scores.tolist()
    }


def print_feature_importances(model: Any, categories: list):
    """Вывод важности признаков"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        
        print("\n📊 Важность признаков:")
        for idx in sorted_idx:
            print(f"   {categories[idx]:15} {importances[idx]:.4f}")
        
        # Рекомендации по сокращению вопросов
        low_importance = [
            categories[i] for i in sorted_idx
            if importances[i] < 0.05
        ]
        if low_importance:
            print(f"\n💡 Вопросы с низкой важностью: {', '.join(low_importance)}")
    
    elif hasattr(model, 'best_estimator_') and hasattr(model.best_estimator_, 'feature_importances_'):
        print_feature_importances(model.best_estimator_, categories)
    elif hasattr(model, 'estimators_'):
        # Для Voting/Stacking показываем первую модель
        print_feature_importances(model.estimators_[0], categories)


def apply_calibration(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    method: str = "isotonic",
    cv: int = 5
) -> CalibratedClassifierCV:
    """
    Калибровка вероятностей модели
    
    Args:
        model: обученная модель
        X: признаки
        y: лейблы
        method: "sigmoid" или "isotonic"
        cv: количество folds
    
    Returns:
        откалиброванная модель
    """
    print(f"\n🎯 Калибровка вероятностей (method={method}, cv={cv})...")
    
    calibrated = CalibratedClassifierCV(model, method=method, cv=cv)
    calibrated.fit(X, y)
    
    print("✅ Калибровка завершена")
    return calibrated


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "random_forest",
    optimize: bool = False,
    calibrate: bool = False,
    n_iter: int = 100
) -> Tuple[Any, Dict[str, Any]]:
    """
    Обучение модели с опциональной оптимизацией и калибровкой
    
    Returns:
        model, metrics
    """
    # Разделение на train/validation с сохранением пропорций
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    print(f"\n📚 Размер train: {len(X_train)}, validation: {len(X_val)}")
    
    # Оптимизация или обычное обучение
    if optimize:
        model, best_params = optimize_hyperparameters(X_train, y_train, model_type, n_iter)
    else:
        model = create_model(model_type)
        best_params = {}
        print(f"\n🔧 Обучение модели: {model_type}")
        model.fit(X_train, y_train)

    # Валидация
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    print(f"\n✅ Точность на validation: {accuracy:.4f}")
    
    # Проверяем целевую метрику
    if accuracy >= 0.90:
        print("   🎯 ЦЕЛЬ ДОСТИГНУТА: точность >= 90%!")
    else:
        print(f"   ⚠️ Точность ниже целевой (90%). Рекомендуется использовать --optimize")
    
    print("\nClassification report:")
    print(classification_report(y_val, y_pred, zero_division=0))

    

    # Оценка стабильности
    stability = evaluate_model_stability(model, X, y, cv=10)
    print(f"\n📈 Стабильность модели (10-fold CV):")
    print(f"   Mean: {stability['mean']:.4f}")
    print(f"   Std:  {stability['std']:.4f}")
    print(f"   Min:  {stability['min']:.4f}")
    print(f"   Max:  {stability['max']:.4f}")
    
    
    
    #print_feature_importances(model, CATEGORIES)

    # Калибровка (опционально, может немного снизить точность)
    if calibrate:
        model = apply_calibration(model, X_train, y_train, method="isotonic")
        
        # Переоценка после калибровки
        y_pred_calib = model.predict(X_val)
        calib_accuracy = accuracy_score(y_val, y_pred_calib)
        print(f"\n📊 Точность после калибровки: {calib_accuracy:.4f}")

    metrics = {
        'validation_accuracy': float(accuracy),
        'cv_mean': stability['mean'],
        'cv_std': stability['std'],
        'best_params': best_params,
        'calibrated': calibrate
    }

    return model, metrics


def save_model(model: Any, label_encoder: LabelEncoder, output_path: str, metrics: Optional[Dict] = None):
    """Сохранение модели"""
    artifact = {
        "model": model,
        "label_encoder": label_encoder,
        "categories": CATEGORIES,
        "metrics": metrics or {}
    }

    with open(output_path, "wb") as f:
        #TODO: использовать не pickle
        pickle.dump(artifact, f)

    print(f"\n💾 Модель сохранена в {output_path}")
    print(f"   Классы: {list(label_encoder.classes_)}")
    if metrics:
        print(f"   Validation accuracy: {metrics.get('validation_accuracy', 'N/A'):.4f}")
        print(f"   CV mean: {metrics.get('cv_mean', 'N/A'):.4f} (+/- {metrics.get('cv_std', 0) * 2:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description="Trainer для модели профориентации (улучшенная версия)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python trainer.py --dataset dataset.json --model-type random_forest
  python trainer.py --dataset dataset.json --optimize --n-iter 200
  python trainer.py --dataset dataset.json --model-type catboost --optimize --n-iter 300
  python trainer.py --dataset dataset.json --model-type extra_trees --optimize --n-iter 200
        """
    )
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

    #из стабильных пока только xgboost, random_forest, lightgdm
    parser.add_argument(
        "--model-type",
        type=str,
        default="random_forest",
        choices=[
            "decision_tree", "random_forest", "extra_trees", "gradient_boosting",
            "lightgbm", "catboost", "xgboost", "neural_network", "voting", "stacking"
        ],
        help="Тип модели для обучения"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Использовать RandomizedSearchCV для оптимизации гиперпараметров(долго работает, как минимум часов 15 на сложных моделях)"
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=100,
        help="Количество итераций для RandomizedSearchCV (по умолчанию 100)"
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Применить калибровку вероятностей (CalibratedClassifierCV)"
    )

    args = parser.parse_args()

    # Проверка существования датасета
    if not Path(args.dataset).exists():
        print(f"❌ Ошибка: файл {args.dataset} не найден")
        return 1

    # Проверка доступности моделей
    if args.model_type == "lightgbm" and not LIGHTGBM_AVAILABLE:
        print("❌ LightGBM не установлен. Установите: pip install lightgbm")
        return 1
    
    if args.model_type == "catboost" and not CATBOOST_AVAILABLE:
        print("❌ CatBoost не установлен. Установите: pip install catboost")
        return 1

    if args.model_type == "xgboost" and not XGBOOST_AVAILABLE:
        print("❌ XGBoost не установлен. Установите: pip install xgboost")
        return 1

    
    print(f"📂 Загрузка датасета из {args.dataset}")
    X, y = load_dataset(args.dataset)
    print(f"   Загружено {len(X)} примеров")
    print(f"   Классы: {np.unique(y)}")

    
    label_encoder = LabelEncoder()
    label_encoder.fit(CATEGORIES)
    y_encoded = label_encoder.transform(y)

    
    model, metrics = train_model(
        X, y_encoded,
        model_type=args.model_type,
        optimize=args.optimize,
        calibrate=args.calibrate,
        n_iter=args.n_iter
    )

    
    save_model(model, label_encoder, args.output, metrics)

    print("\n✅ Обучение завершено!")
    return 0


if __name__ == "__main__":
    exit(main())