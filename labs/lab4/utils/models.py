import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
import joblib

from .preprocessing import calculate_metrics


def train_linear_regression(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred, "Линейная регрессия")

    print("\nКоэффициенты:")
    for i, col in enumerate(X_train.columns):
        print(f"  {col}: {model.coef_[i]:.4f}")
    print(f"  Intercept: {model.intercept_:.4f}")

    return {
        'model': model,
        'predictions': y_pred,
        'metrics': metrics,
        'name': 'Linear Regression'
    }


def train_lasso(X_train, y_train, X_test, y_test, alpha=0.1):
    model = Lasso(alpha=alpha, max_iter=10000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred, "Lasso Регрессия")

    print(f"\nAlpha: {alpha}")
    print(f"Количество нулевых коэффициентов: {np.sum(model.coef_ == 0)}")

    return {
        'model': model,
        'predictions': y_pred,
        'metrics': metrics,
        'name': 'Lasso'
    }


def train_elasticnet(X_train, y_train, X_test, y_test, alpha=0.1, l1_ratio=0.5):
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred, "ElasticNet Регрессия")

    print(f"\nAlpha: {alpha}, L1 Ratio: {l1_ratio}")
    print(f"Количество нулевых коэффициентов: {np.sum(model.coef_ == 0)}")

    return {
        'model': model,
        'predictions': y_pred,
        'metrics': metrics,
        'name': 'ElasticNet'
    }


def train_knn(X_train, y_train, X_test, y_test, k_range=range(1, 21)):
    k_scores = []
    for k in k_range:
        knn_temp = KNeighborsRegressor(n_neighbors=k)
        scores = cross_val_score(knn_temp, X_train, y_train, cv=5, scoring='r2')
        k_scores.append(scores.mean())

    optimal_k = k_range[np.argmax(k_scores)]
    print(f"\nОптимальное количество соседей (k): {optimal_k}")

    model = KNeighborsRegressor(n_neighbors=optimal_k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred, "KNN Регрессия")

    return {
        'model': model,
        'predictions': y_pred,
        'metrics': metrics,
        'name': 'KNN',
        'optimal_k': optimal_k,
        'k_scores': k_scores,
        'k_range': k_range
    }


def train_all_models(X_train, y_train, X_test, y_test):
    models = {}

    models['Linear Regression'] = train_linear_regression(X_train, y_train, X_test, y_test)
    models['Lasso'] = train_lasso(X_train, y_train, X_test, y_test)
    models['ElasticNet'] = train_elasticnet(X_train, y_train, X_test, y_test)
    models['KNN'] = train_knn(X_train, y_train, X_test, y_test)

    return models


def save_models(models, scaler, dataset_name):
    for name, model_data in models.items():
        filename = f"{name.lower().replace(' ', '_')}_{dataset_name}_model.joblib"
        joblib.dump(model_data['model'], filename)
        print(f"Модель сохранена: {filename}")

    joblib.dump(scaler, 'scaler.joblib')
    print(f"Скейлер сохранен: scaler.joblib")
