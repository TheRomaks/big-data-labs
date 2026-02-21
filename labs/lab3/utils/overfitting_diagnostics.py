import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def train_test_gap(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    results = {
        "F1_train": f1_score(y_train, y_train_pred),
        "F1_test": f1_score(y_test, y_test_pred),
        "Precision_train": precision_score(y_train, y_train_pred),
        "Precision_test": precision_score(y_test, y_test_pred),
        "Recall_train": recall_score(y_train, y_train_pred),
        "Recall_test": recall_score(y_test, y_test_pred),
    }

    if hasattr(model, "predict_proba"):
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]

        results["ROC_train"] = roc_auc_score(y_train, y_train_proba)
        results["ROC_test"] = roc_auc_score(y_test, y_test_proba)

    return results


def cross_validation_scores(model, X, y, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    f1 = cross_val_score(model, X, y, scoring="f1", cv=skf)
    roc = cross_val_score(model, X, y, scoring="roc_auc", cv=skf)

    return {
        "F1_mean": f1.mean(),
        "F1_std": f1.std(),
        "ROC_mean": roc.mean(),
        "ROC_std": roc.std()
    }


def plot_learning_curve(model, X, y, model_name):
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="f1",
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    plt.figure(figsize=(7, 5))
    plt.plot(train_sizes, train_mean, label="Train F1")
    plt.plot(train_sizes, val_mean, label="Validation F1")
    plt.xlabel("Размер обучающей выборки")
    plt.ylabel("F1-score")
    plt.title(f"Learning Curve: {model_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()