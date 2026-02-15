import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
from typing import Tuple, Dict

def evaluate_model(
        model, X_train, X_test, y_train, y_test, model_name: str, save_dir: str
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.fit(X_train, y_train)

    # Для LinearSVC нужна калибровка для вероятностей
    if 'SVC' in model_name:
        calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=3)
        calibrated_model.fit(X_train, y_train)
        model = calibrated_model

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba)
    }

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Не пульсар', 'Пульсар'],
                yticklabels=['Не пульсар', 'Пульсар'])
    plt.title(f'Матрица ошибок: {model_name}')
    plt.ylabel('Истинный класс')
    plt.xlabel('Предсказанный класс')
    plt.savefig(f'{save_dir}/confusion_matrix_{model_name.replace(" ", "_")}.png',
                dpi=150, bbox_inches='tight')
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {metrics["ROC-AUC"]:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC-кривая: {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}/roc_curve_{model_name.replace(" ", "_")}.png',
                dpi=150, bbox_inches='tight')
    plt.show()

    return metrics, cm, y_pred, y_proba
