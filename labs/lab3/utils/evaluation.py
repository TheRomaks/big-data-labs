import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
)
from typing import Tuple, Dict

def evaluate_model(
        model, X_train, X_test, y_train, y_test, model_name: str, save_dir: str,
        is_multiclass: bool = False
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    if hasattr(model, "decision_function") and not hasattr(model, "predict_proba"):
        model = CalibratedClassifierCV(model, method='sigmoid', cv=3)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if is_multiclass:
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'F1-score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'F1-macro': f1_score(y_test, y_pred, average='macro')
        }
        y_proba = None
    else:
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
    if is_multiclass:
        labels = sorted(y_test.unique())
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.title(f'Матрица ошибок: {model_name} (мультикласс)')
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Не пульсар', 'Пульсар'],
                    yticklabels=['Не пульсар', 'Пульсар'])
        plt.title(f'Матрица ошибок: {model_name}')
    plt.ylabel('Истинный класс')
    plt.xlabel('Предсказанный класс')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrix_{model_name.replace(" ", "_")}.png',
                dpi=150, bbox_inches='tight')
    plt.show()

    if not is_multiclass and y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f'AUC = {metrics["ROC-AUC"]:.3f}')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC-кривая: {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/roc_curve_{model_name.replace(" ", "_")}.png',
                    dpi=150, bbox_inches='tight')
        plt.show()

    return metrics, cm, y_pred, y_proba