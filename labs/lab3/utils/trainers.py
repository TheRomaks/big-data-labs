from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from typing import Dict

def get_models() -> Dict[str, object]:
    return {
        'KNN': KNeighborsClassifier(n_neighbors=7),  # увеличено с 5 → 7
        'Logistic Regression': LogisticRegression(
            C=0.1,  # сильнее регуляризация (по умолчанию 1.0)
            max_iter=1000,
            random_state=42,
            class_weight='balanced'  # компенсация дисбаланса
        ),
        'LinearSVC': LinearSVC(
            C=0.1,  # сильнее регуляризация
            max_iter=10000,
            random_state=42,
            class_weight='balanced'
        )
    }
