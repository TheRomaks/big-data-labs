from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from typing import Dict

def get_models() -> Dict[str, object]:
    return {
        'KNN': KNeighborsClassifier(n_neighbors=7),
        'Logistic Regression': LogisticRegression(
            C=0.1,
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        ),
        'LinearSVC': LinearSVC(
            C=0.1,
            max_iter=10000,
            random_state=42,
            class_weight='balanced'
        )
    }
