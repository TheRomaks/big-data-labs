from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from typing import Dict

def get_models() -> Dict[str, object]:
    return {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'LinearSVC': LinearSVC(max_iter=10000, random_state=42)
    }
