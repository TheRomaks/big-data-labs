import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

df = pd.read_csv('data/train.csv')
df = df.drop(columns=['id', 'Id'], errors='ignore')

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
if 'Class' in numeric_cols: numeric_cols.remove('Class')
df.fillna(df[numeric_cols].median(), inplace=True)

X = df.drop(['Class'], axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'KNN': KNeighborsClassifier(n_neighbors=7),
    'Logistic Regression': LogisticRegression(C=0.1, max_iter=1000, random_state=42, class_weight='balanced'),
    'LinearSVC': CalibratedClassifierCV(LinearSVC(C=0.1, max_iter=10000, random_state=42, class_weight='balanced'))
}

saved_models = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    saved_models[name] = model

with open('models_pack.pkl', 'wb') as f:
    pickle.dump({
        'models': saved_models,
        'scaler': scaler,
        'X_train': X_train_scaled,
        'y_train': y_train,
        'X_test': X_test_scaled,
        'y_test': y_test
    }, f)