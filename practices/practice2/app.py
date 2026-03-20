import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score

st.set_page_config(page_title="Анализ Пульсаров", layout="wide")

@st.cache_data
def load_raw_data():
    df = pd.read_csv('data/train.csv')
    df = df.drop(columns=['id', 'Id'], errors='ignore')
    return df


@st.cache_resource
def load_models_pack():
    with open('models_pack.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def run_cross_validation(_model, X, y):
    scores = cross_val_score(_model, X, y, cv=5, scoring='f1')
    return scores.mean(), scores.std()


df = load_raw_data()
data_pack = load_models_pack()
models_dict = data_pack['models']
scaler = data_pack['scaler']

st.sidebar.header("Управление")
selected_model_name = st.sidebar.selectbox("Выберите модель", list(models_dict.keys()))
page = st.sidebar.radio("Навигация",
                        ["Общее описание", "Результаты EDA", "Метрики и Переобучение", "Выборка и Предсказания"])

model = models_dict[selected_model_name]

def get_metrics(model_obj):
    X_test = data_pack['X_test']
    y_test = data_pack['y_test']
    y_pred = model_obj.predict(X_test)

    if hasattr(model_obj, "predict_proba"):
        y_proba = model_obj.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred

    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0,
        'y_test': y_test,
        'y_proba': y_proba,
        'cm': confusion_matrix(y_test, y_pred)
    }

metrics = get_metrics(model)


if page == "Общее описание":
    st.title("Анализ характеристик пульсаров")
    st.markdown("""
    **Цель:** Определить, является ли объект пульсаром на основе физических показателей.
    """)
    st.write("### Исходные данные (первые 5 строк):")
    st.dataframe(df.head())
    st.info(f"Статистика: {df.shape[0]} объектов, {df.shape[1] - 1} признаков.")

elif page == "Результаты EDA":
    st.title("Исследовательский анализ (EDA)")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Баланс классов")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='Class', palette='viridis', ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Корреляция признаков")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        st.pyplot(fig)

elif page == "Метрики и Переобучение":
    st.title(f"Оценка модели: {selected_model_name}")

    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy (Тест)", f"{metrics['Accuracy']:.3f}")
    m2.metric("F1-Score (Тест)", f"{metrics['F1-Score']:.3f}")
    m3.metric("ROC-AUC", f"{metrics['ROC-AUC']:.3f}")

    train_acc = model.score(data_pack['X_train'], data_pack['y_train'])
    test_acc = metrics['Accuracy']
    gap = train_acc - test_acc

    with st.spinner('Выполняется кросс-валидация...'):
        cv_mean, cv_std = run_cross_validation(model, data_pack['X_train'], data_pack['y_train'])

    c1, c2 = st.columns(2)
    with c1:
        st.write(f"Точность на обучении: **{train_acc:.3f}**")
        st.write(f"Точность на тесте: **{test_acc:.3f}**")
        st.write(f"Разрыв (Gap): **{gap:.3f}**")

    with c2:
        st.write(f"Средний F1 (Кросс-валидация): **{cv_mean:.3f}**")
        st.write(f"Стандартное отклонение: **±{cv_std:.4f}**")

    if gap > 0.1 or cv_std > 0.05:
        st.error("Признаки переобучения: либо большой разрыв между выборками, либо высокая нестабильность на кросс-валидации.")
    else:
        st.success("Модель стабильна: показатели кросс-валидации и теста подтверждают хорошую обобщающую способность.")

    st.info(f"Стабильность модели: стандартное отклонение {cv_std:.4f} говорит о том, что результат не зависит от того, какие именно данные попали в обучение.")

    st.subheader("Матрица ошибок (Confusion Matrix)")
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(metrics['cm'], annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    st.pyplot(fig_cm)

elif page == "Выборка и Предсказания":
    st.title("Тестирование модели")

    st.write("### Введите параметры объекта:")
    features = df.drop('Class', axis=1).columns
    user_input = {}

    col_in1, col_in2 = st.columns(2)
    for i, feat in enumerate(features):
        with col_in1 if i % 2 == 0 else col_in2:
            user_input[feat] = st.number_input(f"{feat}",
                                               value=float(df[feat].mean()),
                                               format="%.4f")

    if st.button("Выполнить классификацию", type="primary"):
        input_df = pd.DataFrame([user_input])
        input_scaled = scaler.transform(input_df)

        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1] if hasattr(model, "predict_proba") else None

        st.divider()
        if pred == 1:
            st.error(f"### Результат: ПУЛЬСАР")
        else:
            st.success(f"### Результат: НЕ ПУЛЬСАР")

        if prob is not None:
            st.write(f"Вероятность принадлежности к пульсарам: **{prob:.2%}**")
            st.progress(prob)