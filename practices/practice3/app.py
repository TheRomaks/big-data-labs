import streamlit as st
import pandas as pd
import requests
import seaborn as sns
import matplotlib.pyplot as plt


API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Pulsar AutoML Explorer", layout="wide")


def get_api_status():
    try:
        response = requests.get(f"{API_URL}/")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def predict(model_type, model_name, features):
    try:
        url = f"{API_URL}/predict/{model_type}/{model_name}"
        response = requests.post(url, json=features)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Ошибка сервера: {response.json().get('detail')}")
            return None
    except Exception as e:
        st.error(f"Ошибка соединения: {e}")
        return None


def run_automl_training():
    try:
        response = requests.post(f"{API_URL}/train/automl")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Не удалось запустить обучение: {e}")
        return None


st.sidebar.title("Pulsar Control Center")
status = get_api_status()

if status:
    st.sidebar.success("API Status: Online")
    manual_models = status.get("manual_models", [])
    automl_ready = status.get("automl_ready", False)
else:
    st.sidebar.error("API Status: Offline")
    st.sidebar.warning("Запустите api_server.py")
    st.stop()

page = st.sidebar.radio("Навигация", ["Инфо и EDA", "ML", "AutoML"])

if page == "Инфо и EDA":
    st.title("Исследование данных (EDA)")

    try:
        df = pd.read_csv('../../labs/lab3/utils/data/train.csv')

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Матрица корреляции")
            fig, ax = plt.subplots()
            sns.heatmap(df.drop(columns=['id', 'Id', 'Class'], errors='ignore').corr(),
                        cmap='coolwarm', annot=True, fmt=".2f", ax=ax)
            st.pyplot(fig)

        with col2:
            st.subheader("Баланс классов")
            fig2, ax2 = plt.subplots()
            sns.countplot(data=df, x='Class', palette='viridis', ax=ax2)
            st.pyplot(fig2)

        st.dataframe(df.head(10))
    except:
        st.warning("Файл данных для визуализации не найден, но API работает.")

elif page == "ML":
    st.title("Предсказание: Ручные модели")
    selected_model = st.selectbox("Выберите модель", manual_models)

    with st.form("manual_predict_form"):
        col_in1, col_in2 = st.columns(2)
        with col_in1:
            m_p = st.number_input("Mean profile", value=121.0, format="%.4f")
            s_p = st.number_input("SD profile", value=48.0, format="%.4f")
            k_p = st.number_input("Kurtosis profile", value=0.12, format="%.4f")
            sk_p = st.number_input("Skewness profile", value=0.2, format="%.4f")
        with col_in2:
            m_d = st.number_input("Mean DM-SNR", value=2.5, format="%.4f")
            s_d = st.number_input("SD DM-SNR", value=18.0, format="%.4f")
            k_d = st.number_input("Kurtosis DM-SNR", value=9.0, format="%.4f")
            sk_d = st.number_input("Skewness DM-SNR", value=110.0, format="%.4f")

        submit = st.form_submit_button("Выполнить классификацию", type="primary")

    if submit:
        payload = {
            "mean_profile": m_p, "sd_profile": s_p, "kurt_profile": k_p, "skew_profile": sk_p,
            "mean_dmsnr": m_d, "sd_dmsnr": s_d, "kurt_dmsnr": k_d, "skew_dmsnr": sk_d
        }
        res = predict("manual", selected_model, payload)
        if res:
            if res['prediction'] == 1:
                st.error("### Результат: ПУЛЬСАР")
            else:
                st.success("### Результат: НЕ ПУЛЬСАР")
            st.write(f"Вероятность: {res['probability']:.2%}")

elif page == "AutoML":
    st.title("Управление AutoML (FLAML)")

    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.subheader("Обучение")
        if st.button("Запустить переобучение AutoML", help="FLAML начнет поиск лучшей модели"):
            with st.spinner("Поиск лучшей модели (30 секунд)..."):
                results = run_automl_training()
                if results:
                    st.session_state['automl_results'] = results
                    st.success("Модель обучена и сохранена!")

    with col_b:
        st.subheader("Сравнение качества")
        if 'automl_results' in st.session_state:
            res = st.session_state['automl_results']

            comparison_data = [
                {"Метод": "Ручной (KNN)", "Лучший алгоритм": "KNeighbors", "F1-Score": "0.9442"},
                {"Метод": "AutoML (FLAML)", "Лучший алгоритм": res['best_model_name'],
                 "F1-Score": f"{1 - res['best_f1']:.4f}"}
            ]
            st.table(comparison_data)

            with st.expander("Посмотреть лучшие гиперпараметры"):
                st.json(res['best_config'])
        else:
            st.info("Нажмите кнопку слева, чтобы запустить AutoML и увидеть сравнение.")

    st.divider()

    st.subheader("Предсказание через лучшую AutoML модель")
    if not automl_ready and 'automl_results' not in st.session_state:
        st.warning("Сначала обучите AutoML модель.")
    else:
        c1, c2 = st.columns(2)
        f1 = c1.number_input("Mean profile ", value=140.0, key="a1")
        f2 = c1.number_input("SD profile ", value=55.0, key="a2")
        f3 = c1.number_input("Kurtosis profile ", value=0.05, key="a3")
        f4 = c1.number_input("Skewness profile ", value=0.1, key="a4")
        f5 = c2.number_input("Mean DM-SNR ", value=3.2, key="a5")
        f6 = c2.number_input("SD DM-SNR ", value=20.0, key="a6")
        f7 = c2.number_input("Kurtosis DM-SNR ", value=7.0, key="a7")
        f8 = c2.number_input("Skewness DM-SNR ", value=80.0, key="a8")

        if st.button("Предсказать через AutoML"):
            payload = {
                "mean_profile": f1, "sd_profile": f2, "kurt_profile": f3, "skew_profile": f4,
                "mean_dmsnr": f5, "sd_dmsnr": f6, "kurt_dmsnr": f7, "skew_dmsnr": f8
            }

            res = predict("automl", "best_automl", payload)
            if res:
                st.markdown(f"**Вердикт AutoML:** {'ПУЛЬСАР' if res['prediction'] == 1 else 'НЕ ПУЛЬСАР'}")
                st.write(f"Уверенность: {res['probability']:.2%}")