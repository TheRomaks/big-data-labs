import streamlit as st
import pandas as pd
import requests
import seaborn as sns
import matplotlib.pyplot as plt

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Pulsar API Explorer", layout="wide")

def get_api_status():
    try:
        response = requests.get(f"{API_URL}/")
        return response.json()
    except:
        return None


def predict_pulsar(model_name, features):
    try:
        response = requests.post(f"{API_URL}/predict/{model_name}", json=features)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Ошибка API: {response.json().get('detail')}")
            return None
    except Exception as e:
        st.error(f"Не удалось соединиться с сервером: {e}")
        return None


st.sidebar.title("Pulsar Control Center")
api_info = get_api_status()

if api_info:
    st.sidebar.success("API Status: Online")
    available_models = api_info.get("available_models", [])
    selected_model = st.sidebar.selectbox("Выберите модель на сервере", available_models)
else:
    st.sidebar.error("API Status: Offline")
    st.sidebar.warning("Запустите api_server.py для работы")
    st.stop()

page = st.sidebar.radio("Навигация", ["Инфо и EDA", "Предсказание через API"])


if page == "Инфо и EDA":
    st.title("Исследование пульсаров")
    st.markdown("""
    В этой работе интерфейс отделен от логики. 
    """)

    try:
        df = pd.read_csv('../../labs/lab3/utils/data/train.csv')

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Матрица корреляции")
            fig, ax = plt.subplots()
            sns.heatmap(df.drop(columns=['id', 'Id'], errors='ignore').corr(), cmap='coolwarm', ax=ax, annot=True)
            st.pyplot(fig)

        with col2:
            st.subheader("Распределение классов")
            fig2, ax2 = plt.subplots()
            sns.countplot(data=df, x='Class', palette='magma', ax=ax2)
            st.pyplot(fig2)

        st.dataframe(df.head(10))
    except:
        st.info("Файл данных не найден для EDA")

elif page == "Предсказание через API":
    st.title(f"Прогноз модели: {selected_model}")

    st.info("Введите параметры космического объекта. Данные будут отправлены на сервер FastAPI для обработки.")

    col_in1, col_in2 = st.columns(2)

    with col_in1:
        mean_profile = st.number_input("Mean of the integrated profile", value=121.0, format="%.4f")
        sd_profile = st.number_input("Standard deviation of the profile", value=48.0, format="%.4f")
        kurt_profile = st.number_input("Excess kurtosis of the profile", value=0.12, format="%.4f")
        skew_profile = st.number_input("Skewness of the profile", value=0.2, format="%.4f")

    with col_in2:
        mean_dmsnr = st.number_input("Mean of the DM-SNR curve", value=2.5, format="%.4f")
        sd_dmsnr = st.number_input("Standard deviation of the DM-SNR curve", value=18.0, format="%.4f")
        kurt_dmsnr = st.number_input("Excess kurtosis of the DM-SNR curve", value=9.0, format="%.4f")
        skew_dmsnr = st.number_input("Skewness of the DM-SNR curve", value=110.0, format="%.4f")

    if st.button("Отправить запрос на сервер", type="primary"):
        payload = {
            "mean_profile": mean_profile,
            "sd_profile": sd_profile,
            "kurt_profile": kurt_profile,
            "skew_profile": skew_profile,
            "mean_dmsnr": mean_dmsnr,
            "sd_dmsnr": sd_dmsnr,
            "kurt_dmsnr": kurt_dmsnr,
            "skew_dmsnr": skew_dmsnr
        }

        with st.spinner('Сервер обрабатывает данные...'):
            result = predict_pulsar(selected_model, payload)

        if result:
            st.divider()
            res_col1, res_col2 = st.columns(2)

            with res_col1:
                if result['prediction'] == 1:
                    st.error("### Результат: ПУЛЬСАР")
                else:
                    st.success("### Результат: НЕ ПУЛЬСАР")


            st.json(result)